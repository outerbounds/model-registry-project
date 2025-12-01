"""
Cloud-agnostic storage layer for Parquet datasets.

This module provides a "fast bakery" pattern for storing and loading
market snapshots as Parquet files in cloud storage.

Storage Layout:
    {datastore_root}/projects/{project}/branches/{branch}/snapshots/
        dt=2025-11-30/
            hour=10/
                snapshot_185268.parquet
            hour=11/
                snapshot_185269.parquet

Dataset Layout (accumulated):
    {datastore_root}/projects/{project}/branches/{branch}/datasets/
        training_dataset/
            v_185270/
                part-0000.parquet
                part-0001.parquet
                _metadata.json
        eval_holdout/
            v_185270/
                part-0000.parquet
                _metadata.json

References:
- https://outerbounds.com/blog/metaflow-fast-data
- https://github.com/outerbounds/fast-data-blog/blob/main/table_loader.py
"""

import json
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor


def get_datastore_root() -> Tuple[str, str]:
    """
    Get the datastore root path and cloud provider.

    Returns:
        Tuple of (root_path, provider) where provider is 's3', 'gs', 'azure', or 'local'
    """
    import metaflow.metaflow_config as config

    # Check cloud providers in order of preference
    if hasattr(config, 'DATASTORE_SYSROOT_S3') and config.DATASTORE_SYSROOT_S3:
        return config.DATASTORE_SYSROOT_S3, 's3'
    if hasattr(config, 'DATASTORE_SYSROOT_GS') and config.DATASTORE_SYSROOT_GS:
        return config.DATASTORE_SYSROOT_GS, 'gs'
    if hasattr(config, 'DATASTORE_SYSROOT_AZURE') and config.DATASTORE_SYSROOT_AZURE:
        return config.DATASTORE_SYSROOT_AZURE, 'azure'

    # Fallback to local
    local_root = getattr(config, 'DATASTORE_SYSROOT_LOCAL', '/tmp/metaflow')
    return local_root, 'local'


def get_project_storage_path(project: str, branch: str, asset_type: str = "snapshots") -> str:
    """
    Get the storage path for a project's data.

    Args:
        project: Project name
        branch: Branch name (should be sanitized)
        asset_type: 'snapshots' or 'datasets'

    Returns:
        Full path like s3://bucket/metaflow/projects/crypto_anomaly/branches/main/snapshots/
    """
    root, provider = get_datastore_root()
    return f"{root.rstrip('/')}/projects/{project}/branches/{branch}/{asset_type}"


@dataclass
class SnapshotMetadata:
    """Metadata for a stored snapshot."""
    timestamp: str
    n_samples: int
    n_features: int
    feature_names: List[str]
    source_flow: str
    source_run_id: str
    path: str


@dataclass
class DatasetMetadata:
    """Metadata for an accumulated dataset."""
    created_at: str
    snapshot_count: int
    snapshot_range: Tuple[str, str]  # (oldest_timestamp, newest_timestamp)
    total_samples: int
    n_features: int
    feature_names: List[str]
    builder_flow: str
    builder_run_id: str
    paths: List[str]  # List of parquet file paths


def snapshot_to_parquet_bytes(feature_set) -> bytes:
    """
    Convert a FeatureSet to Parquet bytes.

    Args:
        feature_set: FeatureSet dataclass from src.data

    Returns:
        Parquet file as bytes
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import io

    # Build columns: features + metadata
    columns = {}

    # Add feature columns
    for i, name in enumerate(feature_set.feature_names):
        columns[name] = [row[i] for row in feature_set.features]

    # Add coin metadata columns
    for key in ['coin_id', 'symbol', 'name', 'current_price', 'market_cap']:
        columns[key] = [info.get(key) for info in feature_set.coin_info]

    # Add timestamp column (same for all rows in this snapshot)
    columns['snapshot_timestamp'] = [feature_set.timestamp] * len(feature_set.features)

    # Convert to Arrow table
    table = pa.Table.from_pydict(columns)

    # Write to bytes
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression='snappy')
    return buffer.getvalue()


def parquet_bytes_to_table(data: bytes):
    """
    Convert Parquet bytes to PyArrow Table.

    Args:
        data: Parquet file bytes

    Returns:
        pyarrow.Table
    """
    import pyarrow.parquet as pq
    import io

    buffer = io.BytesIO(data)
    return pq.read_table(buffer)


class SnapshotStore:
    """
    Store and retrieve market snapshots as Parquet files.

    Uses Metaflow's S3 client for efficient cloud operations.
    """

    def __init__(self, project: str, branch: str):
        self.project = project
        self.branch = branch
        self.base_path = get_project_storage_path(project, branch, "snapshots")
        self.root, self.provider = get_datastore_root()

    def _get_snapshot_path(self, timestamp: str, run_id: str) -> str:
        """Generate path for a snapshot based on timestamp."""
        # Parse timestamp to get date/hour partitions
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        date_str = dt.strftime('%Y-%m-%d')
        hour_str = dt.strftime('%H')
        return f"{self.base_path}/dt={date_str}/hour={hour_str}/snapshot_{run_id}.parquet"

    def write_snapshot(self, feature_set, flow_name: str, run_id: str) -> SnapshotMetadata:
        """
        Write a FeatureSet to cloud storage as Parquet.

        Args:
            feature_set: FeatureSet from src.data
            flow_name: Name of the flow writing this snapshot
            run_id: Run ID of the flow

        Returns:
            SnapshotMetadata with path info
        """
        from metaflow import S3

        path = self._get_snapshot_path(feature_set.timestamp, run_id)
        parquet_bytes = snapshot_to_parquet_bytes(feature_set)

        if self.provider == 's3':
            with S3() as s3:
                s3.put(path, parquet_bytes)
        else:
            # Local fallback
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(parquet_bytes)

        return SnapshotMetadata(
            timestamp=feature_set.timestamp,
            n_samples=feature_set.n_samples,
            n_features=feature_set.n_features,
            feature_names=feature_set.feature_names,
            source_flow=flow_name,
            source_run_id=run_id,
            path=path,
        )

    def list_snapshots(self, limit: Optional[int] = None) -> List[str]:
        """
        List all snapshot paths, sorted by timestamp (newest first).

        Args:
            limit: Maximum number of snapshots to return

        Returns:
            List of snapshot paths
        """
        from metaflow import S3

        if self.provider == 's3':
            with S3() as s3:
                files = list(s3.list_recursive([self.base_path]))
                paths = [f.url for f in files if f.url.endswith('.parquet')]
        else:
            # Local fallback
            paths = []
            if os.path.exists(self.base_path):
                for root, _, files in os.walk(self.base_path):
                    for f in files:
                        if f.endswith('.parquet'):
                            paths.append(os.path.join(root, f))

        # Sort by path (which includes date/hour, so this sorts by time)
        paths.sort(reverse=True)

        if limit:
            paths = paths[:limit]

        return paths

    def load_snapshots(self, paths: List[str], num_threads: int = 8):
        """
        Load multiple snapshots efficiently using parallel reads.

        Based on fast-data pattern: https://github.com/outerbounds/fast-data-blog

        Args:
            paths: List of snapshot paths to load
            num_threads: Number of parallel threads for loading

        Returns:
            pyarrow.Table with all snapshots concatenated
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        from metaflow import S3

        if not paths:
            return None

        if self.provider == 's3':
            with S3() as s3:
                # Use get_many for efficient parallel download
                loaded = s3.get_many(paths)

                # Parallel Parquet decoding
                with ThreadPoolExecutor(max_workers=num_threads) as exe:
                    tables = list(exe.map(
                        lambda f: pq.read_table(f.path, use_threads=False),
                        loaded
                    ))
        else:
            # Local fallback
            with ThreadPoolExecutor(max_workers=num_threads) as exe:
                tables = list(exe.map(
                    lambda p: pq.read_table(p, use_threads=False),
                    paths
                ))

        # Concatenate all tables
        return pa.concat_tables(tables)

    def load_recent_snapshots(self, n: int, num_threads: int = 8):
        """
        Load the N most recent snapshots.

        Args:
            n: Number of recent snapshots to load
            num_threads: Number of parallel threads

        Returns:
            pyarrow.Table with concatenated snapshots
        """
        paths = self.list_snapshots(limit=n)
        if not paths:
            return None
        return self.load_snapshots(paths, num_threads)


class DatasetStore:
    """
    Store and retrieve accumulated datasets.

    Datasets are versioned collections of Parquet files with metadata.
    """

    def __init__(self, project: str, branch: str):
        self.project = project
        self.branch = branch
        self.base_path = get_project_storage_path(project, branch, "datasets")
        self.root, self.provider = get_datastore_root()

    def _get_dataset_path(self, name: str, version: str) -> str:
        """Get path for a dataset version."""
        return f"{self.base_path}/{name}/v_{version}"

    def write_dataset(
        self,
        name: str,
        table,
        version: str,
        metadata: Dict[str, Any],
        max_rows_per_file: int = 100000,
    ) -> DatasetMetadata:
        """
        Write a PyArrow table as a partitioned dataset.

        Args:
            name: Dataset name (e.g., 'training_dataset', 'eval_holdout')
            table: PyArrow Table to write
            version: Version string (typically run_id)
            metadata: Additional metadata dict
            max_rows_per_file: Max rows per Parquet file

        Returns:
            DatasetMetadata with paths
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        from metaflow import S3
        import io

        dataset_path = self._get_dataset_path(name, version)
        paths = []

        # Split table into parts if needed
        total_rows = table.num_rows
        n_parts = max(1, (total_rows + max_rows_per_file - 1) // max_rows_per_file)

        for i in range(n_parts):
            start = i * max_rows_per_file
            end = min((i + 1) * max_rows_per_file, total_rows)
            part_table = table.slice(start, end - start)

            part_path = f"{dataset_path}/part-{i:04d}.parquet"
            paths.append(part_path)

            # Write to bytes
            buffer = io.BytesIO()
            pq.write_table(part_table, buffer, compression='snappy')
            parquet_bytes = buffer.getvalue()

            if self.provider == 's3':
                with S3() as s3:
                    s3.put(part_path, parquet_bytes)
            else:
                os.makedirs(os.path.dirname(part_path), exist_ok=True)
                with open(part_path, 'wb') as f:
                    f.write(parquet_bytes)

        # Write metadata
        dataset_meta = DatasetMetadata(
            created_at=datetime.now(timezone.utc).isoformat(),
            snapshot_count=metadata.get('snapshot_count', 0),
            snapshot_range=metadata.get('snapshot_range', ('', '')),
            total_samples=table.num_rows,
            n_features=metadata.get('n_features', 0),
            feature_names=metadata.get('feature_names', []),
            builder_flow=metadata.get('builder_flow', ''),
            builder_run_id=metadata.get('builder_run_id', ''),
            paths=paths,
        )

        meta_path = f"{dataset_path}/_metadata.json"
        meta_bytes = json.dumps(asdict(dataset_meta), indent=2).encode()

        if self.provider == 's3':
            with S3() as s3:
                s3.put(meta_path, meta_bytes)
        else:
            with open(meta_path, 'w') as f:
                f.write(meta_bytes.decode())

        return dataset_meta

    def load_dataset(self, name: str, version: str = "latest", num_threads: int = 8):
        """
        Load a dataset by name and version.

        Args:
            name: Dataset name
            version: Version string or 'latest'
            num_threads: Parallel threads for loading

        Returns:
            Tuple of (pyarrow.Table, DatasetMetadata)
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        from metaflow import S3

        if version == "latest":
            version = self._get_latest_version(name)
            if not version:
                return None, None

        dataset_path = self._get_dataset_path(name, version)
        meta_path = f"{dataset_path}/_metadata.json"

        # Load metadata
        if self.provider == 's3':
            with S3() as s3:
                meta_obj = s3.get(meta_path)
                meta_dict = json.loads(meta_obj.text)
        else:
            with open(meta_path, 'r') as f:
                meta_dict = json.load(f)

        metadata = DatasetMetadata(**meta_dict)

        # Load parquet files
        if self.provider == 's3':
            with S3() as s3:
                loaded = s3.get_many(metadata.paths)
                with ThreadPoolExecutor(max_workers=num_threads) as exe:
                    tables = list(exe.map(
                        lambda f: pq.read_table(f.path, use_threads=False),
                        loaded
                    ))
        else:
            with ThreadPoolExecutor(max_workers=num_threads) as exe:
                tables = list(exe.map(
                    lambda p: pq.read_table(p, use_threads=False),
                    metadata.paths
                ))

        return pa.concat_tables(tables), metadata

    def _get_latest_version(self, name: str) -> Optional[str]:
        """Get the latest version of a dataset."""
        from metaflow import S3

        dataset_base = f"{self.base_path}/{name}"

        if self.provider == 's3':
            with S3() as s3:
                files = list(s3.list_recursive([dataset_base]))
                versions = set()
                for f in files:
                    # Extract version from path like .../v_12345/...
                    parts = f.url.split('/')
                    for p in parts:
                        if p.startswith('v_'):
                            versions.add(p[2:])  # Remove 'v_' prefix
                if versions:
                    return max(versions)  # Highest run_id is latest
        else:
            if os.path.exists(dataset_base):
                versions = [d[2:] for d in os.listdir(dataset_base) if d.startswith('v_')]
                if versions:
                    return max(versions)

        return None

    def list_versions(self, name: str) -> List[str]:
        """List all versions of a dataset."""
        from metaflow import S3

        dataset_base = f"{self.base_path}/{name}"
        versions = set()

        if self.provider == 's3':
            with S3() as s3:
                files = list(s3.list_recursive([dataset_base]))
                for f in files:
                    parts = f.url.split('/')
                    for p in parts:
                        if p.startswith('v_'):
                            versions.add(p[2:])
        else:
            if os.path.exists(dataset_base):
                versions = {d[2:] for d in os.listdir(dataset_base) if d.startswith('v_')}

        return sorted(versions, reverse=True)


def table_to_feature_set(table, feature_names: List[str]):
    """
    Convert a PyArrow table back to FeatureSet format.

    Args:
        table: PyArrow Table with feature columns
        feature_names: List of feature column names

    Returns:
        FeatureSet with features and coin_info compatible with model.py
    """
    from .data import FeatureSet

    # Convert to pandas for easier row-wise access
    df = table.to_pandas()

    # Extract features
    features = []
    for _, row in df.iterrows():
        feature_row = [float(row[name]) for name in feature_names]
        features.append(feature_row)

    # Extract coin info - must match the structure expected by get_anomalies()
    # which accesses: coin_info["symbol"], coin_info["price_change_24h"], etc.
    coin_info = []
    for _, row in df.iterrows():
        info = {
            "coin_id": row.get("coin_id", ""),
            "symbol": str(row.get("symbol", "")).upper(),
            "name": row.get("name", ""),
            "current_price": float(row.get("current_price", 0) or 0),
            "market_cap": float(row.get("market_cap", 0) or 0),
            # Map feature columns to expected keys for reporting
            "price_change_1h": float(row.get("price_change_pct_1h", 0) or 0),
            "price_change_24h": float(row.get("price_change_pct_24h", 0) or 0),
            "price_change_7d": float(row.get("price_change_pct_7d", 0) or 0),
            "volatility_7d": float(row.get("sparkline_volatility", 0) or 0),
        }
        coin_info.append(info)

    # Get timestamp range (use earliest and latest)
    timestamp = ""
    if 'snapshot_timestamp' in df.columns and len(df) > 0:
        timestamps = df['snapshot_timestamp'].unique()
        if len(timestamps) == 1:
            timestamp = str(timestamps[0])
        else:
            timestamp = f"{min(timestamps)} to {max(timestamps)}"

    return FeatureSet(
        features=features,
        coin_info=coin_info,
        feature_names=feature_names,
        timestamp=timestamp,
    )
