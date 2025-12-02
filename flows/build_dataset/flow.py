"""
Build training and evaluation datasets from accumulated snapshots.

This flow implements temporal feature engineering for crash prediction:
1. Lists recent market snapshot paths from Parquet storage
2. Loads ALL snapshots (no deduplication - we need temporal history)
3. Adds lag features and rolling statistics per coin
4. Splits by TIME (train on past, evaluate on "future")
5. Registers as DataAssets for downstream consumption

Dataset Strategy:
-----------------
Unlike typical ML where we deduplicate, we KEEP all temporal observations:

    Snapshot t=1: [BTC, ETH, ...] 100 coins
    Snapshot t=2: [BTC, ETH, ...] 100 coins
    ...
    Snapshot t=24: [BTC, ETH, ...] 100 coins

    Total: 24 × 100 = 2,400 rows (not deduplicated!)

Each row gets lag features computed from prior snapshots of the same coin:
    - price_change_pct_24h_lag_1h (what was this 1 hour ago?)
    - price_change_pct_24h_lag_6h
    - price_change_pct_24h_rolling_mean_24h

This enables the model to learn TEMPORAL patterns, not just static anomalies.

Time Split:
-----------
    |<------- Training (older) ------->|<-- Holdout (recent) -->|

    Training: Learn patterns from historical data
    Holdout: Test if model predicts on "future" data it hasn't seen

Triggering:
- @schedule(daily=True) - runs daily to build fresh datasets
- Publishes 'dataset_ready' event for TrainFlow

Usage:
    # Use last 168 hours (7 days), hold out last 24 hours
    python flow.py run --max_history_hours 168 --holdout_hours 24

    # Quick test with less data
    python flow.py run --max_history_hours 48 --holdout_hours 6
"""

from metaflow import step, card, Parameter, schedule
from obproject import ProjectFlow

# Import src at module level so Metaflow detects METAFLOW_PACKAGE_POLICY
# and includes it in the code package for remote execution
import src


@schedule(daily=True)
class BuildDatasetFlow(ProjectFlow):
    """
    Build training and evaluation datasets from accumulated snapshots.

    Preserves ALL temporal observations and adds lag/rolling features
    for crash prediction.
    """

    max_history_hours = Parameter(
        "max_history_hours",
        default=168,  # 7 days
        type=int,
        help="Maximum hours of history to include (rolling window)"
    )

    holdout_hours = Parameter(
        "holdout_hours",
        default=24,
        type=int,
        help="Hours of most recent data to hold out for evaluation"
    )

    min_snapshots_per_coin = Parameter(
        "min_snapshots_per_coin",
        default=6,
        type=int,
        help="Minimum snapshots needed per coin to compute lag features"
    )

    add_targets = Parameter(
        "add_targets",
        default=False,
        type=bool,
        help="Add future crash targets (for evaluation/backtesting only)"
    )

    @step
    def start(self):
        """Load all snapshots and prepare for temporal processing."""
        from src.storage import SnapshotStore

        print(f"Project: {self.prj.project}, Branch: {self.prj.branch}")
        print(f"\nBuilding dataset with temporal features")
        print(f"  Max history: {self.max_history_hours} hours")
        print(f"  Holdout: {self.holdout_hours} hours")
        print(f"  Min snapshots per coin: {self.min_snapshots_per_coin}")

        # List all available snapshots
        store = SnapshotStore(self.prj.project, self.prj.branch)
        all_paths = store.list_snapshots()
        print(f"\nFound {len(all_paths)} snapshots in storage")

        if len(all_paths) < self.min_snapshots_per_coin + 1:
            raise ValueError(
                f"Need at least {self.min_snapshots_per_coin + 1} snapshots "
                f"(got {len(all_paths)}). Run IngestFlow more times first."
            )

        # We'll load ALL snapshots and filter by timestamp later
        # This is more accurate than filtering by path count
        self.snapshot_paths = all_paths
        print(f"Will process {len(self.snapshot_paths)} snapshots")

        self.next(self.build_and_write)

    @card
    @step
    def build_and_write(self):
        """Load snapshots, add temporal features, split by time, write datasets."""
        from src.storage import SnapshotStore, DatasetStore, get_datastore_root
        from src.features import DatasetConfig, build_ml_dataset, get_feature_columns
        from metaflow import current
        from metaflow.cards import Markdown, Table
        from datetime import timedelta
        import pandas as pd

        snapshot_store = SnapshotStore(self.prj.project, self.prj.branch)
        dataset_store = DatasetStore(self.prj.project, self.prj.branch)

        root, provider = get_datastore_root()
        print(f"Storage: {provider} ({root})")

        # --- Load ALL snapshots ---
        print(f"\nLoading {len(self.snapshot_paths)} snapshots...")
        raw_table = snapshot_store.load_snapshots(self.snapshot_paths)
        print(f"  Loaded {raw_table.num_rows} raw rows (NOT deduplicated)")

        # --- Configure feature engineering ---
        config = DatasetConfig(
            max_history_hours=self.max_history_hours,
            min_snapshots_per_coin=self.min_snapshots_per_coin,
            lag_hours=[1, 6, 24],
            rolling_windows_hours=[6, 24],
        )

        # --- Build ML dataset with temporal features ---
        print(f"\nAdding temporal features...")
        ml_table, feature_cols = build_ml_dataset(
            raw_table,
            config=config,
            include_targets=self.add_targets,
        )
        print(f"  After feature engineering: {ml_table.num_rows} rows")
        print(f"  Feature columns: {len(feature_cols)}")

        # --- Split by TIME (not by path count) ---
        df = ml_table.to_pandas()
        df['snapshot_timestamp'] = pd.to_datetime(df['snapshot_timestamp'])

        max_ts = df['snapshot_timestamp'].max()
        holdout_cutoff = max_ts - timedelta(hours=self.holdout_hours)

        train_df = df[df['snapshot_timestamp'] < holdout_cutoff]
        holdout_df = df[df['snapshot_timestamp'] >= holdout_cutoff]

        print(f"\nTime-based split:")
        print(f"  Training: {len(train_df)} rows (before {holdout_cutoff})")
        print(f"  Holdout: {len(holdout_df)} rows (after {holdout_cutoff})")

        # --- Write Training Dataset ---
        import pyarrow as pa
        train_table = pa.Table.from_pandas(train_df, preserve_index=False)
        train_ts = train_df['snapshot_timestamp']
        train_ts_range = (str(train_ts.min()), str(train_ts.max())) if len(train_ts) > 0 else ('', '')

        # Count unique snapshots (not rows)
        train_snapshot_count = train_df['snapshot_timestamp'].nunique()

        self.train_meta = dataset_store.write_dataset(
            name="training_dataset",
            table=train_table,
            version=current.run_id,
            metadata={
                'snapshot_count': train_snapshot_count,
                'snapshot_range': train_ts_range,
                'n_features': len(feature_cols),
                'feature_names': feature_cols,
                'builder_flow': current.flow_name,
                'builder_run_id': current.run_id,
                'temporal_features': True,
                'deduplicated': False,  # Important: we keep all rows!
            }
        )
        print(f"  Wrote training_dataset: {self.train_meta.total_samples} samples")

        del train_table, train_df

        # --- Write Holdout Dataset ---
        holdout_table = pa.Table.from_pandas(holdout_df, preserve_index=False)
        holdout_ts = holdout_df['snapshot_timestamp']
        holdout_ts_range = (str(holdout_ts.min()), str(holdout_ts.max())) if len(holdout_ts) > 0 else ('', '')
        holdout_snapshot_count = holdout_df['snapshot_timestamp'].nunique()

        self.holdout_meta = dataset_store.write_dataset(
            name="eval_holdout",
            table=holdout_table,
            version=current.run_id,
            metadata={
                'snapshot_count': holdout_snapshot_count,
                'snapshot_range': holdout_ts_range,
                'n_features': len(feature_cols),
                'feature_names': feature_cols,
                'builder_flow': current.flow_name,
                'builder_run_id': current.run_id,
                'temporal_features': True,
                'deduplicated': False,
            }
        )
        print(f"  Wrote eval_holdout: {self.holdout_meta.total_samples} samples")

        # Store feature cols for downstream
        self.feature_cols = feature_cols

        del holdout_table, holdout_df

        # Build card
        current.card.append(Markdown("# Dataset Builder (Temporal)"))
        current.card.append(Markdown(f"**Run ID:** {current.run_id}"))
        current.card.append(Markdown(f"**Storage:** {provider}"))
        current.card.append(Markdown(f"**Temporal Features:** Yes (lag + rolling)"))

        current.card.append(Markdown("## Training Dataset"))
        current.card.append(Table([
            ["Rows (observations)", str(self.train_meta.total_samples)],
            ["Unique Snapshots", str(self.train_meta.snapshot_count)],
            ["From", train_ts_range[0][:19] if train_ts_range[0] else ""],
            ["To", train_ts_range[1][:19] if train_ts_range[1] else ""],
            ["Features", str(len(feature_cols))],
            ["Deduplicated", "No (kept all temporal data)"],
        ], headers=["Property", "Value"]))

        current.card.append(Markdown("## Evaluation Holdout"))
        current.card.append(Table([
            ["Rows (observations)", str(self.holdout_meta.total_samples)],
            ["Unique Snapshots", str(self.holdout_meta.snapshot_count)],
            ["From", holdout_ts_range[0][:19] if holdout_ts_range[0] else ""],
            ["To", holdout_ts_range[1][:19] if holdout_ts_range[1] else ""],
        ], headers=["Property", "Value"]))

        current.card.append(Markdown("## Feature Columns"))
        current.card.append(Markdown(f"```\n{chr(10).join(feature_cols[:20])}\n{'...' if len(feature_cols) > 20 else ''}\n```"))

        self.next(self.register)

    @step
    def register(self):
        """Register datasets as DataAssets and publish event."""
        from metaflow import current

        # Register training dataset
        self.prj.register_external_data(
            "training_dataset",
            blobs=self.train_meta.paths,
            kind="parquet_dataset",
            annotations={
                "total_samples": self.train_meta.total_samples,
                "snapshot_count": self.train_meta.snapshot_count,
                "snapshot_range_start": self.train_meta.snapshot_range[0],
                "snapshot_range_end": self.train_meta.snapshot_range[1],
                "n_features": len(self.feature_cols),
                "feature_names": ",".join(self.feature_cols[:20]),  # First 20 to avoid overflow
                "builder_flow": current.flow_name,
                "builder_run_id": current.run_id,
                "temporal_features": "True",
                "deduplicated": "False",
            },
            tags={"dataset_type": "training", "has_temporal_features": "true"},
            description=f"Training: {self.train_meta.total_samples} rows from {self.train_meta.snapshot_count} snapshots (temporal features)"
        )

        # Register holdout dataset
        self.prj.register_external_data(
            "eval_holdout",
            blobs=self.holdout_meta.paths,
            kind="parquet_dataset",
            annotations={
                "total_samples": self.holdout_meta.total_samples,
                "snapshot_count": self.holdout_meta.snapshot_count,
                "snapshot_range_start": self.holdout_meta.snapshot_range[0],
                "snapshot_range_end": self.holdout_meta.snapshot_range[1],
                "n_features": len(self.feature_cols),
                "feature_names": ",".join(self.feature_cols[:20]),
                "builder_flow": current.flow_name,
                "builder_run_id": current.run_id,
                "temporal_features": "True",
                "deduplicated": "False",
            },
            tags={"dataset_type": "eval_holdout", "has_temporal_features": "true"},
            description=f"Holdout: {self.holdout_meta.total_samples} rows from {self.holdout_meta.snapshot_count} snapshots (temporal features)"
        )

        # Publish event for TrainFlow
        self.prj.safe_publish_event("dataset_ready", payload={
            "training_samples": self.train_meta.total_samples,
            "holdout_samples": self.holdout_meta.total_samples,
            "n_features": len(self.feature_cols),
            "builder_run_id": current.run_id,
        })
        print("\nPublished 'dataset_ready' event")

        self.next(self.end)

    @step
    def end(self):
        """Summary."""
        print(f"\n{'='*50}")
        print("Dataset Build Complete (Temporal Features)")
        print(f"{'='*50}")
        print(f"\nTraining Dataset:")
        print(f"  Rows: {self.train_meta.total_samples} (NOT deduplicated)")
        print(f"  Snapshots: {self.train_meta.snapshot_count}")
        print(f"  Features: {len(self.feature_cols)} (includes lag + rolling)")
        tr = self.train_meta.snapshot_range
        print(f"  Range: {tr[0][:19] if tr[0] else 'N/A'} to {tr[1][:19] if tr[1] else 'N/A'}")
        print(f"\nEval Holdout:")
        print(f"  Rows: {self.holdout_meta.total_samples}")
        print(f"  Snapshots: {self.holdout_meta.snapshot_count}")
        hr = self.holdout_meta.snapshot_range
        print(f"  Range: {hr[0][:19] if hr[0] else 'N/A'} to {hr[1][:19] if hr[1] else 'N/A'}")
        print(f"\nTrainFlow will be triggered by 'dataset_ready' event")


if __name__ == "__main__":
    BuildDatasetFlow()
