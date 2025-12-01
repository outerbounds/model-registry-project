"""
Build training and evaluation datasets from accumulated snapshots.

This flow implements the "fast bakery" pattern:
1. Lists N recent market snapshot paths from Parquet storage
2. Splits paths by time (train = older, holdout = recent)
3. Loads, deduplicates, and writes each dataset in a single step
4. Registers as DataAssets for downstream consumption
5. Publishes 'dataset_ready' event to trigger TrainFlow

Dataset Split Strategy:
-----------------------
For temporal data, we split by snapshot paths (not loaded data) to prevent
memory issues with large datasets:

    |<------- Training Paths ------->|<-- Holdout Paths -->|
    |  snapshots 1 to N-holdout_n    |  last holdout_n     |
    |        (older data)            |   (recent data)     |

This ensures:
- No data leakage (eval uses "future" data)
- Memory efficient (load/write each split independently)
- Scalable (never holds full dataset in memory)

Triggering:
- @schedule(daily=True) - runs daily to build fresh datasets
- Publishes 'dataset_ready' event for TrainFlow

Usage:
    # Use last 24 snapshots, hold out last 3 for eval
    python flow.py run --n_snapshots 24 --holdout_snapshots 3

    # Quick test with fewer snapshots
    python flow.py run --n_snapshots 5 --holdout_snapshots 1
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

    Memory-efficient: splits by paths, loads each split independently,
    never holds full dataset in Metaflow artifacts.
    """

    n_snapshots = Parameter(
        "n_snapshots",
        default=24,
        type=int,
        help="Number of recent snapshots to include in dataset"
    )

    holdout_snapshots = Parameter(
        "holdout_snapshots",
        default=3,
        type=int,
        help="Number of most recent snapshots to hold out for evaluation"
    )

    deduplicate = Parameter(
        "deduplicate",
        default=True,
        type=bool,
        help="Deduplicate by coin_id within each snapshot window"
    )

    @step
    def start(self):
        """List snapshots and split paths by time."""
        from src.storage import SnapshotStore

        print(f"Project: {self.prj.project}, Branch: {self.prj.branch}")
        print(f"\nBuilding dataset from {self.n_snapshots} snapshots")
        print(f"Holding out {self.holdout_snapshots} for evaluation")

        # List available snapshots (newest first)
        store = SnapshotStore(self.prj.project, self.prj.branch)
        all_paths = store.list_snapshots()
        print(f"\nFound {len(all_paths)} snapshots in storage")

        if len(all_paths) < self.n_snapshots:
            print(f"[WARN] Only {len(all_paths)} snapshots available, using all")
            snapshot_paths = all_paths
        else:
            snapshot_paths = all_paths[:self.n_snapshots]

        if len(snapshot_paths) < self.holdout_snapshots + 1:
            raise ValueError(
                f"Need at least {self.holdout_snapshots + 1} snapshots "
                f"(got {len(snapshot_paths)})"
            )

        # Split by PATHS, not by loaded data
        # Paths are sorted newest-first, so holdout = first N paths
        self.holdout_paths = snapshot_paths[:self.holdout_snapshots]
        self.train_paths = snapshot_paths[self.holdout_snapshots:]

        print(f"\nPath split:")
        print(f"  Training: {len(self.train_paths)} snapshots (older)")
        print(f"  Holdout: {len(self.holdout_paths)} snapshots (recent)")

        for i, p in enumerate(self.train_paths[:2]):
            print(f"    Train[{i}]: {p.split('/')[-1]}")
        if len(self.train_paths) > 2:
            print(f"    ... and {len(self.train_paths) - 2} more")

        self.next(self.build_and_write)

    @card
    @step
    def build_and_write(self):
        """Load, deduplicate, and write datasets - single step, no artifacts."""
        from src.storage import SnapshotStore, DatasetStore, get_datastore_root
        from src.data import FEATURE_COLS
        from metaflow import current
        from metaflow.cards import Markdown, Table
        import pyarrow as pa

        snapshot_store = SnapshotStore(self.prj.project, self.prj.branch)
        dataset_store = DatasetStore(self.prj.project, self.prj.branch)

        root, provider = get_datastore_root()
        print(f"Storage: {provider} ({root})")

        # --- Process Training Dataset ---
        print(f"\nProcessing training dataset ({len(self.train_paths)} snapshots)...")
        train_table = snapshot_store.load_snapshots(self.train_paths)
        print(f"  Loaded {train_table.num_rows} rows")

        if self.deduplicate:
            train_table = self._deduplicate_table(train_table)
            print(f"  After dedup: {train_table.num_rows} rows")

        train_ts = train_table.column('snapshot_timestamp').to_pylist()
        train_ts_range = (min(train_ts), max(train_ts)) if train_ts else ('', '')

        self.train_meta = dataset_store.write_dataset(
            name="training_dataset",
            table=train_table,
            version=current.run_id,
            metadata={
                'snapshot_count': len(self.train_paths),
                'snapshot_range': train_ts_range,
                'n_features': len(FEATURE_COLS),
                'feature_names': FEATURE_COLS,
                'builder_flow': current.flow_name,
                'builder_run_id': current.run_id,
            }
        )
        print(f"  Wrote training_dataset: {self.train_meta.total_samples} samples")

        # Free memory before loading holdout
        del train_table

        # --- Process Holdout Dataset ---
        print(f"\nProcessing holdout dataset ({len(self.holdout_paths)} snapshots)...")
        holdout_table = snapshot_store.load_snapshots(self.holdout_paths)
        print(f"  Loaded {holdout_table.num_rows} rows")

        if self.deduplicate:
            holdout_table = self._deduplicate_table(holdout_table)
            print(f"  After dedup: {holdout_table.num_rows} rows")

        holdout_ts = holdout_table.column('snapshot_timestamp').to_pylist()
        holdout_ts_range = (min(holdout_ts), max(holdout_ts)) if holdout_ts else ('', '')

        self.holdout_meta = dataset_store.write_dataset(
            name="eval_holdout",
            table=holdout_table,
            version=current.run_id,
            metadata={
                'snapshot_count': len(self.holdout_paths),
                'snapshot_range': holdout_ts_range,
                'n_features': len(FEATURE_COLS),
                'feature_names': FEATURE_COLS,
                'builder_flow': current.flow_name,
                'builder_run_id': current.run_id,
            }
        )
        print(f"  Wrote eval_holdout: {self.holdout_meta.total_samples} samples")

        del holdout_table

        # Build card
        current.card.append(Markdown("# Dataset Builder"))
        current.card.append(Markdown(f"**Run ID:** {current.run_id}"))
        current.card.append(Markdown(f"**Storage:** {provider}"))

        current.card.append(Markdown("## Training Dataset"))
        current.card.append(Table([
            ["Samples", str(self.train_meta.total_samples)],
            ["Snapshots", str(self.train_meta.snapshot_count)],
            ["From", self.train_meta.snapshot_range[0][:19] if self.train_meta.snapshot_range[0] else ""],
            ["To", self.train_meta.snapshot_range[1][:19] if self.train_meta.snapshot_range[1] else ""],
            ["Features", str(self.train_meta.n_features)],
        ], headers=["Property", "Value"]))

        current.card.append(Markdown("## Evaluation Holdout"))
        current.card.append(Table([
            ["Samples", str(self.holdout_meta.total_samples)],
            ["Snapshots", str(self.holdout_meta.snapshot_count)],
            ["From", self.holdout_meta.snapshot_range[0][:19] if self.holdout_meta.snapshot_range[0] else ""],
            ["To", self.holdout_meta.snapshot_range[1][:19] if self.holdout_meta.snapshot_range[1] else ""],
        ], headers=["Property", "Value"]))

        self.next(self.register)

    def _deduplicate_table(self, table):
        """Deduplicate by coin_id, keeping most recent entry."""
        import pyarrow as pa

        if table.num_rows == 0:
            return table

        df = table.to_pandas()
        df = df.sort_values('snapshot_timestamp', ascending=False)
        df = df.drop_duplicates(subset=['coin_id'], keep='first')

        return pa.Table.from_pandas(df, preserve_index=False)

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
                "n_features": self.train_meta.n_features,
                "feature_names": ",".join(self.train_meta.feature_names),
                "builder_flow": current.flow_name,
                "builder_run_id": current.run_id,
                "deduplicated": str(self.deduplicate),
            },
            tags={"dataset_type": "training"},
            description=f"Training: {self.train_meta.total_samples} samples from {self.train_meta.snapshot_count} snapshots"
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
                "n_features": self.holdout_meta.n_features,
                "feature_names": ",".join(self.holdout_meta.feature_names),
                "builder_flow": current.flow_name,
                "builder_run_id": current.run_id,
            },
            tags={"dataset_type": "eval_holdout"},
            description=f"Holdout: {self.holdout_meta.total_samples} samples from {self.holdout_meta.snapshot_count} snapshots"
        )

        # Publish event for TrainFlow
        self.prj.safe_publish_event("dataset_ready", payload={
            "training_samples": self.train_meta.total_samples,
            "holdout_samples": self.holdout_meta.total_samples,
            "builder_run_id": current.run_id,
        })
        print("\nPublished 'dataset_ready' event")

        self.next(self.end)

    @step
    def end(self):
        """Summary."""
        print(f"\n{'='*50}")
        print("Dataset Build Complete")
        print(f"{'='*50}")
        print(f"\nTraining Dataset:")
        print(f"  Samples: {self.train_meta.total_samples}")
        print(f"  Snapshots: {self.train_meta.snapshot_count}")
        tr = self.train_meta.snapshot_range
        print(f"  Range: {tr[0][:19] if tr[0] else 'N/A'} to {tr[1][:19] if tr[1] else 'N/A'}")
        print(f"\nEval Holdout:")
        print(f"  Samples: {self.holdout_meta.total_samples}")
        print(f"  Snapshots: {self.holdout_meta.snapshot_count}")
        hr = self.holdout_meta.snapshot_range
        print(f"  Range: {hr[0][:19] if hr[0] else 'N/A'} to {hr[1][:19] if hr[1] else 'N/A'}")
        print(f"\nTrainFlow will be triggered by 'dataset_ready' event")


if __name__ == "__main__":
    BuildDatasetFlow()
