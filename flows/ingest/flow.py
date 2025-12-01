"""
Ingest live market data and register as versioned DataAsset.

This flow:
1. Fetches live cryptocurrency market data from CoinGecko
2. Extracts features for anomaly detection
3. Stores FeatureSet as Metaflow artifact (for quick access)
4. Writes Parquet snapshot to cloud storage (for fast-data accumulation)
5. Registers asset pointing to the artifact

Scheduling:
- Runs hourly when deployed to Argo Workflows
- @schedule decorator has no effect on local runs

Data Flow:
- IngestFlow: Fetches data, writes Parquet, registers asset
- BuildDatasetFlow: Accumulates Parquet snapshots into train/eval datasets
- TrainFlow: Consumes accumulated dataset or single snapshot
- EvaluateFlow: Uses eval holdout dataset or fetches fresh data
"""

from metaflow import step, card, Parameter, schedule
from obproject import ProjectFlow

# Import src at module level so Metaflow detects METAFLOW_PACKAGE_POLICY
# and includes it in the code package for remote execution
import src


@schedule(hourly=True)
class IngestMarketDataFlow(ProjectFlow):
    """
    Ingest live market data and register as DataAsset.

    The FeatureSet is stored as a Metaflow artifact and the asset
    points to it for downstream consumption.
    """

    num_coins = Parameter(
        "num_coins",
        default=100,
        help="Number of top coins to fetch (max 250)"
    )

    @step
    def start(self):
        """Fetch live market data from CoinGecko."""
        from src import data

        print(f"Project: {self.prj.project}, Branch: {self.prj.branch}")
        print(f"Fetching live data for top {self.num_coins} cryptocurrencies...")

        snapshot = data.fetch_market_data(num_coins=self.num_coins)
        print(f"Fetched {snapshot.count} coins at {snapshot.timestamp}")

        # Extract features and store as artifact
        print("Extracting features...")
        self.feature_set = data.extract_features(snapshot)
        self.timestamp = snapshot.timestamp

        print(f"Feature matrix: {self.feature_set.n_samples} samples x {self.feature_set.n_features} features")

        self.next(self.register)

    @card
    @step
    def register(self):
        """Write Parquet snapshot and register as DataAsset."""
        from src import data
        from src.storage import SnapshotStore
        from metaflow import current
        from metaflow.cards import Markdown, Table

        # Write Parquet snapshot to cloud storage for fast-data accumulation
        print("\nWriting Parquet snapshot to storage...")
        try:
            store = SnapshotStore(self.prj.project, self.prj.branch)
            snapshot_meta = store.write_snapshot(
                self.feature_set,
                flow_name=current.flow_name,
                run_id=current.run_id,
            )
            print(f"  Wrote: {snapshot_meta.path}")
            self.snapshot_path = snapshot_meta.path
        except Exception as e:
            print(f"  [WARN] Failed to write Parquet snapshot: {e}")
            self.snapshot_path = None

        # Register asset pointing to the feature_set artifact
        self.prj.register_data(
            "market_snapshot",
            "feature_set",  # artifact name
            annotations={
                "n_samples": self.feature_set.n_samples,
                "n_features": self.feature_set.n_features,
                "feature_names": ",".join(self.feature_set.feature_names),
                "timestamp": self.timestamp,
                "data_source": "coingecko",
                "ingest_flow": current.flow_name,
                "ingest_run_id": current.run_id,
                "parquet_path": self.snapshot_path or "",
            },
            tags={"data_source": "coingecko"},
        )

        print(f"\nRegistered market_snapshot")
        print(f"  Artifact: feature_set")
        print(f"  Samples: {self.feature_set.n_samples}")
        print(f"  Features: {self.feature_set.n_features}")
        if self.snapshot_path:
            print(f"  Parquet: {self.snapshot_path}")

        # Build card
        stats = data.get_feature_stats(self.feature_set)

        current.card.append(Markdown("# Market Data Ingestion"))
        current.card.append(Markdown(f"**Timestamp:** {self.timestamp}"))
        current.card.append(Markdown(f"**Source:** CoinGecko API"))

        current.card.append(Markdown("## Data Summary"))
        current.card.append(Table([
            ["Coins Fetched", str(self.feature_set.n_samples)],
            ["Features Extracted", str(self.feature_set.n_features)],
        ], headers=["Metric", "Value"]))

        current.card.append(Markdown("## Feature Statistics"))
        rows = [
            [col, f"{s['mean']:.2f}", f"{s['stdev']:.2f}", f"{s['min']:.2f}", f"{s['max']:.2f}"]
            for col, s in stats.items()
        ]
        current.card.append(Table(rows, headers=["Feature", "Mean", "StdDev", "Min", "Max"]))

        self.next(self.end)

    @step
    def end(self):
        """Summary."""
        print(f"\n{'='*50}")
        print("Ingestion Complete")
        print(f"{'='*50}")
        print(f"DataAsset: market_snapshot")
        print(f"Samples: {self.feature_set.n_samples} coins")
        print(f"Timestamp: {self.timestamp}")
        print(f"\nNext: Run TrainAnomalyFlow to train on this data")


if __name__ == "__main__":
    IngestMarketDataFlow()
