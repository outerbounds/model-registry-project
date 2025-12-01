"""
Train anomaly detection model on market data.

This flow is config-driven:
- model_config: Algorithm selection and hyperparameters
- training_config: Data source and training parameters

Triggering:
- Listens for 'dataset_ready' event from BuildDatasetFlow
- Does NOT trigger on every ingest (that would retrain hourly!)
- For manual runs, use command line directly

Data Sources (via training_config.data_asset):
1. "training_dataset" - Accumulated Parquet dataset from BuildDatasetFlow (default)
2. "market_snapshot" - Single snapshot artifact from IngestFlow (backwards compat)
3. null/omit - Fetch fresh data from CoinGecko (for quick testing)

Usage:
    # Use accumulated training dataset (default)
    python flow.py run

    # Use single snapshot (backwards compatible)
    python flow.py --config-value training_config '{"data_asset": "market_snapshot", "data_version": "latest"}' run

    # Override model config (experiment with different algorithm)
    python flow.py --config model_config configs/model_lof.json run

    # Fresh data (skip asset, fetch live)
    python flow.py --config-value training_config '{"data_version": null}' run
"""

from metaflow import step, card, Config
from obproject import ProjectFlow, project_trigger

# Import src at module level so Metaflow detects METAFLOW_PACKAGE_POLICY
# and includes it in the code package for remote execution
import src


# Train when BuildDatasetFlow publishes 'dataset_ready' event
# This replaces @trigger_on_finish(flow='IngestMarketDataFlow') which would
# retrain on every hourly ingest - wasteful when using accumulated datasets.
@project_trigger(event='dataset_ready')
class TrainAnomalyFlow(ProjectFlow):
    """
    Train anomaly detection model on market data.

    Config-driven: model algorithm and training parameters come from
    JSON config files. The flow consumes data assets and is agnostic
    to both model implementation and data source details.
    """

    model_config = Config("model_config", default="configs/model.json")
    training_config = Config("training_config", default="configs/training.json")

    @step
    def start(self):
        """Load training data from asset or fetch fresh."""
        print(f"Project: {self.prj.project}, Branch: {self.prj.branch}")
        print(f"\nModel config: {dict(self.model_config)}")
        print(f"Training config: {dict(self.training_config)}")
        
        self.next(self.train)

    def _fetch_fresh_data(self):
        """Fetch fresh data from CoinGecko."""
        from src import data

        num_coins = self.training_config.get("num_coins", 100)
        print(f"Fetching live data for top {num_coins} cryptocurrencies...")

        snapshot = data.fetch_market_data(num_coins=num_coins)
        self.feature_set = data.extract_features(snapshot)
        self.data_source = "coingecko_live"

        print(f"Fetched {self.feature_set.n_samples} coins")

    def _load_parquet_dataset(self, asset_name: str, version: str):
        """Load accumulated Parquet dataset from storage."""
        from src.storage import DatasetStore, table_to_feature_set
        from src.data import FEATURE_COLS

        print(f"Loading Parquet dataset: {asset_name} (version: {version})...")

        store = DatasetStore(self.prj.project, self.prj.branch)
        table, metadata = store.load_dataset(asset_name, version=version)

        if table is None:
            raise ValueError(f"Dataset {asset_name}:{version} not found")

        # Convert to FeatureSet
        self.feature_set = table_to_feature_set(table, FEATURE_COLS)
        self.data_source = f"{asset_name}:{version}"

        print(f"Loaded {self.feature_set.n_samples} samples from {metadata.snapshot_count} snapshots")
        print(f"Snapshot range: {metadata.snapshot_range[0][:19]} to {metadata.snapshot_range[1][:19]}")

    @card
    @step
    def train(self):
        """Train model using model_config."""
        from src.model import Model, get_anomalies
        from metaflow import current
        from metaflow.cards import Markdown, Table

        data_version = self.training_config.get("data_version")
        data_asset = self.training_config.get("data_asset", "market_snapshot")

        if data_version:
            print(f"\nLoading {data_asset} (version: {data_version})...")

            try:
                if data_asset == "training_dataset":
                    # Load accumulated Parquet dataset (fast-data pattern)
                    self._load_parquet_dataset(data_asset, data_version)
                else:
                    # Load single snapshot artifact (backwards compatible)
                    self.feature_set = self.prj.get_data(data_asset, instance=data_version)
                    self.data_source = f"{data_asset}:{data_version}"
                    print(f"Loaded FeatureSet: {self.feature_set.n_samples} samples x {self.feature_set.n_features} features")
                    print(f"Timestamp: {self.feature_set.timestamp}")
            except Exception as e:
                print(f"[WARN] Could not load data asset: {e}")
                print("Falling back to fresh fetch...")
                self._fetch_fresh_data()
        else:
            # Fetch fresh data (for quick testing without ingest)
            self._fetch_fresh_data()

        # Instantiate model from config
        model = Model(dict(self.model_config))
        print(f"\nTraining: {model.description}")

        # Train on feature set
        self.trained_model, self.prediction = model.train(self.feature_set)

        print(f"\nResults:")
        print(f"  Anomalies detected: {self.prediction.n_anomalies} / {self.prediction.n_samples}")
        print(f"  Anomaly rate: {self.prediction.anomaly_rate:.1%}")

        # Get top anomalies for reporting
        self.top_anomalies = get_anomalies(self.prediction, self.feature_set, top_n=10)

        print(f"\nTop anomalies:")
        for a in self.top_anomalies[:5]:
            c = a.coin_info
            print(f"  {c['symbol']:6} 24h: {c['price_change_24h']:+.1f}%  score: {a.anomaly_score:.3f}")

        # Build card
        current.card.append(Markdown("# Crypto Anomaly Detection Training"))
        current.card.append(Markdown(f"**Data Source:** {self.data_source}"))
        current.card.append(Markdown(f"**Timestamp:** {self.feature_set.timestamp}"))

        current.card.append(Markdown("## Model Configuration"))
        current.card.append(Table([
            ["Algorithm", model.algorithm],
            *[[k, str(v)] for k, v in model.hyperparameters.items()],
            ["Training Samples", str(self.feature_set.n_samples)],
            ["Features", str(self.feature_set.n_features)],
        ], headers=["Parameter", "Value"]))

        current.card.append(Markdown("## Training Results"))
        current.card.append(Table([
            ["Anomalies Detected", str(self.prediction.n_anomalies)],
            ["Anomaly Rate", f"{self.prediction.anomaly_rate:.1%}"],
        ], headers=["Metric", "Value"]))

        current.card.append(Markdown("## Top Anomalies"))
        if self.top_anomalies:
            rows = [
                [a.coin_info["symbol"], a.coin_info["name"],
                 f"{a.coin_info['price_change_24h']:+.1f}%", f"{a.anomaly_score:.3f}"]
                for a in self.top_anomalies
            ]
            current.card.append(Table(rows, headers=["Symbol", "Name", "24h Change", "Score"]))

        self.next(self.register)

    @step
    def register(self):
        """Register trained model as candidate."""
        from src import registry
        from metaflow import current

        annotations = {
            "algorithm": self.trained_model.algorithm,
            **self.trained_model.hyperparameters,
            "n_features": self.feature_set.n_features,
            "feature_names": ",".join(self.feature_set.feature_names),
            "training_samples": self.feature_set.n_samples,
            "anomalies_detected": self.prediction.n_anomalies,
            "anomaly_rate": float(self.prediction.anomaly_rate),
            "training_timestamp": self.feature_set.timestamp,
            "training_flow": current.flow_name,
            "training_run_id": current.run_id,
            "data_source": self.data_source,
        }

        registry.register_model(
            self.prj,
            "anomaly_detector",
            status=registry.ModelStatus.CANDIDATE,
            annotations=annotations,
            description=f"Anomaly detector: {self.prediction.anomaly_rate:.1%} rate on {self.feature_set.n_samples} coins",
        )

        print(f"\nRegistered anomaly_detector as CANDIDATE")
        print(f"  Algorithm: {self.trained_model.algorithm}")
        print(f"  Data: {self.data_source}")
        print(f"  Run ID: {current.run_id}")

        self.next(self.end)

    @step
    def end(self):
        """Summary."""
        print(f"\n{'='*50}")
        print("Training Complete")
        print(f"{'='*50}")
        print(f"Model: anomaly_detector (status: candidate)")
        print(f"Algorithm: {self.trained_model.algorithm}")
        print(f"Data source: {self.data_source}")
        print(f"Anomaly rate: {self.prediction.anomaly_rate:.1%}")
        print(f"\nNext: Run EvaluateAnomalyFlow to evaluate on fresh data")


if __name__ == "__main__":
    TrainAnomalyFlow()
