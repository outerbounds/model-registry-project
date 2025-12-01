"""
Evaluate anomaly detection model on fresh market data.

This flow:
1. Loads candidate model from asset registry
2. Fetches fresh market data (not from asset - point is to test on new data)
3. Runs inference with same model config
4. Applies quality gates
5. Updates status to 'evaluated' if passed

Triggering:
- Auto-triggers when TrainAnomalyFlow completes (in Argo)
- Publishes 'approval_requested' event on success
"""

from metaflow import step, card, Parameter, trigger_on_finish
from obproject import ProjectFlow

# Import src at module level so Metaflow detects METAFLOW_PACKAGE_POLICY
# and includes it in the code package for remote execution
import src


@trigger_on_finish(flow='TrainAnomalyFlow')
class EvaluateAnomalyFlow(ProjectFlow):
    """
    Evaluate anomaly detector on fresh market data.

    Fetches new data (not from asset) to test model generalization.
    """

    max_anomaly_rate = Parameter(
        "max_anomaly_rate",
        default=0.20,
        help="Maximum acceptable anomaly rate"
    )
    min_anomaly_rate = Parameter(
        "min_anomaly_rate",
        default=0.02,
        help="Minimum acceptable anomaly rate"
    )
    num_coins = Parameter(
        "num_coins",
        default=100,
        help="Number of coins to fetch for evaluation"
    )

    @step
    def start(self):
        """Load candidate model from registry."""
        from src import registry

        print(f"Project: {self.prj.project}, Branch: {self.prj.branch}")

        # Load candidate (latest)
        try:
            self.candidate = registry.load_model(
                self.prj.asset, "anomaly_detector", instance="latest"
            )
        except Exception as e:
            raise RuntimeError(f"No anomaly_detector found. Run TrainAnomalyFlow first: {e}")

        print(f"\nCandidate (v{self.candidate.version}):")
        print(f"  Status: {self.candidate.status.value}")
        print(f"  Algorithm: {self.candidate.annotations.get('algorithm')}")
        print(f"  Training anomaly rate: {float(self.candidate.annotations.get('anomaly_rate', 0)):.1%}")

        # Load previous version for comparison
        try:
            self.champion = registry.load_model(
                self.prj.asset, "anomaly_detector", instance="latest-1"
            )
            self.has_champion = True
            print(f"\nPrevious (v{self.champion.version}):")
            print(f"  Training anomaly rate: {float(self.champion.annotations.get('anomaly_rate', 0)):.1%}")
        except Exception:
            print("\nNo previous version (first model)")
            self.has_champion = False
            self.champion = None

        self.next(self.fetch_eval_data)

    @step
    def fetch_eval_data(self):
        """Fetch fresh market data for evaluation."""
        from src import data

        print(f"\nFetching fresh data for top {self.num_coins} coins...")
        snapshot = data.fetch_market_data(num_coins=self.num_coins)
        self.feature_set = data.extract_features(snapshot)

        print(f"Prepared {self.feature_set.n_samples} samples x {self.feature_set.n_features} features")
        print(f"Timestamp: {self.feature_set.timestamp}")

        self.next(self.evaluate)

    @card
    @step
    def evaluate(self):
        """Run model on fresh data and apply quality gates."""
        from src.model import Model, get_anomalies
        from src import eval as evaluation
        from metaflow import current
        from metaflow.cards import Markdown, Table

        print("\nEvaluating candidate on fresh data...")

        # Recreate model from annotations
        model_config = {
            "algorithm": self.candidate.annotations.get("algorithm", "isolation_forest"),
            "hyperparameters": {
                "n_estimators": int(float(self.candidate.annotations.get("n_estimators", 100))),
                "contamination": float(self.candidate.annotations.get("contamination", 0.1)),
            }
        }
        model = Model(model_config)
        _, self.prediction = model.train(self.feature_set)

        print(f"\nResults:")
        print(f"  Anomalies: {self.prediction.n_anomalies} / {self.prediction.n_samples}")
        print(f"  Anomaly rate: {self.prediction.anomaly_rate:.1%}")

        # Get detected anomalies
        self.anomalies = get_anomalies(self.prediction, self.feature_set, top_n=10)

        print(f"\nTop anomalies:")
        for a in self.anomalies[:5]:
            c = a.coin_info
            print(f"  {c['symbol']:6} 24h: {c['price_change_24h']:+.1f}%  score: {a.anomaly_score:.3f}")

        # Run quality gates with features for silhouette computation
        training_rate = float(self.candidate.annotations.get("anomaly_rate", 0))
        self.eval_result = evaluation.run_quality_gates(
            self.prediction,
            training_anomaly_rate=training_rate,
            max_anomaly_rate=self.max_anomaly_rate,
            min_anomaly_rate=self.min_anomaly_rate,
            features=self.feature_set.features,  # Enable silhouette score
        )

        print(f"\n{evaluation.format_gate_summary(self.eval_result)}")

        # Build card
        current.card.append(Markdown(f"# Anomaly Detector Evaluation"))
        current.card.append(Markdown(f"**Model:** v{self.candidate.version}"))
        current.card.append(Markdown(f"**Evaluation Time:** {self.feature_set.timestamp}"))

        current.card.append(Markdown("## Model Info"))
        current.card.append(Table([
            ["Version", f"v{self.candidate.version}"],
            ["Algorithm", model.algorithm],
            *[[k, str(v)] for k, v in model.hyperparameters.items()],
        ], headers=["Property", "Value"]))

        current.card.append(Markdown("## Evaluation Results"))
        current.card.append(Table([
            ["Evaluation Samples", str(self.prediction.n_samples)],
            ["Anomalies Detected", str(self.prediction.n_anomalies)],
            ["Eval Anomaly Rate", f"{self.prediction.anomaly_rate:.1%}"],
            ["Training Anomaly Rate", f"{training_rate:.1%}"],
        ], headers=["Metric", "Value"]))

        current.card.append(Markdown("## Quality Gates"))
        gate_rows = [
            [g.name, g.threshold, g.actual, "PASS" if g.passed else "FAIL"]
            for g in self.eval_result.gates
        ]
        current.card.append(Table(gate_rows, headers=["Gate", "Threshold", "Actual", "Status"]))

        current.card.append(Markdown("## Detected Anomalies"))
        if self.anomalies:
            rows = [
                [a.coin_info["symbol"], a.coin_info["name"],
                 f"${a.coin_info['current_price']:,.2f}",
                 f"{a.coin_info['price_change_24h']:+.1f}%",
                 f"{a.anomaly_score:.3f}"]
                for a in self.anomalies
            ]
            current.card.append(Table(rows, headers=["Symbol", "Name", "Price", "24h Change", "Score"]))

        self.next(self.update_status)

    @step
    def update_status(self):
        """Update model status if gates passed."""
        from src import registry
        from metaflow import current

        if self.eval_result.all_passed:
            registry.update_status(
                self.prj.asset,
                "anomaly_detector",
                new_status=registry.ModelStatus.EVALUATED,
                current_annotations=self.candidate.annotations,
                additional_annotations={
                    "evaluated_by_flow": current.flow_name,
                    "evaluated_by_run_id": current.run_id,
                    "eval_anomaly_rate": float(self.prediction.anomaly_rate),
                    "eval_timestamp": self.feature_set.timestamp,
                },
            )
            print(f"\nModel v{self.candidate.version}: candidate -> evaluated")

            # Publish approval request event
            registry.publish_event(self.prj, "approval_requested", payload={
                "model_asset": "anomaly_detector",
                "candidate_version": self.candidate.version,
                "anomaly_rate": float(self.prediction.anomaly_rate),
                "evaluation_run_id": current.run_id,
            })
            print("Published 'approval_requested' event")
        else:
            print(f"\nModel v{self.candidate.version} failed gates - status unchanged")

        self.next(self.end)

    @step
    def end(self):
        """Summary."""
        result = "PASSED" if self.eval_result.all_passed else "FAILED"
        print(f"\n{'='*50}")
        print("Evaluation Complete")
        print(f"{'='*50}")
        print(f"Model: anomaly_detector v{self.candidate.version}")
        print(f"Result: {result}")
        print(f"Anomaly rate: {self.prediction.anomaly_rate:.1%}")
        if self.eval_result.all_passed:
            print(f"\nNext: Run PromoteAnomalyFlow to promote to champion")


if __name__ == "__main__":
    EvaluateAnomalyFlow()
