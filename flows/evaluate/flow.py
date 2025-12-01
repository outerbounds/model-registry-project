"""
Evaluate anomaly detection model on fresh market data.

This flow:
1. Loads candidate model from asset registry
2. Fetches fresh market data (not from asset - point is to test on new data)
3. Runs inference with same model config
4. Applies quality gates
5. Updates status to 'evaluated' if passed

Triggering:
- Auto-triggers when TrainDetectorFlow completes (in Argo)
- Publishes 'approval_requested' event on success
"""

from metaflow import step, card, Parameter, Config, trigger_on_finish
from obproject import ProjectFlow

# Import src at module level so Metaflow detects METAFLOW_PACKAGE_POLICY
# and includes it in the code package for remote execution
import src


@trigger_on_finish(flow='TrainDetectorFlow')
class EvaluateDetectorFlow(ProjectFlow):
    """
    Evaluate anomaly detector on fresh market data.

    Fetches new data (not from asset) to test model generalization.
    """

    # Centralized configs
    training_config = Config("training_config", default="configs/training.json")
    eval_config = Config("eval_config", default="configs/evaluation.json")

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
            raise RuntimeError(f"No anomaly_detector found. Run TrainDetectorFlow first: {e}")

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

        # Read num_coins from centralized config
        num_coins = self.training_config.get("num_coins", 100)

        print(f"\nFetching fresh data for top {num_coins} coins...")
        snapshot = data.fetch_market_data(num_coins=num_coins)
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
            max_anomaly_rate=self.eval_config.get("max_anomaly_rate", 0.20),
            min_anomaly_rate=self.eval_config.get("min_anomaly_rate", 0.02),
            max_rate_diff=self.eval_config.get("max_rate_diff", 0.25),
            min_silhouette=self.eval_config.get("min_silhouette", 0.0),
            min_score_gap=self.eval_config.get("min_score_gap", 0.05),
            features=self.feature_set.features,  # Enable silhouette score
        )

        print(f"\n{evaluation.format_gate_summary(self.eval_result)}")

        # Build card with visualizations
        from src.cards import (
            score_distribution_chart,
            gates_summary,
            model_comparison_table,
            top_anomalies_table,
        )

        current.card.append(Markdown("# Anomaly Detector Evaluation"))
        current.card.append(Markdown(f"**Model:** v{self.candidate.version}"))
        current.card.append(Markdown(f"**Evaluation Time:** {self.feature_set.timestamp}"))

        # Quality gates with visual indicators
        current.card.append(gates_summary(self.eval_result))

        # Score distribution chart
        current.card.append(Markdown("## Score Distribution"))
        current.card.append(score_distribution_chart(
            scores=list(self.prediction.anomaly_scores),
            labels=list(self.prediction.predictions),
            title="Evaluation Score Distribution"
        ))

        # Model comparison (candidate vs previous)
        candidate_metrics = {
            "version": f"v{self.candidate.version}",
            "anomaly_rate": self.prediction.anomaly_rate,
            "silhouette_score": self.eval_result.metrics.silhouette_score,
            "score_gap": self.eval_result.metrics.score_gap,
            "training_samples": int(self.candidate.annotations.get("training_samples", 0)),
        }
        champion_metrics = None
        if self.has_champion:
            champion_metrics = {
                "version": f"v{self.champion.version}",
                "anomaly_rate": float(self.champion.annotations.get("anomaly_rate", 0)),
                "silhouette_score": float(self.champion.annotations.get("silhouette_score", 0)) if self.champion.annotations.get("silhouette_score") else None,
                "score_gap": float(self.champion.annotations.get("score_gap", 0)) if self.champion.annotations.get("score_gap") else None,
                "training_samples": int(self.champion.annotations.get("training_samples", 0)),
            }
        current.card.append(model_comparison_table(candidate_metrics, champion_metrics))

        # Detected anomalies table
        current.card.append(Markdown("## Detected Anomalies"))
        if self.anomalies:
            current.card.append(top_anomalies_table(self.anomalies, limit=10))

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
            print(f"\nNext: Run PromoteDetectorFlow to promote to champion")


if __name__ == "__main__":
    EvaluateDetectorFlow()
