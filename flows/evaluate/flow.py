"""
Evaluate anomaly detection model on fresh market data.

This flow:
1. Loads the candidate model (configurable version, default=latest)
2. Optionally loads a comparison model (e.g., current champion)
3. Fetches fresh market data for evaluation
4. Runs inference and applies quality gates
5. Registers evaluation results linked to the model version

Model Loading:
- By default, evaluates "latest" (the candidate)
- Can specify any version or alias via parameters
- Comparison model is optional (for first model, or A/B tests)

Triggering:
- Auto-triggers when TrainDetectorFlow completes (in Argo)
- Publishes 'evaluation_passed' event on success
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

    Loads models by version or alias - no hardcoded assumptions.
    """

    # Centralized configs
    training_config = Config("training_config", default="configs/training.json")
    eval_config = Config("eval_config", default="configs/evaluation.json")

    # Model selection parameters
    candidate_version = Parameter(
        "candidate_version",
        default="latest",
        help="Version of model to evaluate (e.g., 'latest', 'v5', or version ID)"
    )

    compare_to = Parameter(
        "compare_to",
        default="latest-1",
        help="Version to compare against (e.g., 'latest-1', 'champion', or version ID). Use 'none' to skip comparison."
    )

    @step
    def start(self):
        """Load models from registry."""
        from src import registry

        print(f"Project: {self.prj.project}, Branch: {self.prj.branch}")

        # Load candidate model
        print(f"\nLoading candidate model (version={self.candidate_version})...")
        try:
            self.candidate = registry.load_model(
                self.prj.asset,
                "anomaly_detector",
                version=self.candidate_version
            )
        except Exception as e:
            raise RuntimeError(f"Could not load anomaly_detector version '{self.candidate_version}'. "
                             f"Run TrainDetectorFlow first: {e}")

        print(f"Candidate (v{self.candidate.version}):")
        print(f"  Algorithm: {self.candidate.algorithm}")
        print(f"  Alias: {self.candidate.alias or 'none'}")
        print(f"  Training anomaly rate: {float(self.candidate.annotations.get('anomaly_rate', 0)):.1%}")
        print(f"  Training run: {self.candidate.training_run_id}")

        # Optionally load comparison model
        self.comparison = None
        if self.compare_to and self.compare_to.lower() != "none":
            print(f"\nLoading comparison model (version={self.compare_to})...")
            try:
                self.comparison = registry.load_model(
                    self.prj.asset,
                    "anomaly_detector",
                    version=self.compare_to
                )
                print(f"Comparison (v{self.comparison.version}):")
                print(f"  Algorithm: {self.comparison.algorithm}")
                print(f"  Alias: {self.comparison.alias or 'none'}")
                print(f"  Training anomaly rate: {float(self.comparison.annotations.get('anomaly_rate', 0)):.1%}")
            except Exception as e:
                print(f"[INFO] No comparison model found ({self.compare_to}): {e}")
                print("Proceeding without comparison (first model or invalid version)")

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
        current.card.append(Markdown(f"**Candidate:** v{self.candidate.version}"))
        if self.candidate.alias:
            current.card.append(Markdown(f"**Alias:** {self.candidate.alias}"))
        if self.comparison:
            current.card.append(Markdown(f"**Comparing to:** v{self.comparison.version} ({self.comparison.alias or 'no alias'})"))
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

        # Model comparison (if we have a comparison model)
        candidate_metrics = {
            "version": f"v{self.candidate.version}",
            "anomaly_rate": self.prediction.anomaly_rate,
            "silhouette_score": self.eval_result.metrics.silhouette_score,
            "score_gap": self.eval_result.metrics.score_gap,
            "training_samples": int(self.candidate.annotations.get("training_samples", 0)),
        }
        comparison_metrics = None
        if self.comparison:
            comparison_metrics = {
                "version": f"v{self.comparison.version}",
                "anomaly_rate": float(self.comparison.annotations.get("anomaly_rate", 0)),
                "silhouette_score": float(self.comparison.annotations.get("silhouette_score", 0)) if self.comparison.annotations.get("silhouette_score") else None,
                "score_gap": float(self.comparison.annotations.get("score_gap", 0)) if self.comparison.annotations.get("score_gap") else None,
                "training_samples": int(self.comparison.annotations.get("training_samples", 0)),
            }
        current.card.append(model_comparison_table(candidate_metrics, comparison_metrics))

        # Detected anomalies table
        current.card.append(Markdown("## Detected Anomalies"))
        if self.anomalies:
            current.card.append(top_anomalies_table(self.anomalies, limit=10))

        self.next(self.register_evaluation)

    @step
    def register_evaluation(self):
        """Register evaluation results linked to model version."""
        from src import registry
        from metaflow import current

        # Create evaluation record artifact (required by prj.register_data)
        self.evaluation_record = {
            "model_name": "anomaly_detector",
            "model_version": self.candidate.version,
            "passed": self.eval_result.all_passed,
            "compared_to_version": self.comparison.version if self.comparison else None,
            "eval_anomaly_rate": float(self.prediction.anomaly_rate),
            "silhouette_score": self.eval_result.metrics.silhouette_score,
            "score_gap": self.eval_result.metrics.score_gap,
            "eval_timestamp": self.feature_set.timestamp,
        }

        # Register evaluation results linked to this specific model version
        registry.register_evaluation(
            self.prj,
            model_name="anomaly_detector",
            model_version=self.candidate.version,
            passed=self.eval_result.all_passed,
            metrics={
                "eval_anomaly_rate": float(self.prediction.anomaly_rate),
                "silhouette_score": self.eval_result.metrics.silhouette_score,
                "score_gap": self.eval_result.metrics.score_gap,
                "eval_timestamp": self.feature_set.timestamp,
            },
            eval_dataset=f"coingecko_live:{self.feature_set.timestamp}",
            compared_to_version=self.comparison.version if self.comparison else None,
            run_metadata={
                "evaluated_by_flow": current.flow_name,
                "evaluated_by_run_id": current.run_id,
            },
        )

        if self.eval_result.all_passed:
            print(f"\nEvaluation PASSED for v{self.candidate.version}")

            # Publish event for downstream (e.g., notifications, promotion workflows)
            registry.publish_event(self.prj, "evaluation_passed", payload={
                "model_name": "anomaly_detector",
                "model_version": self.candidate.version,
                "compared_to_version": self.comparison.version if self.comparison else None,
                "anomaly_rate": float(self.prediction.anomaly_rate),
                "evaluation_run_id": current.run_id,
            })
            print("Published 'evaluation_passed' event")
        else:
            print(f"\nEvaluation FAILED for v{self.candidate.version}")

        self.next(self.end)

    @step
    def end(self):
        """Summary."""
        result = "PASSED" if self.eval_result.all_passed else "FAILED"
        print(f"\n{'='*50}")
        print("Evaluation Complete")
        print(f"{'='*50}")
        print(f"Model: anomaly_detector v{self.candidate.version}")
        if self.comparison:
            print(f"Compared to: v{self.comparison.version}")
        print(f"Result: {result}")
        print(f"Anomaly rate: {self.prediction.anomaly_rate:.1%}")

        if self.eval_result.all_passed:
            print(f"\nModel passed quality gates!")
            print(f"To promote to champion, run PromoteFlow or use the dashboard.")


if __name__ == "__main__":
    EvaluateDetectorFlow()
