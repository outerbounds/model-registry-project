"""
Validate predictions against actual market outcomes.

This flow closes the feedback loop for crash prediction:
1. Loads predictions from ~24h ago
2. Fetches current prices for those coins
3. Checks if flagged anomalies actually crashed (>15% drop)
4. Computes outcome metrics (crash recall, precision, false alarm rate)
5. Updates model annotations with outcome-based metrics

This is the key to answering: "Does our anomaly detector actually
predict crashes, or is it just detecting noise?"

Outcome Metrics:
----------------
- crash_recall: % of actual crashes we flagged in advance
- anomaly_precision: % of our flags that preceded real crashes
- false_alarm_rate: % of flags that were wrong

Quality Gate (optional):
------------------------
If outcome_precision < threshold, model shouldn't be promoted.
This is a stronger signal than silhouette score or rate stability.

Triggering:
- @schedule(hourly=True) - runs hourly to validate predictions from 24h ago
- Can also run manually for backtesting

Usage:
    # Validate predictions from 24h ago
    python flow.py run

    # Validate specific lookback window
    python flow.py run --lookback_hours 48 --max_age_hours 72
"""

from metaflow import step, card, Parameter, schedule, Config
from obproject import ProjectFlow

import src


@schedule(hourly=True)
class ValidateOutcomesFlow(ProjectFlow):
    """
    Validate predictions against actual price movements.

    Answers: "Did the coins we flagged as anomalies actually crash?"
    """

    # Config for outcome validation thresholds
    eval_config = Config("eval_config", default="configs/evaluation.json")

    lookback_hours = Parameter(
        "lookback_hours",
        default=24,
        type=int,
        help="Minimum hours since prediction (outcome observation window)"
    )

    max_age_hours = Parameter(
        "max_age_hours",
        default=48,
        type=int,
        help="Maximum hours since prediction to include"
    )

    crash_threshold_pct = Parameter(
        "crash_threshold_pct",
        default=-15.0,
        type=float,
        help="Price drop threshold to count as 'crash' (e.g., -15 = 15% drop)"
    )

    @step
    def start(self):
        """Load predictions from lookback window."""
        from src.predictions import PredictionStore

        print(f"Project: {self.prj.project}, Branch: {self.prj.branch}")
        print(f"\nValidating predictions from {self.lookback_hours}-{self.max_age_hours}h ago")
        print(f"Crash threshold: {self.crash_threshold_pct}%")

        # Load predictions
        store = PredictionStore(self.prj.project, self.prj.branch)
        self.predictions = store.get_predictions_for_validation(
            lookback_hours=self.lookback_hours,
            max_age_hours=self.max_age_hours,
        )

        if not self.predictions:
            print("\nNo predictions found in validation window.")
            print("Run TrainDetectorFlow first to generate predictions.")
            self.has_predictions = False
        else:
            print(f"\nLoaded {len(self.predictions)} predictions for validation")

            # Group by model version
            versions = {}
            for p in self.predictions:
                versions[p.model_version] = versions.get(p.model_version, 0) + 1
            print("Predictions by model:")
            for v, count in versions.items():
                print(f"  {v}: {count}")

            # Count anomalies
            anomalies = [p for p in self.predictions if p.is_anomaly]
            print(f"\nAnomalies flagged: {len(anomalies)} / {len(self.predictions)}")

            self.has_predictions = True

        self.next(self.fetch_outcomes)

    @step
    def fetch_outcomes(self):
        """Fetch current prices and compute outcomes."""
        from src import data
        from src.predictions import OutcomeRecord
        from datetime import datetime, timezone

        if not self.has_predictions:
            self.outcomes = []
            self.next(self.compute_metrics)
            return

        # Get unique coin IDs from predictions
        coin_ids = list(set(p.coin_id for p in self.predictions))
        print(f"\nFetching current prices for {len(coin_ids)} coins...")

        # Fetch current market data
        try:
            snapshot = data.fetch_market_data(num_coins=250)  # Get more to ensure coverage
            current_prices = {
                coin['id']: coin.get('current_price', 0)
                for coin in snapshot.coins
            }
            print(f"Fetched prices for {len(current_prices)} coins")
        except Exception as e:
            print(f"Error fetching prices: {e}")
            current_prices = {}

        # Build outcome records
        self.outcomes = []
        missing_prices = 0

        for pred in self.predictions:
            current_price = current_prices.get(pred.coin_id)

            if current_price is None or current_price == 0:
                missing_prices += 1
                continue

            # Calculate price change
            price_change_pct = (
                (current_price - pred.current_price) / pred.current_price * 100
                if pred.current_price > 0 else 0
            )

            # Did it actually crash?
            actual_crash = price_change_pct <= self.crash_threshold_pct

            self.outcomes.append(OutcomeRecord(
                coin_id=pred.coin_id,
                symbol=pred.symbol,
                prediction_timestamp=pred.prediction_timestamp,
                is_anomaly=pred.is_anomaly,
                anomaly_score=pred.anomaly_score,
                price_at_prediction=pred.current_price,
                price_after_24h=current_price,
                price_change_pct=price_change_pct,
                actual_crash=actual_crash,
                model_version=pred.model_version,
            ))

        if missing_prices > 0:
            print(f"[WARN] Missing prices for {missing_prices} coins (may have been delisted)")

        print(f"\nBuilt {len(self.outcomes)} outcome records")

        # Quick stats
        crashes = [o for o in self.outcomes if o.actual_crash]
        print(f"Actual crashes in period: {len(crashes)}")

        self.next(self.compute_metrics)

    @card
    @step
    def compute_metrics(self):
        """Compute outcome-based evaluation metrics."""
        from src.predictions import compute_outcome_metrics, format_outcome_summary
        from metaflow import current
        from metaflow.cards import Markdown, Table

        current.card.append(Markdown("# Outcome Validation"))
        current.card.append(Markdown(f"**Lookback:** {self.lookback_hours}-{self.max_age_hours}h"))
        current.card.append(Markdown(f"**Crash Threshold:** {self.crash_threshold_pct}%"))

        if not self.outcomes:
            print("\nNo outcomes to compute metrics.")
            current.card.append(Markdown("*No predictions found in validation window.*"))
            self.metrics = None
            self.next(self.update_model_metrics)
            return

        # Compute metrics
        self.metrics = compute_outcome_metrics(self.outcomes, self.crash_threshold_pct)
        print(f"\n{format_outcome_summary(self.metrics)}")

        # Build card
        current.card.append(Markdown("## Summary"))
        current.card.append(Table([
            ["Total Predictions", str(self.metrics.total_predictions)],
            ["Anomalies Flagged", str(self.metrics.total_anomalies)],
            ["Actual Crashes", str(self.metrics.total_crashes)],
        ], headers=["Metric", "Value"]))

        current.card.append(Markdown("## Confusion Matrix"))
        current.card.append(Table([
            ["True Positives", str(self.metrics.true_positives), "Flagged AND crashed"],
            ["False Positives", str(self.metrics.false_positives), "Flagged but no crash"],
            ["False Negatives", str(self.metrics.false_negatives), "Not flagged but crashed"],
            ["True Negatives", str(self.metrics.true_negatives), "Not flagged, no crash"],
        ], headers=["", "Count", "Description"]))

        current.card.append(Markdown("## Outcome Metrics"))

        # Color-code the metrics
        recall_status = "✅" if self.metrics.crash_recall >= 0.5 else "⚠️"
        precision_status = "✅" if self.metrics.anomaly_precision >= 0.3 else "⚠️"
        far_status = "✅" if self.metrics.false_alarm_rate <= 0.7 else "⚠️"

        current.card.append(Table([
            [f"{recall_status} Crash Recall", f"{self.metrics.crash_recall:.1%}", "% of crashes we caught"],
            [f"{precision_status} Anomaly Precision", f"{self.metrics.anomaly_precision:.1%}", "% of flags that were real"],
            [f"{far_status} False Alarm Rate", f"{self.metrics.false_alarm_rate:.1%}", "% of flags that were wrong"],
        ], headers=["Metric", "Value", "Description"]))

        # Show worst false alarms (flagged but didn't crash)
        false_alarms = [o for o in self.outcomes if o.is_anomaly and not o.actual_crash]
        if false_alarms:
            false_alarms.sort(key=lambda x: x.price_change_pct, reverse=True)
            current.card.append(Markdown("## Top False Alarms"))
            current.card.append(Markdown("*Coins flagged as anomalies that didn't crash:*"))
            fa_rows = [
                [o.symbol, f"{o.price_change_pct:+.1f}%", f"{o.anomaly_score:.3f}"]
                for o in false_alarms[:5]
            ]
            current.card.append(Table(fa_rows, headers=["Coin", "Actual Change", "Score"]))

        # Show missed crashes (crashed but not flagged)
        missed = [o for o in self.outcomes if not o.is_anomaly and o.actual_crash]
        if missed:
            missed.sort(key=lambda x: x.price_change_pct)
            current.card.append(Markdown("## Missed Crashes"))
            current.card.append(Markdown("*Coins that crashed but weren't flagged:*"))
            missed_rows = [
                [o.symbol, f"{o.price_change_pct:+.1f}%", f"{o.anomaly_score:.3f}"]
                for o in missed[:5]
            ]
            current.card.append(Table(missed_rows, headers=["Coin", "Actual Change", "Score"]))

        # Show true positives (correctly flagged crashes)
        true_pos = [o for o in self.outcomes if o.is_anomaly and o.actual_crash]
        if true_pos:
            true_pos.sort(key=lambda x: x.price_change_pct)
            current.card.append(Markdown("## Correctly Flagged Crashes ✅"))
            tp_rows = [
                [o.symbol, f"{o.price_change_pct:+.1f}%", f"{o.anomaly_score:.3f}"]
                for o in true_pos[:5]
            ]
            current.card.append(Table(tp_rows, headers=["Coin", "Actual Change", "Score"]))

        self.next(self.update_model_metrics)

    @step
    def update_model_metrics(self):
        """Update model annotations with outcome metrics."""
        from src import registry
        from metaflow import current

        if not self.metrics:
            print("\nNo metrics to update.")
            self.next(self.end)
            return

        # Group outcomes by model version to update each
        version_outcomes = {}
        for o in self.outcomes:
            if o.model_version not in version_outcomes:
                version_outcomes[o.model_version] = []
            version_outcomes[o.model_version].append(o)

        print(f"\nUpdating metrics for {len(version_outcomes)} model versions...")

        for version, outcomes in version_outcomes.items():
            from src.predictions import compute_outcome_metrics
            version_metrics = compute_outcome_metrics(outcomes, self.crash_threshold_pct)

            # Note: We don't have a direct way to update existing model annotations
            # without re-registering. In a real system, you'd store these metrics
            # in a separate table or use the registry's annotation update API.
            print(f"\n{version}:")
            print(f"  Crash Recall: {version_metrics.crash_recall:.1%}")
            print(f"  Anomaly Precision: {version_metrics.anomaly_precision:.1%}")
            print(f"  False Alarm Rate: {version_metrics.false_alarm_rate:.1%}")

        # Store summary for the flow
        self.outcome_metrics_summary = {
            "crash_recall": self.metrics.crash_recall,
            "anomaly_precision": self.metrics.anomaly_precision,
            "false_alarm_rate": self.metrics.false_alarm_rate,
            "total_predictions": self.metrics.total_predictions,
            "total_crashes": self.metrics.total_crashes,
            "validated_at": current.run_id,
        }

        self.next(self.end)

    @step
    def end(self):
        """Summary."""
        print(f"\n{'='*50}")
        print("Outcome Validation Complete")
        print(f"{'='*50}")

        if self.metrics:
            print(f"\nResults:")
            print(f"  Crash Recall: {self.metrics.crash_recall:.1%}")
            print(f"  Anomaly Precision: {self.metrics.anomaly_precision:.1%}")
            print(f"  False Alarm Rate: {self.metrics.false_alarm_rate:.1%}")

            if self.metrics.crash_recall >= 0.5 and self.metrics.anomaly_precision >= 0.3:
                print(f"\n✅ Model shows predictive value!")
            else:
                print(f"\n⚠️ Model may need improvement")
                if self.metrics.crash_recall < 0.5:
                    print(f"  - Low crash recall: missing too many crashes")
                if self.metrics.anomaly_precision < 0.3:
                    print(f"  - Low precision: too many false alarms")
        else:
            print("\nNo predictions to validate.")
            print("Run TrainDetectorFlow to generate predictions first.")


if __name__ == "__main__":
    ValidateOutcomesFlow()
