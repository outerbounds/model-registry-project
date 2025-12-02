"""
Prediction logging for outcome validation.

This module handles:
- Logging predictions with timestamps for later outcome validation
- Loading historical predictions to check against actual price movements
- Computing outcome-based metrics (crash recall, false alarm rate)

Storage Layout:
    {datastore_root}/projects/{project}/branches/{branch}/predictions/
        dt=2025-12-01/
            hour=18/
                pred_{run_id}.parquet

Each prediction record contains:
- coin_id, symbol: Which coin
- prediction_timestamp: When the prediction was made
- current_price: Price at prediction time
- anomaly_score: Model's anomaly score
- is_anomaly: Binary prediction (True/False)
- model_version: Which model made this prediction
- flow_run_id: Which flow run
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
import json


@dataclass
class PredictionRecord:
    """A single prediction to be validated later."""
    coin_id: str
    symbol: str
    prediction_timestamp: str
    current_price: float
    anomaly_score: float
    is_anomaly: bool
    model_version: str
    flow_run_id: str


@dataclass
class OutcomeRecord:
    """A prediction with its outcome (did the coin actually crash?)."""
    coin_id: str
    symbol: str
    prediction_timestamp: str
    is_anomaly: bool
    anomaly_score: float
    price_at_prediction: float
    price_after_24h: Optional[float]
    price_change_pct: Optional[float]
    actual_crash: Optional[bool]  # Did it actually drop >15%?
    model_version: str


@dataclass
class OutcomeMetrics:
    """Metrics from outcome validation."""
    total_predictions: int
    total_anomalies: int
    total_crashes: int
    true_positives: int  # Flagged as anomaly AND crashed
    false_positives: int  # Flagged as anomaly but didn't crash
    false_negatives: int  # Not flagged but crashed
    true_negatives: int  # Not flagged and didn't crash

    @property
    def crash_recall(self) -> float:
        """% of actual crashes that were flagged as anomalies."""
        if self.total_crashes == 0:
            return 0.0
        return self.true_positives / self.total_crashes

    @property
    def anomaly_precision(self) -> float:
        """% of flagged anomalies that actually crashed."""
        if self.total_anomalies == 0:
            return 0.0
        return self.true_positives / self.total_anomalies

    @property
    def false_alarm_rate(self) -> float:
        """% of flagged anomalies that didn't crash."""
        if self.total_anomalies == 0:
            return 0.0
        return self.false_positives / self.total_anomalies


class PredictionStore:
    """Store and retrieve predictions for outcome validation."""

    def __init__(self, project: str, branch: str):
        from src.storage import get_datastore_root, get_project_storage_path
        self.project = project
        self.branch = branch
        self.base_path = get_project_storage_path(project, branch, "predictions")
        self.root, self.provider = get_datastore_root()

    def _get_prediction_path(self, timestamp: str, run_id: str) -> str:
        """Generate path for a prediction log."""
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        date_str = dt.strftime('%Y-%m-%d')
        hour_str = dt.strftime('%H')
        return f"{self.base_path}/dt={date_str}/hour={hour_str}/pred_{run_id}.parquet"

    def log_predictions(
        self,
        predictions: List[PredictionRecord],
        flow_name: str,
        run_id: str,
    ) -> str:
        """
        Write predictions to storage for later outcome validation.

        Args:
            predictions: List of PredictionRecord objects
            flow_name: Name of the flow logging these predictions
            run_id: Run ID of the flow

        Returns:
            Path where predictions were stored
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        from metaflow import S3
        import io

        if not predictions:
            return ""

        # Convert to columnar format
        columns = {
            'coin_id': [p.coin_id for p in predictions],
            'symbol': [p.symbol for p in predictions],
            'prediction_timestamp': [p.prediction_timestamp for p in predictions],
            'current_price': [p.current_price for p in predictions],
            'anomaly_score': [p.anomaly_score for p in predictions],
            'is_anomaly': [p.is_anomaly for p in predictions],
            'model_version': [p.model_version for p in predictions],
            'flow_run_id': [p.flow_run_id for p in predictions],
        }

        table = pa.Table.from_pydict(columns)

        # Get path based on first prediction timestamp
        path = self._get_prediction_path(predictions[0].prediction_timestamp, run_id)

        # Write to bytes
        buffer = io.BytesIO()
        pq.write_table(table, buffer, compression='snappy')
        parquet_bytes = buffer.getvalue()

        if self.provider == 's3':
            with S3() as s3:
                s3.put(path, parquet_bytes)
        else:
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(parquet_bytes)

        return path

    def get_predictions_for_validation(
        self,
        lookback_hours: int = 24,
        max_age_hours: int = 48,
    ) -> List[PredictionRecord]:
        """
        Load predictions from ~lookback_hours ago for outcome validation.

        We want predictions that are:
        - Old enough to have outcome data (at least lookback_hours old)
        - Not too old (max_age_hours)

        Args:
            lookback_hours: Minimum hours since prediction
            max_age_hours: Maximum hours since prediction

        Returns:
            List of PredictionRecord objects
        """
        import pyarrow.parquet as pq
        from metaflow import S3
        from concurrent.futures import ThreadPoolExecutor

        now = datetime.now(timezone.utc)
        min_ts = now - timedelta(hours=max_age_hours)
        max_ts = now - timedelta(hours=lookback_hours)

        # List prediction files
        if self.provider == 's3':
            with S3() as s3:
                files = list(s3.list_recursive([self.base_path]))
                paths = [f.url for f in files if f.url.endswith('.parquet')]
        else:
            import os
            paths = []
            if os.path.exists(self.base_path):
                for root, _, files in os.walk(self.base_path):
                    for f in files:
                        if f.endswith('.parquet'):
                            paths.append(os.path.join(root, f))

        if not paths:
            return []

        # Load and filter
        predictions = []

        if self.provider == 's3':
            with S3() as s3:
                loaded = s3.get_many(paths)
                for f in loaded:
                    table = pq.read_table(f.path)
                    df = table.to_pandas()
                    for _, row in df.iterrows():
                        ts = datetime.fromisoformat(row['prediction_timestamp'].replace('Z', '+00:00'))
                        if min_ts <= ts <= max_ts:
                            predictions.append(PredictionRecord(
                                coin_id=row['coin_id'],
                                symbol=row['symbol'],
                                prediction_timestamp=row['prediction_timestamp'],
                                current_price=float(row['current_price']),
                                anomaly_score=float(row['anomaly_score']),
                                is_anomaly=bool(row['is_anomaly']),
                                model_version=row['model_version'],
                                flow_run_id=row['flow_run_id'],
                            ))
        else:
            for path in paths:
                table = pq.read_table(path)
                df = table.to_pandas()
                for _, row in df.iterrows():
                    ts = datetime.fromisoformat(row['prediction_timestamp'].replace('Z', '+00:00'))
                    if min_ts <= ts <= max_ts:
                        predictions.append(PredictionRecord(
                            coin_id=row['coin_id'],
                            symbol=row['symbol'],
                            prediction_timestamp=row['prediction_timestamp'],
                            current_price=float(row['current_price']),
                            anomaly_score=float(row['anomaly_score']),
                            is_anomaly=bool(row['is_anomaly']),
                            model_version=row['model_version'],
                            flow_run_id=row['flow_run_id'],
                        ))

        return predictions


def compute_outcome_metrics(
    outcomes: List[OutcomeRecord],
    crash_threshold_pct: float = -15.0,
) -> OutcomeMetrics:
    """
    Compute outcome-based evaluation metrics.

    Args:
        outcomes: List of OutcomeRecord with actual price changes
        crash_threshold_pct: Threshold for "crash" (e.g., -15%)

    Returns:
        OutcomeMetrics with precision, recall, false alarm rate
    """
    total = len(outcomes)
    anomalies = [o for o in outcomes if o.is_anomaly]
    crashes = [o for o in outcomes if o.actual_crash]

    tp = len([o for o in outcomes if o.is_anomaly and o.actual_crash])
    fp = len([o for o in outcomes if o.is_anomaly and not o.actual_crash])
    fn = len([o for o in outcomes if not o.is_anomaly and o.actual_crash])
    tn = len([o for o in outcomes if not o.is_anomaly and not o.actual_crash])

    return OutcomeMetrics(
        total_predictions=total,
        total_anomalies=len(anomalies),
        total_crashes=len(crashes),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
    )


def format_outcome_summary(metrics: OutcomeMetrics) -> str:
    """Format outcome metrics for display."""
    lines = [
        "Outcome Validation Results:",
        f"  Total predictions: {metrics.total_predictions}",
        f"  Anomalies flagged: {metrics.total_anomalies}",
        f"  Actual crashes: {metrics.total_crashes}",
        "",
        "Confusion Matrix:",
        f"  True Positives (flagged + crashed): {metrics.true_positives}",
        f"  False Positives (flagged + no crash): {metrics.false_positives}",
        f"  False Negatives (not flagged + crashed): {metrics.false_negatives}",
        f"  True Negatives (not flagged + no crash): {metrics.true_negatives}",
        "",
        "Metrics:",
        f"  Crash Recall: {metrics.crash_recall:.1%} (% of crashes we caught)",
        f"  Anomaly Precision: {metrics.anomaly_precision:.1%} (% of flags that were real)",
        f"  False Alarm Rate: {metrics.false_alarm_rate:.1%} (% of flags that were wrong)",
    ]
    return "\n".join(lines)
