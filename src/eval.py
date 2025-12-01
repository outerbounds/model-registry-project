"""
Evaluation and quality gates for anomaly detection models.

This module handles:
- Defining and running quality gates
- Computing unsupervised evaluation metrics (silhouette, score distribution)
- Comparing candidate vs champion
- Generating evaluation reports

Evaluation Philosophy:
----------------------
Without ground truth labels, we evaluate anomaly detectors using:

1. **Sanity Gates**: Basic checks that the model isn't broken
   - Anomaly rate bounds (not too high, not too low)
   - Rate stability vs training (detect drift)

2. **Separation Quality** (Silhouette Score):
   - Measures how well anomalies are separated from normal points
   - Range [-1, 1]: higher = better separation
   - Threshold: > 0.0 means anomalies form a distinct cluster

3. **Score Distribution** (Bimodality):
   - Healthy detector: bimodal distribution (normal vs anomaly clusters)
   - Broken detector: uniform or unimodal (can't distinguish)
   - We measure the "gap" between score distributions

4. **Temporal Consistency** (for time-series datasets):
   - Anomalies that persist across snapshots = higher confidence
   - One-off spikes = likely noise
   - Only applicable when eval data has temporal structure

References:
- Goldstein & Uchida (2016): "A Comparative Evaluation of Unsupervised Anomaly Detection"
- Silhouette analysis for cluster quality assessment
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import statistics

from .model import PredictionResult


class GateStatus(str, Enum):
    """Quality gate status."""
    PASS = "pass"
    FAIL = "fail"


@dataclass
class QualityGate:
    """Result of a single quality gate check."""
    name: str
    threshold: str
    actual: str
    status: GateStatus
    description: str = ""

    @property
    def passed(self) -> bool:
        return self.status == GateStatus.PASS


@dataclass
class UnsupervisedMetrics:
    """Metrics for unsupervised anomaly detection quality."""
    silhouette_score: Optional[float] = None  # [-1, 1], higher = better separation
    score_bimodality: Optional[float] = None  # [0, 1], higher = more bimodal
    score_gap: Optional[float] = None  # Gap between normal/anomaly score distributions
    mean_anomaly_score: Optional[float] = None
    mean_normal_score: Optional[float] = None
    score_std: Optional[float] = None


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    gates: List[QualityGate]
    all_passed: bool
    anomaly_rate: float
    n_anomalies: int
    n_samples: int
    training_rate: Optional[float] = None
    rate_diff: Optional[float] = None
    metrics: UnsupervisedMetrics = field(default_factory=UnsupervisedMetrics)

    @property
    def gates_passed(self) -> int:
        return sum(1 for g in self.gates if g.passed)

    @property
    def gates_failed(self) -> int:
        return sum(1 for g in self.gates if not g.passed)


def compute_unsupervised_metrics(
    prediction: PredictionResult,
    features: Optional[List[List[float]]] = None,
) -> UnsupervisedMetrics:
    """
    Compute unsupervised quality metrics for anomaly detection.

    These metrics help assess detection quality without ground truth labels.

    Args:
        prediction: PredictionResult with anomaly_scores and predictions
        features: Optional feature matrix for silhouette computation

    Returns:
        UnsupervisedMetrics with computed values
    """
    scores = list(prediction.anomaly_scores)
    labels = list(prediction.predictions)  # -1 = anomaly, 1 = normal

    # Separate scores by predicted class
    anomaly_scores = [s for s, l in zip(scores, labels) if l == -1]
    normal_scores = [s for s, l in zip(scores, labels) if l == 1]

    metrics = UnsupervisedMetrics()

    # Score distribution statistics
    if scores:
        metrics.score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0

    if anomaly_scores:
        metrics.mean_anomaly_score = statistics.mean(anomaly_scores)
    if normal_scores:
        metrics.mean_normal_score = statistics.mean(normal_scores)

    # Score gap: difference between mean anomaly and normal scores
    # For isolation forest, anomalies have MORE NEGATIVE scores
    # A larger gap (in absolute terms) = better separation
    if metrics.mean_anomaly_score is not None and metrics.mean_normal_score is not None:
        metrics.score_gap = abs(metrics.mean_anomaly_score - metrics.mean_normal_score)

    # Bimodality: measure how bimodal the score distribution is
    # Using Hartigan's dip statistic approximation via coefficient of bimodality
    # CB = (skewness^2 + 1) / (kurtosis + 3 * (n-1)^2 / ((n-2)(n-3)))
    # Simplified: just check if there's a clear gap in the histogram
    if len(scores) > 10 and anomaly_scores and normal_scores:
        # Simple bimodality: ratio of between-group variance to total variance
        overall_mean = statistics.mean(scores)
        between_var = (
            len(anomaly_scores) * (metrics.mean_anomaly_score - overall_mean) ** 2 +
            len(normal_scores) * (metrics.mean_normal_score - overall_mean) ** 2
        ) / len(scores)
        total_var = statistics.variance(scores) if len(scores) > 1 else 1.0
        if total_var > 0:
            metrics.score_bimodality = between_var / total_var

    # Silhouette score: requires features and sklearn
    if features is not None and len(features) > 2:
        try:
            from sklearn.metrics import silhouette_score
            import numpy as np

            # Convert labels: sklearn expects 0/1, we have -1/1
            binary_labels = [0 if l == -1 else 1 for l in labels]

            # Only compute if we have both classes
            if len(set(binary_labels)) == 2:
                metrics.silhouette_score = float(silhouette_score(
                    np.array(features),
                    binary_labels,
                    metric='euclidean'
                ))
        except Exception:
            pass  # Silhouette computation failed, leave as None

    return metrics


def run_quality_gates(
    prediction: PredictionResult,
    training_anomaly_rate: float,
    max_anomaly_rate: float = 0.20,
    min_anomaly_rate: float = 0.02,
    max_rate_diff: float = 0.25,
    min_silhouette: float = 0.0,
    min_score_gap: float = 0.05,
    features: Optional[List[List[float]]] = None,
) -> EvaluationResult:
    """
    Run quality gates on prediction result.

    Gates are divided into two categories:

    **Sanity Gates** (hard failures):
    - Max/min anomaly rate: catch broken models
    - Rate stability: catch distribution drift

    **Quality Gates** (soft failures, logged as metrics):
    - Silhouette score: separation quality
    - Score gap: distribution separation

    Args:
        prediction: PredictionResult from model inference
        training_anomaly_rate: Anomaly rate during training
        max_anomaly_rate: Maximum acceptable anomaly rate (sanity check)
        min_anomaly_rate: Minimum acceptable anomaly rate (sanity check)
        max_rate_diff: Maximum allowed rate drift from training
        min_silhouette: Minimum silhouette score for separation quality
        min_score_gap: Minimum gap between normal/anomaly score means
        features: Optional feature matrix for silhouette computation

    Returns:
        EvaluationResult with gate outcomes and metrics
    """
    gates = []

    # Compute unsupervised metrics
    metrics = compute_unsupervised_metrics(prediction, features)

    # === SANITY GATES ===
    # These catch obviously broken models

    # Gate 1: Anomaly rate not too high
    # Rationale: If >20% are anomalies, either contamination is wrong or model is broken
    gate1_passed = prediction.anomaly_rate <= max_anomaly_rate
    gates.append(QualityGate(
        name="Max anomaly rate",
        threshold=f"<= {max_anomaly_rate:.0%}",
        actual=f"{prediction.anomaly_rate:.1%}",
        status=GateStatus.PASS if gate1_passed else GateStatus.FAIL,
        description="Catches models declaring too many anomalies (broken contamination or overfitting to noise)",
    ))

    # Gate 2: Anomaly rate not too low
    # Rationale: If <2% are anomalies, model may be underfitting or features collapsed
    gate2_passed = prediction.anomaly_rate >= min_anomaly_rate
    gates.append(QualityGate(
        name="Min anomaly rate",
        threshold=f">= {min_anomaly_rate:.0%}",
        actual=f"{prediction.anomaly_rate:.1%}",
        status=GateStatus.PASS if gate2_passed else GateStatus.FAIL,
        description="Catches models declaring too few anomalies (underfitting or feature collapse)",
    ))

    # Gate 3: Rate stability (compared to training)
    # Rationale: Large drift suggests concept drift or unstable model
    rate_diff = abs(prediction.anomaly_rate - training_anomaly_rate)
    gate3_passed = rate_diff <= max_rate_diff
    gates.append(QualityGate(
        name="Rate stability",
        threshold=f"<= {max_rate_diff:.0%} drift",
        actual=f"{rate_diff:.1%} drift",
        status=GateStatus.PASS if gate3_passed else GateStatus.FAIL,
        description="Catches distribution drift between training and evaluation data",
    ))

    # === QUALITY GATES ===
    # These measure detection quality (not just sanity)

    # Gate 4: Silhouette score (separation quality)
    # Rationale: Measures how well anomalies cluster separately from normal points
    # Range [-1, 1]: >0 means anomalies form a distinct cluster
    if metrics.silhouette_score is not None:
        gate4_passed = metrics.silhouette_score >= min_silhouette
        gates.append(QualityGate(
            name="Separation quality",
            threshold=f">= {min_silhouette:.2f}",
            actual=f"{metrics.silhouette_score:.3f}",
            status=GateStatus.PASS if gate4_passed else GateStatus.FAIL,
            description="Silhouette score: measures how well anomalies separate from normal points in feature space",
        ))

    # Gate 5: Score distribution gap
    # Rationale: Healthy detector has bimodal score distribution with clear gap
    if metrics.score_gap is not None:
        gate5_passed = metrics.score_gap >= min_score_gap
        gates.append(QualityGate(
            name="Score separation",
            threshold=f">= {min_score_gap:.2f}",
            actual=f"{metrics.score_gap:.3f}",
            status=GateStatus.PASS if gate5_passed else GateStatus.FAIL,
            description="Gap between mean anomaly and normal scores (larger = more confident detection)",
        ))

    all_passed = all(g.passed for g in gates)

    return EvaluationResult(
        gates=gates,
        all_passed=all_passed,
        anomaly_rate=prediction.anomaly_rate,
        n_anomalies=prediction.n_anomalies,
        n_samples=prediction.n_samples,
        training_rate=training_anomaly_rate,
        rate_diff=rate_diff,
        metrics=metrics,
    )


def compare_models(
    candidate_result: EvaluationResult,
    champion_result: Optional[EvaluationResult],
) -> Dict[str, any]:
    """
    Compare candidate evaluation against champion.

    Args:
        candidate_result: Evaluation result for candidate
        champion_result: Evaluation result for champion (if exists)

    Returns:
        Comparison metrics dict
    """
    comparison = {
        "candidate_anomaly_rate": candidate_result.anomaly_rate,
        "candidate_gates_passed": candidate_result.gates_passed,
        "candidate_all_passed": candidate_result.all_passed,
    }

    if champion_result:
        comparison.update({
            "champion_anomaly_rate": champion_result.anomaly_rate,
            "champion_gates_passed": champion_result.gates_passed,
            "rate_diff_vs_champion": abs(
                candidate_result.anomaly_rate - champion_result.anomaly_rate
            ),
            "candidate_better": (
                candidate_result.all_passed and
                candidate_result.anomaly_rate <= champion_result.anomaly_rate
            ),
        })

    return comparison


def format_gate_summary(result: EvaluationResult) -> str:
    """
    Format gate results for logging/display.

    Returns:
        Multi-line string summary
    """
    lines = ["Quality Gates:"]
    for gate in result.gates:
        status = "PASS" if gate.passed else "FAIL"
        lines.append(f"  {gate.name}: {gate.actual} {gate.threshold} -> {status}")
    lines.append(f"\nAll gates passed: {result.all_passed}")
    return "\n".join(lines)


def get_evaluation_metrics(result: EvaluationResult) -> Dict[str, any]:
    """
    Extract metrics dict for registration.

    Returns:
        Dict suitable for asset annotations
    """
    metrics_dict = {
        "eval_anomaly_rate": float(result.anomaly_rate),
        "eval_anomalies": result.n_anomalies,
        "eval_samples": result.n_samples,
        "gates_passed": result.gates_passed,
        "gates_failed": result.gates_failed,
        "training_rate": result.training_rate,
        "rate_diff": result.rate_diff,
    }

    # Add unsupervised quality metrics
    m = result.metrics
    if m.silhouette_score is not None:
        metrics_dict["silhouette_score"] = m.silhouette_score
    if m.score_gap is not None:
        metrics_dict["score_gap"] = m.score_gap
    if m.score_bimodality is not None:
        metrics_dict["score_bimodality"] = m.score_bimodality
    if m.mean_anomaly_score is not None:
        metrics_dict["mean_anomaly_score"] = m.mean_anomaly_score
    if m.mean_normal_score is not None:
        metrics_dict["mean_normal_score"] = m.mean_normal_score

    return metrics_dict
