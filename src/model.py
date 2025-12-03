"""
Anomaly detection model abstraction.

Separates model configuration from training orchestration:
- Model class: instantiated with model_config (algorithm + hyperparameters)
- train(): takes training_config and data, returns trained model + predictions

This allows flows to be algorithm-agnostic while supporting experimentation
with different algorithms via config files.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

from .data import FeatureSet


@dataclass
class PredictionResult:
    """Container for prediction results."""
    predictions: np.ndarray  # -1 = anomaly, 1 = normal
    anomaly_scores: np.ndarray
    anomaly_indices: np.ndarray
    anomaly_rate: float
    n_anomalies: int
    n_samples: int


@dataclass
class AnomalyResult:
    """Detailed anomaly result with coin metadata."""
    coin_info: Dict
    anomaly_score: float
    is_anomaly: bool = True


@dataclass
class TrainedModel:
    """Container for trained model with metadata."""
    model: object  # sklearn model
    scaler: object  # sklearn StandardScaler
    algorithm: str
    hyperparameters: Dict
    scaler_mean: List[float] = field(default_factory=list)
    scaler_scale: List[float] = field(default_factory=list)


class Model:
    """
    Algorithm-agnostic anomaly detection model.

    Instantiate with model_config (from configs/model.json), then call train()
    with training data. The flow doesn't need to know about algorithm internals.

    Usage:
        model = Model(self.model_config)
        trained, predictions = model.train(feature_set)
    """

    SUPPORTED_ALGORITHMS = ["isolation_forest", "local_outlier_factor"]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model with configuration.

        Args:
            config: Dict with 'algorithm' and 'hyperparameters' keys
        """
        self.algorithm = config.get("algorithm", "isolation_forest")
        self.hyperparameters = config.get("hyperparameters", {})

        if self.algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm: {self.algorithm}. "
                f"Supported: {self.SUPPORTED_ALGORITHMS}"
            )

    def train(self, feature_set: FeatureSet) -> Tuple[TrainedModel, PredictionResult]:
        """
        Train model on feature set.

        Args:
            feature_set: FeatureSet from data.extract_features()

        Returns:
            Tuple of (TrainedModel, PredictionResult on training data)
        """
        from sklearn.preprocessing import StandardScaler

        X = np.array(feature_set.features)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train algorithm-specific model
        if self.algorithm == "isolation_forest":
            sklearn_model = self._train_isolation_forest(X_scaled)
        elif self.algorithm == "local_outlier_factor":
            sklearn_model = self._train_lof(X_scaled)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Create trained model container
        trained = TrainedModel(
            model=sklearn_model,
            scaler=scaler,
            algorithm=self.algorithm,
            hyperparameters=self.hyperparameters,
            scaler_mean=scaler.mean_.tolist(),
            scaler_scale=scaler.scale_.tolist(),
        )

        # Get predictions on training data
        result = self.predict(trained, feature_set)

        return trained, result

    def _train_isolation_forest(self, X_scaled: np.ndarray):
        """Train Isolation Forest model."""
        from sklearn.ensemble import IsolationForest

        hp = self.hyperparameters
        model = IsolationForest(
            n_estimators=hp.get("n_estimators", 100),
            contamination=hp.get("contamination", 0.1),
            random_state=hp.get("random_state", 42),
            n_jobs=hp.get("n_jobs", -1),
        )
        model.fit(X_scaled)
        return model

    def _train_lof(self, X_scaled: np.ndarray):
        """Train Local Outlier Factor model."""
        from sklearn.neighbors import LocalOutlierFactor

        hp = self.hyperparameters
        model = LocalOutlierFactor(
            n_neighbors=hp.get("n_neighbors", 20),
            contamination=hp.get("contamination", 0.1),
            novelty=True,  # Required for predict() on new data
            n_jobs=hp.get("n_jobs", -1),
        )
        model.fit(X_scaled)
        return model

    @staticmethod
    def predict(trained_model: TrainedModel, feature_set: FeatureSet) -> PredictionResult:
        """
        Run inference on feature set using trained model.

        Args:
            trained_model: TrainedModel from train()
            feature_set: FeatureSet to predict on

        Returns:
            PredictionResult with predictions and scores
        """
        X = np.array(feature_set.features)
        X_scaled = trained_model.scaler.transform(X)

        predictions = trained_model.model.predict(X_scaled)
        anomaly_scores = trained_model.model.decision_function(X_scaled)

        anomaly_indices = np.where(predictions == -1)[0]
        n_anomalies = len(anomaly_indices)

        return PredictionResult(
            predictions=predictions,
            anomaly_scores=anomaly_scores,
            anomaly_indices=anomaly_indices,
            anomaly_rate=n_anomalies / len(predictions),
            n_anomalies=n_anomalies,
            n_samples=len(predictions),
        )

    @property
    def description(self) -> str:
        """Human-readable description of model configuration."""
        hp_str = ", ".join(f"{k}={v}" for k, v in self.hyperparameters.items())
        return f"{self.algorithm}({hp_str})"


def get_anomalies(
    prediction: PredictionResult,
    feature_set: FeatureSet,
    top_n: Optional[int] = None,
) -> List[AnomalyResult]:
    """
    Get detailed anomaly results with coin metadata.

    Args:
        prediction: PredictionResult from predict()
        feature_set: FeatureSet with coin_info
        top_n: Limit to top N most anomalous (by score)

    Returns:
        List of AnomalyResult sorted by score (most anomalous first)
    """
    anomalies = []
    for i in prediction.anomaly_indices:
        anomalies.append(AnomalyResult(
            coin_info=feature_set.coin_info[i],
            anomaly_score=float(prediction.anomaly_scores[i]),
        ))

    # Sort by score (most negative = most anomalous)
    anomalies.sort(key=lambda x: x.anomaly_score)

    if top_n:
        anomalies = anomalies[:top_n]

    return anomalies
