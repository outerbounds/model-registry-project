"""
Model Registry - Asset registration, versioning, and champion management.

Core capabilities demonstrated:
1. Create/register model versions with lineage (via Asset API)
2. Load models by version
3. Compare any two model versions
4. Track evaluation results linked to specific versions
5. Champion management via Metaflow run tags (not Asset tags)

Note: Champion/alias management uses Metaflow's native run tagging system,
not Asset API tags. This is because Asset API doesn't support tag-based queries.
"""

from typing import Dict, Optional, Any, List
from dataclasses import dataclass


@dataclass
class ModelRef:
    """Reference to a registered model version."""
    version: str
    annotations: Dict[str, Any]

    @classmethod
    def from_asset_ref(cls, ref: Dict) -> "ModelRef":
        """Create from consume_model_asset result."""
        props = ref.get("model_properties", {})

        return cls(
            version=ref.get("id"),
            annotations=props.get("annotations", {}),
        )

    @property
    def algorithm(self) -> Optional[str]:
        return self.annotations.get("algorithm")

    @property
    def training_run_id(self) -> Optional[str]:
        return self.annotations.get("training_run_id")

    @property
    def training_flow(self) -> Optional[str]:
        return self.annotations.get("training_flow")

    @property
    def data_source(self) -> Optional[str]:
        return self.annotations.get("data_source")

    @property
    def alias(self) -> Optional[str]:
        """Alias is determined by Metaflow run tags, not stored in Asset."""
        return None  # Use get_champion_run_id() to check if this is champion


def _get_asset(prj_or_asset):
    """Get Asset from ProjectContext or return Asset directly."""
    if hasattr(prj_or_asset, "asset"):
        return prj_or_asset.asset
    return prj_or_asset


# =============================================================================
# Model Registration
# =============================================================================

def register_model(
    prj_or_asset,
    name: str,
    annotations: Dict[str, Any],
    description: str = "",
) -> None:
    """
    Register a new model version.

    Args:
        prj_or_asset: ProjectFlow.prj (ProjectContext) or Asset
        name: Model asset name (e.g., "iris_classifier", "anomaly_detector")
        annotations: Model metadata including:
            - algorithm: Model type/algorithm
            - hyperparameters: Training hyperparameters
            - training_run_id: Flow run ID for lineage
            - training_flow: Flow name for lineage
            - data_source: Training dataset reference
            - training_samples: Number of training samples
            - metrics: Training metrics (accuracy, loss, etc.)
        description: Human-readable description

    Note:
        Champion/alias management is handled separately via Metaflow run tags.
        See set_champion_run() and get_champion_run_id().
    """
    # Get Asset
    if hasattr(prj_or_asset, "asset"):
        asset = prj_or_asset.asset
    elif hasattr(prj_or_asset, "register_model_asset"):
        asset = prj_or_asset
    else:
        raise TypeError(f"Expected ProjectContext or Asset, got {type(prj_or_asset)}")

    asset.register_model_asset(
        name,
        kind="sklearn",
        annotations=annotations,
    )


# =============================================================================
# Model Loading
# =============================================================================

def load_model(
    prj_or_asset,
    name: str,
    version: Optional[str] = None,
    instance: Optional[str] = None,
) -> ModelRef:
    """
    Load a model by version.

    Args:
        prj_or_asset: ProjectContext or Asset
        name: Model asset name
        version: Specific version ("latest", "latest-1", or version ID)
        instance: Alias for version parameter (for compatibility)

    Returns:
        ModelRef with version info and annotations

    Raises:
        Exception: If model not found

    Examples:
        # Load latest version
        model = load_model(asset, "iris_classifier", version="latest")

        # Load specific version
        model = load_model(asset, "iris_classifier", version="v5")

    Note:
        To load the champion model, first get the champion run ID via
        get_champion_run_id(), then load by that run's version.
    """
    # Support both 'version' and 'instance' parameters
    version = version or instance or "latest"

    asset = _get_asset(prj_or_asset)
    ref = asset.consume_model_asset(name, instance=version)

    return ModelRef.from_asset_ref(ref)


def list_model_versions(
    prj_or_asset,
    name: str,
    limit: int = 20,
) -> List[ModelRef]:
    """
    List model versions (most recent first).

    Args:
        prj_or_asset: ProjectContext or Asset
        name: Model asset name
        limit: Maximum versions to return

    Returns:
        List of ModelRef, most recent first
    """
    asset = _get_asset(prj_or_asset)
    versions = []
    seen = set()

    for i in range(limit):
        try:
            instance = "latest" if i == 0 else f"latest-{i}"
            ref = asset.consume_model_asset(name, instance=instance)
            model_ref = ModelRef.from_asset_ref(ref)

            if model_ref.version not in seen:
                versions.append(model_ref)
                seen.add(model_ref.version)
        except Exception:
            break

    return versions


# =============================================================================
# Model Comparison
# =============================================================================

def compare_models(
    model1: ModelRef,
    model2: ModelRef,
) -> Dict[str, Any]:
    """
    Compare two model versions.

    Args:
        model1: First model reference
        model2: Second model reference

    Returns:
        Dict with comparison of parameters, metrics, and lineage
    """
    def safe_get(d, key, default=None):
        return d.get(key, default) if d else default

    ann1 = model1.annotations
    ann2 = model2.annotations

    return {
        "v1": {
            "version": model1.version,
            "algorithm": model1.algorithm,
            "training_run_id": model1.training_run_id,
            "data_source": model1.data_source,
            "training_samples": safe_get(ann1, "training_samples"),
            "anomaly_rate": safe_get(ann1, "anomaly_rate"),
            "silhouette_score": safe_get(ann1, "silhouette_score"),
            "score_gap": safe_get(ann1, "score_gap"),
        },
        "v2": {
            "version": model2.version,
            "algorithm": model2.algorithm,
            "training_run_id": model2.training_run_id,
            "data_source": model2.data_source,
            "training_samples": safe_get(ann2, "training_samples"),
            "anomaly_rate": safe_get(ann2, "anomaly_rate"),
            "silhouette_score": safe_get(ann2, "silhouette_score"),
            "score_gap": safe_get(ann2, "score_gap"),
        },
    }


# =============================================================================
# Data Assets (for completeness)
# =============================================================================

def register_market_data(
    prj,
    n_samples: int,
    n_features: int,
    timestamp: str,
    run_metadata: Dict[str, Any],
) -> None:
    """Register market data snapshot as DataAsset."""
    prj.register_data(
        "market_snapshot",
        "snapshot",
        annotations={
            "n_samples": n_samples,
            "n_features": n_features,
            "timestamp": timestamp,
            "data_source": "coingecko",
            **run_metadata,
        },
    )


def load_market_data(
    prj_or_asset,
    instance: str = "latest",
) -> Dict[str, Any]:
    """Load market data snapshot."""
    asset = _get_asset(prj_or_asset)
    return asset.consume_data_asset("market_snapshot", instance=instance)


# =============================================================================
# Evaluation Results
# =============================================================================

def register_evaluation(
    prj,
    model_name: str,
    model_version: str,
    passed: bool,
    metrics: Dict[str, Any],
    eval_dataset: Optional[str] = None,
    compared_to_version: Optional[str] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Register evaluation results linked to a specific model version.

    Args:
        prj: ProjectFlow.prj
        model_name: Name of the model that was evaluated
        model_version: Version that was evaluated
        passed: Whether quality gates passed
        metrics: Evaluation metrics (anomaly_rate, silhouette_score, etc.)
        eval_dataset: Reference to evaluation dataset
        compared_to_version: Version this was compared against (if any)
        run_metadata: Flow/run metadata for lineage
    """
    annotations = {
        "model_name": model_name,
        "model_version": model_version,
        "passed": passed,
        "compared_to_version": compared_to_version,
        "eval_dataset": eval_dataset,
        **metrics,
        **(run_metadata or {}),
    }

    prj.register_data(
        "evaluation_results",
        "evaluation_record",
        annotations=annotations,
    )


def publish_event(
    prj,
    event_name: str,
    payload: Dict[str, Any],
) -> None:
    """Publish lifecycle event."""
    prj.publish_event(event_name, payload=payload)


# =============================================================================
# Champion Management (via Metaflow run tags)
# =============================================================================

def get_champion_run_id(flow_name: str = "TrainDetectorFlow") -> Optional[str]:
    """
    Get the run ID of the current champion model.

    Uses Metaflow's native tagging to find runs tagged 'champion'.

    Args:
        flow_name: The training flow name

    Returns:
        The run ID of the champion, or None if no champion set
    """
    from metaflow import Flow

    try:
        flow = Flow(flow_name)
        # Query runs with the 'champion' tag
        for run in flow.runs("champion"):
            return run.id
    except Exception:
        pass

    return None


def set_champion_run(
    run_id: str,
    flow_name: str = "TrainDetectorFlow",
    project: str = "crypto_anomaly",
) -> Optional[str]:
    """
    Set a training run as the champion by tagging it.

    Removes 'champion' tag from any previous champion and adds it to the new one.

    Args:
        run_id: The run ID to promote
        flow_name: The training flow name
        project: Project name for namespace scoping

    Returns:
        The previous champion's run ID, or None if no previous champion
    """
    from metaflow import Flow, Run, namespace

    # Use project namespace to see all runs (dev + argo)
    namespace(f"project:{project}")

    previous_champion = None

    # Find and untag current champion
    try:
        flow = Flow(flow_name)
        for run in flow.runs("champion"):
            previous_champion = run.id
            run.remove_tag("champion")
            break  # Only one champion at a time
    except Exception:
        pass

    # Tag the new champion
    try:
        run = Run(f"{flow_name}/{run_id}")
        run.add_tag("champion")
    except Exception as e:
        raise RuntimeError(f"Failed to tag run {run_id} as champion: {e}")

    return previous_champion
