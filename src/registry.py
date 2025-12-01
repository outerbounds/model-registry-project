"""
Asset registration, versioning, and consumption.

This module handles:
- Registering model versions with annotations
- Consuming models by version or alias
- Updating model status (candidate -> evaluated -> champion)
- Event publishing for lifecycle transitions
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    CANDIDATE = "candidate"
    EVALUATED = "evaluated"
    CHALLENGER = "challenger"
    CHAMPION = "champion"
    RETIRED = "retired"


@dataclass
class ModelRef:
    """Reference to a registered model version."""
    version: int
    status: ModelStatus
    annotations: Dict[str, Any]
    tags: Dict[str, str]

    @classmethod
    def from_asset_ref(cls, ref: Dict) -> "ModelRef":
        """Create from consume_model_asset result."""
        props = ref.get("model_properties", {})

        # Tags are at top level as list of {key, value} dicts
        tags_list = ref.get("tags", [])
        tags = {t["key"]: t["value"] for t in tags_list} if tags_list else {}

        status_str = tags.get("status", "unknown")

        try:
            status = ModelStatus(status_str)
        except ValueError:
            status = ModelStatus.CANDIDATE

        return cls(
            version=ref.get("id"),
            status=status,
            annotations=props.get("annotations", {}),
            tags=tags,
        )


def register_model(
    prj_or_asset,
    name: str,
    status: ModelStatus,
    annotations: Dict[str, Any],
    description: str = "",
) -> None:
    """
    Register a new model version.

    Args:
        prj_or_asset: ProjectFlow.prj (ProjectContext) or Asset
        name: Model asset name (e.g., "anomaly_detector")
        status: Initial status (typically CANDIDATE for new models)
        annotations: Model metadata (hyperparameters, metrics, lineage)
        description: Human-readable description
    """
    tags = {
        "status": status.value,
        "algorithm": annotations.get("algorithm", "unknown"),
        "data_source": annotations.get("data_source", "unknown"),
    }

    # Get Asset - either directly or via ProjectContext.asset
    if hasattr(prj_or_asset, "asset"):
        # ProjectContext - get asset from it
        asset = prj_or_asset.asset
    elif hasattr(prj_or_asset, "register_model_asset"):
        # Already an Asset
        asset = prj_or_asset
    else:
        raise TypeError(f"Expected ProjectContext or Asset, got {type(prj_or_asset)}")

    asset.register_model_asset(
        name,
        kind="sklearn",
        annotations=annotations,
        tags=tags,
    )


def _get_asset(prj_or_asset):
    """Get Asset from ProjectContext or return Asset directly."""
    if hasattr(prj_or_asset, "asset"):
        return prj_or_asset.asset
    return prj_or_asset


def load_model(
    prj_or_asset,
    name: str,
    instance: str = "latest",
) -> ModelRef:
    """
    Load a model by version or alias.

    Args:
        prj_or_asset: ProjectContext or Asset
        name: Model asset name
        instance: Version ("v1", "latest", "latest-1") or alias ("champion")

    Returns:
        ModelRef with version info and annotations

    Raises:
        Exception: If model not found
    """
    asset = _get_asset(prj_or_asset)
    ref = asset.consume_model_asset(name, instance=instance)
    return ModelRef.from_asset_ref(ref)


def load_champion_or_latest(
    prj_or_asset,
    name: str,
) -> ModelRef:
    """
    Load champion model, falling back to latest.

    Args:
        prj_or_asset: ProjectContext or Asset
        name: Model asset name

    Returns:
        ModelRef (champion if exists, else latest)
    """
    # Try champion first
    try:
        return load_model(prj_or_asset, name, instance="champion")
    except Exception:
        pass

    # Fall back to latest
    return load_model(prj_or_asset, name, instance="latest")


def update_status(
    prj_or_asset,
    name: str,
    new_status: ModelStatus,
    current_annotations: Dict[str, Any],
    additional_annotations: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Update model status (e.g., candidate -> evaluated).

    Args:
        prj_or_asset: ProjectContext or Asset
        name: Model asset name
        new_status: New status to set
        current_annotations: Current model annotations to preserve
        additional_annotations: Additional annotations to add
    """
    asset = _get_asset(prj_or_asset)
    annotations = {**current_annotations}
    if additional_annotations:
        annotations.update(additional_annotations)

    asset.register_model_asset(
        name,
        kind="sklearn",
        annotations=annotations,
        tags={
            "status": new_status.value,
            "algorithm": annotations.get("algorithm", "unknown"),
            "data_source": annotations.get("data_source", "unknown"),
        },
    )


def promote_to_champion(
    prj_or_asset,
    name: str,
    source_version: int,
    source_annotations: Dict[str, Any],
    promotion_metadata: Dict[str, Any],
) -> None:
    """
    Promote a model version to champion status.

    Args:
        prj_or_asset: ProjectContext or Asset (target for promotion)
        name: Model asset name
        source_version: Version being promoted
        source_annotations: Annotations from source model
        promotion_metadata: Additional metadata (flow name, run id, etc.)
    """
    asset = _get_asset(prj_or_asset)
    annotations = {
        **source_annotations,
        "promoted_from_version": source_version,
        **promotion_metadata,
    }

    asset.register_model_asset(
        name,
        kind="sklearn",
        annotations=annotations,
        tags={
            "status": ModelStatus.CHAMPION.value,
            "algorithm": "isolation_forest",
            "promoted_from": str(source_version),
        },
    )


def register_market_data(
    prj,
    n_samples: int,
    n_features: int,
    timestamp: str,
    run_metadata: Dict[str, Any],
) -> None:
    """
    Register market data snapshot as DataAsset.

    Args:
        prj: ProjectFlow.prj (ProjectContext)
        n_samples: Number of coins in snapshot
        n_features: Number of features extracted
        timestamp: ISO timestamp of data fetch
        run_metadata: Flow/run metadata
    """
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
        tags={
            "data_source": "coingecko",
        },
    )


def load_market_data(
    prj_or_asset,
    instance: str = "latest",
) -> Dict[str, Any]:
    """
    Load market data snapshot.

    Args:
        prj_or_asset: ProjectContext or Asset
        instance: Version to load

    Returns:
        Dict with snapshot metadata
    """
    asset = _get_asset(prj_or_asset)
    return asset.consume_data_asset("market_snapshot", instance=instance)


def register_evaluation(
    prj,
    decision: str,
    candidate_version: int,
    champion_version: Optional[int],
    metrics: Dict[str, Any],
    run_metadata: Dict[str, Any],
) -> None:
    """
    Register evaluation results as data asset.

    Args:
        prj: ProjectFlow.prj
        decision: "approve" or "reject"
        candidate_version: Version that was evaluated
        champion_version: Champion version compared against (if any)
        metrics: Evaluation metrics
        run_metadata: Flow/run metadata
    """
    prj.register_data(
        "evaluation_results",
        "evaluation_record",
        annotations={
            "decision": decision,
            "candidate_model_asset": "anomaly_detector",
            "candidate_model_version": candidate_version,
            "champion_model_version": champion_version,
            **metrics,
            **run_metadata,
        },
        tags={
            "decision": decision,
            "candidate_version": str(candidate_version),
        },
    )


def publish_event(
    prj,
    event_name: str,
    payload: Dict[str, Any],
) -> None:
    """
    Publish lifecycle event.

    Args:
        prj: ProjectFlow.prj
        event_name: Event name (e.g., "approval_requested", "model_promoted")
        payload: Event payload
    """
    prj.publish_event(event_name, payload=payload)
