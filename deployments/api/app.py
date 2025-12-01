"""
Crypto Market Anomaly Detection API - Enhanced

Real-time anomaly detection on live cryptocurrency data from CoinGecko.

API Categories:
- /inference/ - Run anomaly detection on live market data
- /data/ - Data assets (snapshots, datasets, schema)
- /models/ - Model registry (versions, champion, compare, promote)
- /pipeline/ - Pipeline status (flows, events, schedules)
- /eval/ - Evaluation details (gates, metrics, history)

Model Lifecycle:
- candidate: Just trained, not yet evaluated
- evaluated: Passed point-in-time quality gates
- challenger: Running in parallel with champion for comparison
- champion: Primary model (this API loads champion first)

Core Use Cases Demonstrated:
- Create/version models with training job lineage
- Link model versions to training/evaluation metrics and datasets
- Assign tags/aliases (e.g., "champion")
- Download model artifacts by version or alias
- Request approval for model promotion
- Compare model parameters and metrics
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from enum import Enum

from fastapi import FastAPI, Query, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from obproject.assets import Asset
import os
import statistics


# =============================================================================
# App Configuration
# =============================================================================

app = FastAPI(
    title="Crypto Anomaly Detection API",
    description="Real-time anomaly detection on live cryptocurrency market data with full asset management",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Initialize asset client
asset = Asset(
    project=os.environ.get("OB_PROJECT", "crypto_anomaly"),
    branch=os.environ.get("OB_BRANCH", "main"),
    read_only=True,
)

# Project/branch for display
PROJECT = os.environ.get("OB_PROJECT", "crypto_anomaly")
BRANCH = os.environ.get("OB_BRANCH", "main")

# Cached model and results
cache = {
    "model_ref": None,
    "last_scan": None,
    "anomalies": [],
    "market_data": [],
}


# =============================================================================
# Pydantic Models for Request/Response
# =============================================================================

class ScanRequest(BaseModel):
    num_coins: int = Field(100, ge=10, le=250, description="Number of coins to scan")


class PromoteRequest(BaseModel):
    version: str = Field(..., description="Version to promote (e.g., 'v5' or 'latest')")
    reason: Optional[str] = Field(None, description="Reason for promotion")


class ApprovalRequest(BaseModel):
    version: str = Field(..., description="Version requesting approval")
    requester: Optional[str] = Field(None, description="Who is requesting approval")
    notes: Optional[str] = Field(None, description="Notes for approver")


# =============================================================================
# Startup Event
# =============================================================================

@app.on_event("startup")
async def load_model():
    """Load anomaly detector model on startup."""
    from src import registry

    global cache

    # Try champion first
    try:
        cache["model_ref"] = registry.load_model(asset, "anomaly_detector", instance="champion")
        print(f"Loaded anomaly_detector v{cache['model_ref'].version} (champion)")
        return
    except Exception:
        pass

    # Fallback to latest
    try:
        cache["model_ref"] = registry.load_model(asset, "anomaly_detector", instance="latest")
        print(f"Loaded anomaly_detector v{cache['model_ref'].version} (status: {cache['model_ref'].status.value})")
    except Exception as e:
        print(f"No model loaded: {e}")


# =============================================================================
# Helper Functions
# =============================================================================

def _float(val, default=0.0):
    """Safely convert value to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _int(val, default=0):
    """Safely convert value to int."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _get_storage_stores():
    """Get SnapshotStore and DatasetStore instances."""
    from src.storage import SnapshotStore, DatasetStore
    return SnapshotStore(PROJECT, BRANCH), DatasetStore(PROJECT, BRANCH)


# =============================================================================
# INFERENCE ENDPOINTS (/inference/)
# =============================================================================

inference_router = APIRouter(prefix="/inference", tags=["Inference"])


@inference_router.post("/scan")
async def scan_market(num_coins: int = Query(100, ge=10, le=250)):
    """
    Scan current market for anomalies.

    Fetches live data from CoinGecko and runs anomaly detection.
    Returns detected anomalies sorted by anomaly score.
    """
    from src import data, model

    global cache

    if cache["model_ref"] is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Fetch fresh data
    snapshot = data.fetch_market_data(num_coins=num_coins)
    feature_set = data.extract_features(snapshot)

    # Get model config and run prediction
    config = model.config_from_annotations(cache["model_ref"].annotations)
    prediction = model.predict_fresh(config, feature_set)

    # Get anomalies
    anomalies = model.get_anomalies(prediction, feature_set)
    anomaly_list = [
        {
            **a.coin_info,
            "anomaly_score": a.anomaly_score,
            "is_anomaly": True,
        }
        for a in anomalies
    ]

    # Update cache
    cache["last_scan"] = snapshot.timestamp
    cache["anomalies"] = anomaly_list
    cache["market_data"] = feature_set.coin_info

    return {
        "status": "scanned",
        "timestamp": cache["last_scan"],
        "coins_scanned": feature_set.n_samples,
        "anomalies_detected": prediction.n_anomalies,
        "anomaly_rate": prediction.anomaly_rate,
        "model_version": cache["model_ref"].version,
        "top_anomalies": [
            {"symbol": a["symbol"], "price_change_24h": a["price_change_24h"], "score": a["anomaly_score"]}
            for a in anomaly_list[:5]
        ],
    }


@inference_router.get("/anomalies")
async def get_anomalies(limit: int = Query(20, ge=1, le=100)):
    """Get detected anomalies from the latest scan."""
    if not cache["anomalies"]:
        return {
            "anomalies": [],
            "count": 0,
            "last_scan": cache["last_scan"],
            "message": "No anomalies detected. Run POST /inference/scan to refresh.",
        }

    return {
        "anomalies": cache["anomalies"][:limit],
        "count": len(cache["anomalies"]),
        "last_scan": cache["last_scan"],
        "model_version": cache["model_ref"].version if cache["model_ref"] else None,
    }


@inference_router.get("/market")
async def get_market_overview():
    """Get current market overview from latest scan."""
    if not cache["market_data"]:
        raise HTTPException(
            status_code=404,
            detail="No market data. Run POST /inference/scan first."
        )

    data = cache["market_data"]

    # Calculate market stats
    total_market_cap = sum(c["market_cap"] for c in data)
    total_volume = sum(c["total_volume"] for c in data)
    avg_change_24h = statistics.mean(c["price_change_24h"] for c in data)

    # Top gainers/losers
    sorted_by_change = sorted(data, key=lambda x: x["price_change_24h"], reverse=True)

    return {
        "timestamp": cache["last_scan"],
        "coins_tracked": len(data),
        "total_market_cap": total_market_cap,
        "total_24h_volume": total_volume,
        "avg_24h_change": avg_change_24h,
        "top_gainers": [
            {"symbol": c["symbol"], "change_24h": c["price_change_24h"]}
            for c in sorted_by_change[:5]
        ],
        "top_losers": [
            {"symbol": c["symbol"], "change_24h": c["price_change_24h"]}
            for c in sorted_by_change[-5:][::-1]
        ],
        "anomalies_count": len(cache["anomalies"]),
    }


# =============================================================================
# DATA ENDPOINTS (/data/)
# =============================================================================

data_router = APIRouter(prefix="/data", tags=["Data Assets"])


@data_router.get("/assets")
async def list_data_assets():
    """
    List all data assets (market_snapshot, training_dataset, eval_holdout).

    Returns metadata about each asset type and their recent versions.
    """
    assets_info = []

    # Check for each known data asset type
    for asset_name in ["market_snapshot", "training_dataset", "eval_holdout"]:
        try:
            # Get latest version
            ref = asset.consume_data_asset(asset_name, instance="latest")
            props = ref.get("data_properties", {})
            annotations = props.get("annotations", {})

            assets_info.append({
                "name": asset_name,
                "latest_version": ref.get("id"),
                "description": _get_asset_description(asset_name),
                "annotations": annotations,
                "created_by": ref.get("created_by", {}),
            })
        except Exception:
            # Asset doesn't exist yet
            assets_info.append({
                "name": asset_name,
                "latest_version": None,
                "description": _get_asset_description(asset_name),
                "annotations": {},
                "message": "No versions registered yet",
            })

    return {"assets": assets_info, "count": len(assets_info)}


def _get_asset_description(name: str) -> str:
    """Get description for known asset types."""
    descriptions = {
        "market_snapshot": "Single point-in-time market data from CoinGecko",
        "training_dataset": "Accumulated Parquet dataset from N snapshots for training",
        "eval_holdout": "Time-split holdout dataset for evaluation",
    }
    return descriptions.get(name, "")


@data_router.get("/assets/{name}")
async def get_data_asset(name: str, version: str = Query("latest", description="Version or 'latest'")):
    """
    Get details for a specific data asset.

    Includes annotations, lineage (which flow produced it), and version info.
    """
    try:
        ref = asset.consume_data_asset(name, instance=version)
        props = ref.get("data_properties", {})
        annotations = props.get("annotations", {})

        return {
            "name": name,
            "version": ref.get("id"),
            "description": _get_asset_description(name),
            "annotations": annotations,
            "created_by": ref.get("created_by", {}),
            "lineage": {
                "flow": annotations.get("ingest_flow") or annotations.get("builder_flow"),
                "run_id": annotations.get("ingest_run_id") or annotations.get("builder_run_id"),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Asset not found: {e}")


@data_router.get("/assets/{name}/versions")
async def list_data_asset_versions(name: str, limit: int = Query(10, ge=1, le=50)):
    """
    List versions of a data asset with lineage info.

    Shows which flow/run produced each version.
    """
    versions = []
    for i in range(limit):
        try:
            instance = "latest" if i == 0 else f"latest-{i}"
            ref = asset.consume_data_asset(name, instance=instance)
            props = ref.get("data_properties", {})
            annotations = props.get("annotations", {})

            versions.append({
                "version": ref.get("id"),
                "timestamp": annotations.get("timestamp") or annotations.get("created_at"),
                "n_samples": _int(annotations.get("n_samples") or annotations.get("total_samples")),
                "lineage": {
                    "flow": annotations.get("ingest_flow") or annotations.get("builder_flow"),
                    "run_id": annotations.get("ingest_run_id") or annotations.get("builder_run_id"),
                },
            })
        except Exception:
            break

    return {"asset": name, "versions": versions, "count": len(versions)}


@data_router.get("/snapshots")
async def list_snapshots(limit: int = Query(24, ge=1, le=100)):
    """
    List raw Parquet snapshots in storage.

    These are the hourly market data snapshots stored in cloud storage.
    """
    try:
        snapshot_store, _ = _get_storage_stores()
        paths = snapshot_store.list_snapshots(limit=limit)

        snapshots = []
        for path in paths:
            # Parse path to extract date/hour info
            # Format: .../snapshots/dt=YYYY-MM-DD/hour=HH/snapshot_{run_id}.parquet
            parts = path.split("/")
            date_part = next((p for p in parts if p.startswith("dt=")), None)
            hour_part = next((p for p in parts if p.startswith("hour=")), None)
            filename = parts[-1]

            snapshots.append({
                "path": path,
                "date": date_part[3:] if date_part else None,
                "hour": hour_part[5:] if hour_part else None,
                "filename": filename,
            })

        return {
            "snapshots": snapshots,
            "count": len(snapshots),
            "storage_path": snapshot_store.base_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list snapshots: {e}")


@data_router.get("/snapshots/stats")
async def get_snapshot_stats():
    """
    Get aggregated statistics about stored snapshots.

    Returns count, date range, and estimated total rows.
    """
    try:
        snapshot_store, _ = _get_storage_stores()
        paths = snapshot_store.list_snapshots()

        if not paths:
            return {
                "count": 0,
                "date_range": None,
                "storage_path": snapshot_store.base_path,
                "message": "No snapshots found",
            }

        # Parse dates from paths
        dates = []
        for path in paths:
            parts = path.split("/")
            date_part = next((p for p in parts if p.startswith("dt=")), None)
            if date_part:
                dates.append(date_part[3:])

        return {
            "count": len(paths),
            "date_range": {
                "start": min(dates) if dates else None,
                "end": max(dates) if dates else None,
            },
            "storage_path": snapshot_store.base_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")


@data_router.get("/datasets")
async def list_datasets():
    """
    List accumulated datasets (training_dataset, eval_holdout).

    These are built by BuildDatasetFlow from accumulated snapshots.
    """
    try:
        _, dataset_store = _get_storage_stores()
        datasets_info = []

        for name in ["training_dataset", "eval_holdout"]:
            versions = dataset_store.list_versions(name)
            if versions:
                # Get latest metadata
                try:
                    _, metadata = dataset_store.load_dataset(name, version=versions[0])
                    datasets_info.append({
                        "name": name,
                        "versions": versions[:5],  # Show last 5
                        "latest": {
                            "version": versions[0],
                            "created_at": metadata.created_at if metadata else None,
                            "total_samples": metadata.total_samples if metadata else None,
                            "snapshot_count": metadata.snapshot_count if metadata else None,
                        },
                    })
                except Exception:
                    datasets_info.append({
                        "name": name,
                        "versions": versions[:5],
                        "latest": None,
                    })
            else:
                datasets_info.append({
                    "name": name,
                    "versions": [],
                    "message": "No versions found",
                })

        return {"datasets": datasets_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {e}")


@data_router.get("/schema")
async def get_feature_schema():
    """
    Get the Parquet schema / feature columns used by the pipeline.

    Returns the feature names and their descriptions.
    """
    from src.data import FEATURE_COLS

    feature_descriptions = {
        "price_change_pct_1h": "1-hour price change percentage",
        "price_change_pct_24h": "24-hour price change percentage",
        "price_change_pct_7d": "7-day price change percentage",
        "market_cap_to_volume": "Market cap / 24h trading volume ratio",
        "ath_change_pct": "Percentage change from all-time high",
        "sparkline_volatility": "7-day price volatility (coefficient of variation)",
    }

    metadata_columns = [
        {"name": "coin_id", "type": "string", "description": "CoinGecko coin identifier"},
        {"name": "symbol", "type": "string", "description": "Trading symbol (e.g., BTC)"},
        {"name": "name", "type": "string", "description": "Full coin name"},
        {"name": "current_price", "type": "float", "description": "Current USD price"},
        {"name": "market_cap", "type": "float", "description": "Market capitalization in USD"},
        {"name": "snapshot_timestamp", "type": "timestamp", "description": "When the snapshot was taken"},
    ]

    return {
        "feature_columns": [
            {"name": col, "type": "float", "description": feature_descriptions.get(col, "")}
            for col in FEATURE_COLS
        ],
        "metadata_columns": metadata_columns,
        "total_columns": len(FEATURE_COLS) + len(metadata_columns),
    }


# =============================================================================
# MODEL ENDPOINTS (/models/)
# =============================================================================

models_router = APIRouter(prefix="/models", tags=["Model Registry"])


@models_router.get("/assets")
async def list_model_assets():
    """
    List all model assets registered in the project.

    Currently only anomaly_detector is supported.
    """
    models_info = []

    for model_name in ["anomaly_detector"]:
        try:
            from src import registry
            ref = registry.load_model(asset, model_name, instance="latest")
            ann = ref.annotations

            # Check for champion
            champion_version = None
            try:
                champion_ref = registry.load_model(asset, model_name, instance="champion")
                champion_version = champion_ref.version
            except Exception:
                pass

            models_info.append({
                "name": model_name,
                "latest_version": ref.version,
                "latest_status": ref.status.value,
                "champion_version": champion_version,
                "algorithm": ann.get("algorithm"),
                "description": "Isolation Forest/LOF model for crypto market anomaly detection",
            })
        except Exception:
            models_info.append({
                "name": model_name,
                "latest_version": None,
                "message": "No versions registered yet",
            })

    return {"models": models_info, "count": len(models_info)}


@models_router.get("/assets/{name}")
async def get_model_asset(name: str, version: str = Query("latest")):
    """
    Get detailed info for a model asset.

    Includes algorithm, hyperparameters, training metrics, and lineage.
    """
    from src import registry

    try:
        v = version if version.startswith(("latest", "v", "champion")) else f"v{version}"
        ref = registry.load_model(asset, name, instance=v)
        ann = ref.annotations

        return {
            "name": name,
            "version": ref.version,
            "status": ref.status.value,
            "algorithm": ann.get("algorithm"),
            "hyperparameters": {
                "n_estimators": ann.get("n_estimators"),
                "contamination": _float(ann.get("contamination")),
                "n_neighbors": ann.get("n_neighbors"),  # For LOF
            },
            "training_metrics": {
                "training_samples": _int(ann.get("training_samples")),
                "anomaly_rate": _float(ann.get("anomaly_rate")),
                "anomalies_detected": _int(ann.get("anomalies_detected")),
            },
            "evaluation_metrics": {
                "eval_anomaly_rate": _float(ann.get("eval_anomaly_rate")),
                "silhouette_score": _float(ann.get("silhouette_score")),
                "score_gap": _float(ann.get("score_gap")),
                "gates_passed": _int(ann.get("gates_passed")),
                "gates_failed": _int(ann.get("gates_failed")),
            },
            "lineage": {
                "training_flow": ann.get("training_flow"),
                "training_run_id": ann.get("training_run_id"),
                "training_timestamp": ann.get("training_timestamp"),
                "data_source": ann.get("data_source"),
                "data_version": ann.get("data_version"),
            },
            "tags": ref.tags,
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {e}")


@models_router.get("/assets/{name}/versions")
async def list_model_versions(name: str, limit: int = Query(10, ge=1, le=50)):
    """
    List versions of a model asset with status progression.

    Shows the lifecycle status of each version.
    """
    from src import registry

    versions = []
    for i in range(limit):
        try:
            instance = "latest" if i == 0 else f"latest-{i}"
            ref = registry.load_model(asset, name, instance=instance)
            ann = ref.annotations

            versions.append({
                "version": ref.version,
                "status": ref.status.value,
                "algorithm": ann.get("algorithm"),
                "anomaly_rate": _float(ann.get("anomaly_rate")),
                "training_samples": _int(ann.get("training_samples")),
                "training_timestamp": ann.get("training_timestamp"),
                "eval_metrics": {
                    "silhouette_score": _float(ann.get("silhouette_score")) if ann.get("silhouette_score") else None,
                    "score_gap": _float(ann.get("score_gap")) if ann.get("score_gap") else None,
                },
            })
        except Exception:
            break

    return {"model": name, "versions": versions, "count": len(versions)}


@models_router.get("/assets/{name}/champion")
async def get_champion(name: str):
    """
    Get the current champion model (if any).

    The champion is the model blessed for production serving.
    """
    from src import registry

    try:
        ref = registry.load_model(asset, name, instance="champion")
        ann = ref.annotations

        return {
            "name": name,
            "version": ref.version,
            "status": ref.status.value,
            "algorithm": ann.get("algorithm"),
            "promoted_from_version": ann.get("promoted_from_version"),
            "promotion_timestamp": ann.get("promotion_timestamp"),
            "training_metrics": {
                "training_samples": _int(ann.get("training_samples")),
                "anomaly_rate": _float(ann.get("anomaly_rate")),
            },
            "evaluation_metrics": {
                "silhouette_score": _float(ann.get("silhouette_score")),
                "score_gap": _float(ann.get("score_gap")),
            },
        }
    except Exception:
        return {
            "name": name,
            "champion": None,
            "message": "No champion model has been promoted yet",
        }


@models_router.get("/assets/{name}/compare")
async def compare_models(
    name: str,
    version_a: str = Query(..., description="First version to compare"),
    version_b: str = Query(..., description="Second version to compare"),
):
    """
    Compare two model versions side-by-side.

    Useful for comparing candidate vs champion or any two versions.
    """
    from src import registry

    try:
        ref_a = registry.load_model(asset, name, instance=version_a)
        ref_b = registry.load_model(asset, name, instance=version_b)

        def extract_metrics(ref):
            ann = ref.annotations
            return {
                "version": ref.version,
                "status": ref.status.value,
                "algorithm": ann.get("algorithm"),
                "hyperparameters": {
                    "n_estimators": ann.get("n_estimators"),
                    "contamination": _float(ann.get("contamination")),
                },
                "training_metrics": {
                    "training_samples": _int(ann.get("training_samples")),
                    "anomaly_rate": _float(ann.get("anomaly_rate")),
                },
                "evaluation_metrics": {
                    "eval_anomaly_rate": _float(ann.get("eval_anomaly_rate")),
                    "silhouette_score": _float(ann.get("silhouette_score")),
                    "score_gap": _float(ann.get("score_gap")),
                    "gates_passed": _int(ann.get("gates_passed")),
                },
            }

        model_a = extract_metrics(ref_a)
        model_b = extract_metrics(ref_b)

        # Calculate differences
        diffs = {
            "anomaly_rate_diff": model_a["training_metrics"]["anomaly_rate"] - model_b["training_metrics"]["anomaly_rate"],
            "silhouette_diff": (model_a["evaluation_metrics"]["silhouette_score"] or 0) - (model_b["evaluation_metrics"]["silhouette_score"] or 0),
            "score_gap_diff": (model_a["evaluation_metrics"]["score_gap"] or 0) - (model_b["evaluation_metrics"]["score_gap"] or 0),
        }

        return {
            "model": name,
            "comparison": {
                version_a: model_a,
                version_b: model_b,
            },
            "differences": diffs,
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Comparison failed: {e}")


@models_router.post("/assets/{name}/promote")
async def promote_model(name: str, request: PromoteRequest):
    """
    Trigger promotion of a model version to champion.

    Publishes a 'model_approved' event which triggers PromoteFlow.
    Note: Requires write access to the project.
    """
    # For API-triggered promotions, we need a writable Asset
    try:
        write_asset = Asset(
            project=PROJECT,
            branch=BRANCH,
            read_only=False,  # Need write access
        )
    except Exception as e:
        raise HTTPException(
            status_code=403,
            detail=f"Write access required for promotion: {e}"
        )

    from src import registry
    from obproject import ProjectEvent

    try:
        # Verify the version exists
        v = request.version if request.version.startswith(("latest", "v")) else f"v{request.version}"
        ref = registry.load_model(asset, name, instance=v)

        # Publish approval event
        event = ProjectEvent("model_approved", project=PROJECT, branch=BRANCH)
        event.publish(payload={
            "model_name": name,
            "version": ref.version,
            "reason": request.reason,
            "triggered_by": "api",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return {
            "status": "approval_event_published",
            "model": name,
            "version": ref.version,
            "message": f"Published 'model_approved' event. PromoteFlow will execute promotion.",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Promotion failed: {e}")


@models_router.post("/assets/{name}/request-approval")
async def request_approval(name: str, request: ApprovalRequest):
    """
    Request approval for a model version promotion.

    This creates an approval request that can be reviewed before promotion.
    """
    from src import registry

    try:
        v = request.version if request.version.startswith(("latest", "v")) else f"v{request.version}"
        ref = registry.load_model(asset, name, instance=v)

        # In a real system, this would create a ticket/notification
        # For now, we just return the request details
        return {
            "status": "approval_requested",
            "model": name,
            "version": ref.version,
            "current_status": ref.status.value,
            "requester": request.requester,
            "notes": request.notes,
            "message": "Approval request created. Use POST /models/assets/{name}/promote to approve.",
            "approve_endpoint": f"/models/assets/{name}/promote",
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {e}")


# =============================================================================
# PIPELINE ENDPOINTS (/pipeline/)
# =============================================================================

pipeline_router = APIRouter(prefix="/pipeline", tags=["Pipeline Status"])


@pipeline_router.get("/status")
async def get_pipeline_status():
    """
    Get overall pipeline health status.

    Shows the current state of each pipeline component.
    """
    status = {
        "project": PROJECT,
        "branch": BRANCH,
        "components": {},
    }

    # Check model availability
    try:
        from src import registry
        ref = registry.load_model(asset, "anomaly_detector", instance="latest")
        status["components"]["model"] = {
            "status": "healthy",
            "latest_version": ref.version,
            "latest_status": ref.status.value,
        }
    except Exception:
        status["components"]["model"] = {
            "status": "no_model",
            "message": "No model versions registered",
        }

    # Check data availability
    try:
        ref = asset.consume_data_asset("market_snapshot", instance="latest")
        props = ref.get("data_properties", {})
        ann = props.get("annotations", {})
        status["components"]["data"] = {
            "status": "healthy",
            "latest_snapshot": ann.get("timestamp"),
            "samples": _int(ann.get("n_samples")),
        }
    except Exception:
        status["components"]["data"] = {
            "status": "no_data",
            "message": "No data snapshots registered",
        }

    # Check for champion
    try:
        from src import registry
        champion = registry.load_model(asset, "anomaly_detector", instance="champion")
        status["components"]["serving"] = {
            "status": "champion_available",
            "champion_version": champion.version,
        }
    except Exception:
        status["components"]["serving"] = {
            "status": "no_champion",
            "message": "No champion model promoted yet",
        }

    # Overall health
    all_healthy = all(
        c.get("status") in ["healthy", "champion_available"]
        for c in status["components"].values()
    )
    status["overall_health"] = "healthy" if all_healthy else "degraded"

    return status


@pipeline_router.get("/flows")
async def list_flow_runs(limit: int = Query(10, ge=1, le=50)):
    """
    List recent flow runs in the pipeline.

    Note: This is a simplified view. Full run details available via Metaflow API.
    """
    # Flow information from asset annotations
    flows = []

    # Get model training runs
    from src import registry
    for i in range(limit):
        try:
            instance = "latest" if i == 0 else f"latest-{i}"
            ref = registry.load_model(asset, "anomaly_detector", instance=instance)
            ann = ref.annotations

            flows.append({
                "flow": ann.get("training_flow", "TrainAnomalyFlow"),
                "run_id": ann.get("training_run_id"),
                "timestamp": ann.get("training_timestamp"),
                "type": "training",
                "output": f"anomaly_detector v{ref.version}",
                "status": ref.status.value,
            })
        except Exception:
            break

    # Get data ingest runs
    for i in range(min(5, limit)):
        try:
            instance = "latest" if i == 0 else f"latest-{i}"
            ref = asset.consume_data_asset("market_snapshot", instance=instance)
            props = ref.get("data_properties", {})
            ann = props.get("annotations", {})

            flows.append({
                "flow": ann.get("ingest_flow", "IngestMarketDataFlow"),
                "run_id": ann.get("ingest_run_id"),
                "timestamp": ann.get("timestamp"),
                "type": "ingest",
                "output": f"market_snapshot v{ref.get('id')}",
            })
        except Exception:
            break

    # Sort by timestamp (newest first)
    flows.sort(key=lambda x: x.get("timestamp") or "", reverse=True)

    return {"flows": flows[:limit], "count": len(flows)}


@pipeline_router.get("/schedule")
async def get_schedule():
    """
    Get scheduled flows in the pipeline.

    Shows what's scheduled and when.
    """
    return {
        "schedules": [
            {
                "flow": "IngestMarketDataFlow",
                "schedule": "hourly",
                "description": "Fetches live market data from CoinGecko",
                "output_asset": "market_snapshot",
            },
            {
                "flow": "BuildDatasetFlow",
                "schedule": "daily",
                "description": "Accumulates 24 snapshots into training/eval datasets",
                "output_assets": ["training_dataset", "eval_holdout"],
            },
        ],
        "event_triggers": [
            {
                "flow": "TrainAnomalyFlow",
                "trigger": "dataset_ready",
                "description": "Trains new model when dataset is built",
            },
            {
                "flow": "EvaluateAnomalyFlow",
                "trigger": "TrainAnomalyFlow completion",
                "description": "Evaluates candidate model with quality gates",
            },
            {
                "flow": "PromoteAnomalyFlow",
                "trigger": "model_approved",
                "description": "Promotes evaluated model to champion",
            },
        ],
    }


# =============================================================================
# EVALUATION ENDPOINTS (/eval/)
# =============================================================================

eval_router = APIRouter(prefix="/eval", tags=["Evaluation"])


@eval_router.get("/latest")
async def get_latest_evaluation():
    """
    Get the latest evaluation results.

    Shows quality gate results and metrics for the most recently evaluated model.
    """
    try:
        ref = asset.consume_data_asset("evaluation_results", instance="latest")
        props = ref.get("data_properties", {})
        ann = props.get("annotations", {})

        return {
            "version": ref.get("id"),
            "decision": ann.get("decision"),
            "candidate_version": ann.get("candidate_model_version"),
            "champion_version": ann.get("champion_model_version"),
            "metrics": {
                "anomaly_rate": _float(ann.get("eval_anomaly_rate")),
                "anomalies_detected": _int(ann.get("eval_anomalies")),
                "samples_evaluated": _int(ann.get("eval_samples")),
                "silhouette_score": _float(ann.get("silhouette_score")),
                "score_gap": _float(ann.get("score_gap")),
                "score_bimodality": _float(ann.get("score_bimodality")),
            },
            "gates": {
                "passed": _int(ann.get("gates_passed")),
                "failed": _int(ann.get("gates_failed")),
                "rate_diff": _float(ann.get("rate_diff")),
            },
            "lineage": {
                "eval_flow": ann.get("eval_flow"),
                "eval_run_id": ann.get("eval_run_id"),
            },
        }
    except Exception:
        return {
            "evaluation": None,
            "message": "No evaluation results found. Run EvaluateAnomalyFlow first.",
        }


@eval_router.get("/gates")
async def get_quality_gates():
    """
    Get quality gate thresholds and their descriptions.

    These are the gates used by EvaluateFlow to validate models.
    """
    return {
        "sanity_gates": [
            {
                "name": "max_anomaly_rate",
                "threshold": "≤ 20%",
                "description": "Catches models declaring too many anomalies (broken contamination or overfitting)",
                "default": 0.20,
            },
            {
                "name": "min_anomaly_rate",
                "threshold": "≥ 2%",
                "description": "Catches models declaring too few anomalies (underfitting or feature collapse)",
                "default": 0.02,
            },
            {
                "name": "rate_stability",
                "threshold": "≤ 10% drift",
                "description": "Catches distribution drift between training and evaluation data",
                "default": 0.10,
            },
        ],
        "quality_gates": [
            {
                "name": "separation_quality",
                "threshold": "≥ 0.00",
                "metric": "silhouette_score",
                "description": "Measures how well anomalies separate from normal points in feature space",
                "range": "[-1, 1]",
            },
            {
                "name": "score_separation",
                "threshold": "≥ 0.05",
                "metric": "score_gap",
                "description": "Gap between mean anomaly and normal scores (larger = more confident)",
            },
        ],
    }


@eval_router.get("/metrics")
async def get_evaluation_metrics_info():
    """
    Get information about unsupervised evaluation metrics.

    These metrics help assess detection quality without ground truth labels.
    """
    return {
        "metrics": [
            {
                "name": "silhouette_score",
                "range": "[-1, 1]",
                "interpretation": "Higher = better separation between anomaly and normal clusters",
                "threshold": "> 0.0 means anomalies form a distinct cluster",
            },
            {
                "name": "score_gap",
                "interpretation": "Absolute difference between mean anomaly and normal scores",
                "threshold": "> 0.05 indicates confident detection",
            },
            {
                "name": "score_bimodality",
                "range": "[0, 1]",
                "interpretation": "How bimodal the score distribution is",
                "threshold": "Higher = clearer separation between normal and anomaly scores",
            },
            {
                "name": "mean_anomaly_score",
                "interpretation": "Average anomaly score for detected anomalies",
                "note": "For Isolation Forest, more negative = more anomalous",
            },
            {
                "name": "mean_normal_score",
                "interpretation": "Average anomaly score for normal points",
            },
        ],
    }


@eval_router.get("/history")
async def get_evaluation_history(limit: int = Query(10, ge=1, le=50)):
    """
    Get historical evaluation results for trend analysis.

    Shows how evaluation metrics have changed over time.
    """
    history = []

    for i in range(limit):
        try:
            instance = "latest" if i == 0 else f"latest-{i}"
            ref = asset.consume_data_asset("evaluation_results", instance=instance)
            props = ref.get("data_properties", {})
            ann = props.get("annotations", {})

            history.append({
                "version": ref.get("id"),
                "decision": ann.get("decision"),
                "candidate_version": ann.get("candidate_model_version"),
                "metrics": {
                    "anomaly_rate": _float(ann.get("eval_anomaly_rate")),
                    "silhouette_score": _float(ann.get("silhouette_score")),
                    "score_gap": _float(ann.get("score_gap")),
                },
                "gates_passed": _int(ann.get("gates_passed")),
                "gates_failed": _int(ann.get("gates_failed")),
            })
        except Exception:
            break

    return {"history": history, "count": len(history)}


# =============================================================================
# LEGACY ENDPOINTS (backwards compatibility)
# =============================================================================

# Keep old endpoints for backwards compatibility
@app.post("/scan")
async def legacy_scan(num_coins: int = Query(100, ge=10, le=250)):
    """Legacy endpoint - use POST /inference/scan instead."""
    return await scan_market(num_coins)


@app.get("/anomalies")
async def legacy_anomalies(limit: int = Query(20, ge=1, le=100)):
    """Legacy endpoint - use GET /inference/anomalies instead."""
    return await get_anomalies(limit)


@app.get("/market")
async def legacy_market():
    """Legacy endpoint - use GET /inference/market instead."""
    return await get_market_overview()


@app.get("/model/info")
async def model_info():
    """Get info about the serving model."""
    if cache["model_ref"] is None:
        raise HTTPException(status_code=404, detail="No model loaded")

    ref = cache["model_ref"]
    ann = ref.annotations

    return {
        "version": ref.version,
        "status": ref.status.value,
        "algorithm": ann.get("algorithm"),
        "n_estimators": ann.get("n_estimators"),
        "contamination": _float(ann.get("contamination")),
        "training_samples": ann.get("training_samples"),
        "training_anomaly_rate": _float(ann.get("anomaly_rate")),
        "training_timestamp": ann.get("training_timestamp"),
        "data_source": ann.get("data_source"),
    }


@app.post("/model/reload")
async def reload_model():
    """Hot-reload model to latest champion or latest version."""
    await load_model()
    if cache["model_ref"] is None:
        raise HTTPException(status_code=503, detail="Failed to load model")
    return {
        "status": "reloaded",
        "version": cache["model_ref"].version,
        "model_status": cache["model_ref"].status.value,
    }


@app.get("/versions")
async def legacy_versions(limit: int = Query(10, ge=1, le=50)):
    """Legacy endpoint - use GET /models/assets/{name}/versions instead."""
    return await list_model_versions("anomaly_detector", limit)


@app.get("/versions/{version}")
async def legacy_get_version(version: str):
    """Legacy endpoint - use GET /models/assets/{name}?version={version} instead."""
    return await get_model_asset("anomaly_detector", version)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": cache["model_ref"] is not None,
        "model_version": cache["model_ref"].version if cache["model_ref"] else None,
        "last_scan": cache["last_scan"],
    }


# =============================================================================
# Register Routers
# =============================================================================

app.include_router(inference_router)
app.include_router(data_router)
app.include_router(models_router)
app.include_router(pipeline_router)
app.include_router(eval_router)


# =============================================================================
# Root Endpoint
# =============================================================================

@app.get("/")
async def root():
    """API overview and available endpoints."""
    return {
        "title": "Crypto Anomaly Detection API",
        "version": "2.0.0",
        "project": PROJECT,
        "branch": BRANCH,
        "endpoints": {
            "inference": {
                "POST /inference/scan": "Scan market for anomalies",
                "GET /inference/anomalies": "Get detected anomalies",
                "GET /inference/market": "Get market overview",
            },
            "data": {
                "GET /data/assets": "List data assets",
                "GET /data/assets/{name}": "Get asset details",
                "GET /data/snapshots": "List raw snapshots",
                "GET /data/datasets": "List accumulated datasets",
                "GET /data/schema": "Get feature schema",
            },
            "models": {
                "GET /models/assets": "List model assets",
                "GET /models/assets/{name}": "Get model details",
                "GET /models/assets/{name}/versions": "List versions",
                "GET /models/assets/{name}/champion": "Get champion",
                "GET /models/assets/{name}/compare": "Compare versions",
                "POST /models/assets/{name}/promote": "Trigger promotion",
                "POST /models/assets/{name}/request-approval": "Request approval",
            },
            "pipeline": {
                "GET /pipeline/status": "Pipeline health",
                "GET /pipeline/flows": "Recent flow runs",
                "GET /pipeline/schedule": "Scheduled flows",
            },
            "eval": {
                "GET /eval/latest": "Latest evaluation",
                "GET /eval/gates": "Quality gate info",
                "GET /eval/metrics": "Metrics info",
                "GET /eval/history": "Evaluation history",
            },
        },
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
