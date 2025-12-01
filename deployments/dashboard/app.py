"""
Crypto Anomaly Detection Dashboard

A web dashboard for monitoring the ML pipeline and model registry.

Pages:
- / (Overview) - Pipeline health, champion model, latest evaluation
- /data - Data explorer (snapshots, datasets)
- /models - Model registry (versions, compare, promote)
- /scanner - Live anomaly scanner
"""

from pathlib import Path
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Request, Query, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from obproject.assets import Asset

# =============================================================================
# App Configuration
# =============================================================================

app = FastAPI(
    title="Crypto Anomaly Detection Dashboard",
    description="Web dashboard for monitoring the ML pipeline",
    version="1.0.0",
)

# Mount static files (only if directory exists)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def format_version(version_str):
    """Format long version strings to a readable short form.

    Asset versions in obproject are typically task pathspecs:
    - Format: v{timestamp}_task_{FlowName}_{run_id}_{step}_{task_id}
    - Example: v098235478858_task_TrainAnomalyFlow_185193_train_12345

    We extract the flow name and run ID for display.
    """
    if not version_str:
        return "-"
    v = str(version_str)
    # If it's a simple version like "v5" or "5", return as-is
    if len(v) < 20:
        return f"v{v}" if not v.startswith("v") else v
    # For task pathspec format: v{ts}_task_{Flow}_{run}_{step}_{task}
    if "_task_" in v:
        parts = v.split("_")
        # Find the flow name (after "task")
        try:
            task_idx = parts.index("task")
            flow_name = parts[task_idx + 1] if len(parts) > task_idx + 1 else ""
            run_id = parts[task_idx + 2] if len(parts) > task_idx + 2 else ""
            # Clean up flow name (remove common suffixes for brevity)
            short_flow = flow_name.replace("AnomalyFlow", "").replace("Flow", "")
            if not short_flow:
                short_flow = flow_name[:8]
            return f"{short_flow}/{run_id}"
        except (ValueError, IndexError):
            pass
    # Fallback for other formats
    if "_" in v:
        parts = v.split("_")
        if len(parts) >= 4:
            flow_name = parts[2] if len(parts) > 2 else ""
            run_id = parts[3] if len(parts) > 3 else ""
            return f"{flow_name[:12]}.../{run_id}"
    return v[:12] + "..."


def format_version_tooltip(version_str):
    """Return full version for tooltip."""
    return str(version_str) if version_str else ""


# Register template filters
templates.env.filters["format_version"] = format_version
templates.env.filters["format_version_tooltip"] = format_version_tooltip

# Initialize asset client
PROJECT = os.environ.get("OB_PROJECT", "crypto_anomaly")

# Default branch for initial load (user can change in UI)
DEFAULT_BRANCH = os.environ.get("OB_BRANCH", "main")

# Suggested branches shown in dropdown - users can type any branch
# These are just hints, the UI allows entering any branch name
SUGGESTED_BRANCHES = ["main", "prod"]


def get_asset_client(branch: str) -> Asset:
    """Get an Asset client for a specific branch."""
    return Asset(
        project=PROJECT,
        branch=branch,
        read_only=True,
    )


def get_branch_from_request(request: Request) -> str:
    """Extract branch from query params, falling back to default."""
    return request.query_params.get("branch", DEFAULT_BRANCH)


# Cache for scan results (keyed by branch)
cache = {}


# =============================================================================
# Helper Functions
# =============================================================================

def _float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _int(val, default=0):
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def find_champion_model(asset: Asset):
    """Find champion model by checking if latest model has champion status."""
    try:
        # Since there's only one version at a time (each registration replaces),
        # just check if the latest model has champion status
        from src import registry
        ref = registry.load_model(asset, "anomaly_detector", instance="latest")
        if ref.status == registry.ModelStatus.CHAMPION:
            return {
                "version": ref.version,
                "status": ref.status.value,
                "annotations": ref.annotations,
                "tags": ref.tags,
            }
    except Exception:
        pass
    return None


def get_pipeline_status(asset: Asset, branch: str):
    """Get pipeline health status for display."""
    status = {
        "project": PROJECT,
        "branch": branch,
        "model": None,
        "data": None,
        "champion": None,
        "overall": "unknown",
    }

    # Check model
    try:
        from src import registry
        ref = registry.load_model(asset, "anomaly_detector", instance="latest")
        status["model"] = {
            "status": "healthy",
            "version": ref.version,
            "model_status": ref.status.value,
        }
    except Exception:
        status["model"] = {"status": "no_model"}

    # Check data
    try:
        ref = asset.consume_data_asset("market_snapshot", instance="latest")
        props = ref.get("data_properties", {})
        ann = props.get("annotations", {})
        status["data"] = {
            "status": "healthy",
            "timestamp": ann.get("timestamp"),
            "samples": _int(ann.get("n_samples")),
        }
    except Exception:
        status["data"] = {"status": "no_data"}

    # Check champion - use tag-based lookup instead of instance="champion"
    champion = find_champion_model(asset)
    if champion:
        status["champion"] = {
            "status": "available",
            "version": champion["version"],
        }
    else:
        status["champion"] = {"status": "none"}

    # Overall
    if status["model"] and status["model"].get("status") == "healthy":
        if status["champion"] and status["champion"].get("status") == "available":
            status["overall"] = "healthy"
        else:
            status["overall"] = "degraded"
    else:
        status["overall"] = "unhealthy"

    return status


def get_model_versions(asset: Asset, limit=20):
    """Get list of model versions."""
    from src import registry

    versions = []
    seen_versions = set()

    # Get versions via latest-N iteration
    for i in range(limit):
        try:
            instance = "latest" if i == 0 else f"latest-{i}"
            ref = registry.load_model(asset, "anomaly_detector", instance=instance)

            if ref.version not in seen_versions:
                ann = ref.annotations
                versions.append({
                    "version": ref.version,
                    "status": ref.status.value,
                    "algorithm": ann.get("algorithm"),
                    "anomaly_rate": _float(ann.get("anomaly_rate")),
                    "training_samples": _int(ann.get("training_samples")),
                    "silhouette_score": _float(ann.get("silhouette_score")) if ann.get("silhouette_score") else None,
                    "score_gap": _float(ann.get("score_gap")) if ann.get("score_gap") else None,
                    "training_timestamp": ann.get("training_timestamp"),
                })
                seen_versions.add(ref.version)
        except Exception:
            break

    return versions


def get_latest_evaluation(asset: Asset):
    """Get latest evaluation result from model annotations."""
    try:
        # Evaluation metrics are stored in model annotations when model is evaluated
        from src import registry
        ref = registry.load_model(asset, "anomaly_detector", instance="latest")
        ann = ref.annotations

        # Check if model has been evaluated (has eval_anomaly_rate annotation)
        if not ann.get("eval_anomaly_rate"):
            return None

        # Model was evaluated - determine if it passed based on status
        status = ref.status.value
        passed = status in ("evaluated", "champion")

        return {
            "decision": "approve" if passed else "reject",
            "candidate_version": ref.version,
            "anomaly_rate": _float(ann.get("eval_anomaly_rate")),
            "silhouette_score": _float(ann.get("silhouette_score")) if ann.get("silhouette_score") else 0.0,
            "score_gap": _float(ann.get("score_gap")) if ann.get("score_gap") else 0.0,
            "gates_passed": 5 if passed else 0,  # Approximate - actual gate count not stored
            "gates_failed": 0 if passed else 1,
        }
    except Exception:
        return None


# =============================================================================
# Common template context
# =============================================================================

def get_template_context(request: Request, branch: str, **kwargs):
    """Build common template context with branch info."""
    return {
        "request": request,
        "project": PROJECT,
        "branch": branch,
        "suggested_branches": SUGGESTED_BRANCHES,
        "default_branch": DEFAULT_BRANCH,
        **kwargs,
    }


# =============================================================================
# Pages
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def overview(request: Request):
    """Overview page - pipeline health, champion model, latest evaluation."""
    branch = get_branch_from_request(request)
    asset = get_asset_client(branch)

    status = get_pipeline_status(asset, branch)
    evaluation = get_latest_evaluation(asset)

    # Get champion details using tag-based lookup
    champion_details = None
    champion = find_champion_model(asset)
    if champion:
        ann = champion.get("annotations", {})
        champion_details = {
            "version": champion["version"],
            "algorithm": ann.get("algorithm"),
            "anomaly_rate": _float(ann.get("anomaly_rate")),
            "training_samples": _int(ann.get("training_samples")),
            "silhouette_score": _float(ann.get("silhouette_score")),
        }

    return templates.TemplateResponse("overview.html", get_template_context(
        request, branch,
        status=status,
        evaluation=evaluation,
        champion=champion_details,
    ))


@app.get("/data", response_class=HTMLResponse)
async def data_explorer(request: Request):
    """Data explorer page - snapshots and datasets."""
    branch = get_branch_from_request(request)
    asset = get_asset_client(branch)

    # Get data assets
    data_assets = []
    for name in ["market_snapshot", "training_dataset", "eval_holdout"]:
        try:
            ref = asset.consume_data_asset(name, instance="latest")
            props = ref.get("data_properties", {})
            ann = props.get("annotations", {})
            data_assets.append({
                "name": name,
                "version": ref.get("id"),
                "samples": _int(ann.get("n_samples") or ann.get("total_samples")),
                "timestamp": ann.get("timestamp") or ann.get("created_at"),
            })
        except Exception as e:
            print(f"[DEBUG] Failed to load {name} for branch {branch}: {e}")
            data_assets.append({
                "name": name,
                "version": None,
                "message": "No data yet",
            })

    # Get snapshots from storage
    snapshots = []
    try:
        from src.storage import SnapshotStore
        store = SnapshotStore(PROJECT, branch)
        paths = store.list_snapshots(limit=24)
        for path in paths:
            parts = path.split("/")
            date_part = next((p for p in parts if p.startswith("dt=")), None)
            hour_part = next((p for p in parts if p.startswith("hour=")), None)
            snapshots.append({
                "date": date_part[3:] if date_part else "unknown",
                "hour": hour_part[5:] if hour_part else "unknown",
                "filename": parts[-1],
            })
    except Exception:
        pass

    return templates.TemplateResponse("data.html", get_template_context(
        request, branch,
        data_assets=data_assets,
        snapshots=snapshots,
    ))


@app.get("/models", response_class=HTMLResponse)
async def model_registry(request: Request):
    """Model registry page - versions, compare, promote."""
    branch = get_branch_from_request(request)
    asset = get_asset_client(branch)

    versions = get_model_versions(asset, limit=20)

    # Get champion version using tag-based lookup
    champion_version = None
    champion = find_champion_model(asset)
    if champion:
        champion_version = champion["version"]

    return templates.TemplateResponse("models.html", get_template_context(
        request, branch,
        versions=versions,
        champion_version=champion_version,
    ))




# =============================================================================
# API Endpoints for AJAX
# =============================================================================

@app.get("/api/models/compare")
async def api_compare(request: Request, v1: str, v2: str):
    """Compare two model versions."""
    from src import registry

    branch = get_branch_from_request(request)
    asset = get_asset_client(branch)

    try:
        ref1 = registry.load_model(asset, "anomaly_detector", instance=v1)
        ref2 = registry.load_model(asset, "anomaly_detector", instance=v2)

        def extract(ref):
            ann = ref.annotations
            return {
                "version": ref.version,
                "status": ref.status.value,
                "algorithm": ann.get("algorithm"),
                "anomaly_rate": _float(ann.get("anomaly_rate")),
                "silhouette_score": _float(ann.get("silhouette_score")),
                "score_gap": _float(ann.get("score_gap")),
                "training_samples": _int(ann.get("training_samples")),
            }

        return {
            "v1": extract(ref1),
            "v2": extract(ref2),
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/models/promote")
async def api_promote(request: Request, version: str = Form(...)):
    """Promote a model version to champion."""
    from obproject import ProjectEvent

    branch = get_branch_from_request(request)

    try:
        event = ProjectEvent("model_approved", project=PROJECT, branch=branch)
        event.publish(payload={
            "model_name": "anomaly_detector",
            "version": version,
            "triggered_by": "dashboard",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return {"status": "success", "message": f"Promotion event published for v{version}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
