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
    - Example: v098235478858_task_TrainDetectorFlow_185193_train_12345
    - Argo format: v{ts}_task_{FlowName}_argo_cryptoanomaly_prod_{wf}_{uuid}

    We extract meaningful identifiers for display.
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
        try:
            task_idx = parts.index("task")
            flow_name = parts[task_idx + 1] if len(parts) > task_idx + 1 else ""
            run_id = parts[task_idx + 2] if len(parts) > task_idx + 2 else ""

            # Clean up flow name (remove common suffixes for brevity)
            short_flow = flow_name.replace("DetectorFlow", "").replace("AnomalyFlow", "").replace("MarketDataFlow", "Ingest").replace("DatasetFlow", "Dataset").replace("Flow", "")
            if not short_flow:
                short_flow = flow_name[:10]

            # For Argo runs, extract a more useful identifier
            if run_id == "argo":
                # Argo format: argo_cryptoanomaly_prod_traindetectorflow_xyz123
                # Find the UUID at the end (last part that looks like a hash)
                remaining = "_".join(parts[task_idx + 3:])
                # Try to find a short unique ID - look for UUID-like part or last segment
                uuid_parts = [p for p in parts if len(p) >= 8 and "-" in p]
                if uuid_parts:
                    # Use first 8 chars of UUID
                    return f"{short_flow}/{uuid_parts[0][:8]}"
                # Fallback: use timestamp from version
                ts = parts[0][1:] if parts[0].startswith("v") else parts[0]
                if len(ts) >= 12:
                    # Convert epoch-like timestamp to readable format
                    return f"{short_flow}/argo-{ts[-6:]}"
                return f"{short_flow}/argo"
            else:
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
    return v[:16] + "..."


def format_version_tooltip(version_str):
    """Return full version for tooltip."""
    return str(version_str) if version_str else ""


def format_timestamp(ts_str):
    """Format ISO timestamp to a human-readable short form."""
    if not ts_str:
        return "-"
    ts = str(ts_str)
    # Handle ISO format: 2025-12-01T05:41:00.065898+00:00
    if "T" in ts:
        # Extract date and time parts
        date_part = ts.split("T")[0]
        time_part = ts.split("T")[1].split(".")[0] if "." in ts else ts.split("T")[1][:8]
        # Remove timezone if present
        if "+" in time_part:
            time_part = time_part.split("+")[0]
        return f"{date_part} {time_part}"
    return ts[:19] if len(ts) > 19 else ts


# Register template filters
templates.env.filters["format_version"] = format_version
templates.env.filters["format_version_tooltip"] = format_version_tooltip
templates.env.filters["format_timestamp"] = format_timestamp

# Initialize asset client
PROJECT = os.environ.get("OB_PROJECT", "crypto_anomaly")

# Default branch for initial load (user can change in UI)
DEFAULT_BRANCH = os.environ.get("OB_BRANCH", "prod")

# Suggested branches shown in dropdown - users can type any branch
# These are just hints, the UI allows entering any branch name
SUGGESTED_BRANCHES = ["prod"]


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

    # Get versions with error handling
    versions = []
    error_message = None
    try:
        versions = get_model_versions(asset, limit=20)
    except Exception as e:
        error_message = f"Failed to load model versions: {str(e)}"
        print(f"[ERROR] {error_message}")

    # Get champion version using tag-based lookup
    champion_version = None
    try:
        champion = find_champion_model(asset)
        if champion:
            champion_version = champion["version"]
    except Exception as e:
        print(f"[ERROR] Failed to find champion: {e}")

    return templates.TemplateResponse("models.html", get_template_context(
        request, branch,
        versions=versions,
        champion_version=champion_version,
        error_message=error_message,
    ))




# =============================================================================
# Card Embedding Endpoints
# =============================================================================

# Flow name mapping for card lookups
FLOW_NAME_MAP = {
    "train": "TrainDetectorFlow",
    "evaluate": "EvaluateDetectorFlow",
    "promote": "PromoteDetectorFlow",
    "ingest": "IngestMarketDataFlow",
    "build_dataset": "BuildDatasetFlow",
}


def get_card_html(flow_name: str, run_id: str = "latest", step_name: str = None, card_index: int = 0) -> Optional[str]:
    """
    Fetch card HTML for a specific flow run using Metaflow Client API.

    Args:
        flow_name: Metaflow flow class name (e.g., 'TrainDetectorFlow')
        run_id: Run ID or 'latest' for most recent successful run
        step_name: Specific step to get card from (optional)
        card_index: Which card to get if multiple exist (default 0)

    Returns:
        HTML string of the card, or None if not found
    """
    try:
        from metaflow import Flow
        from metaflow.cards import get_cards

        flow = Flow(flow_name)

        # Find the run
        if run_id == "latest":
            run = flow.latest_successful_run
            if run is None:
                return None
        else:
            run = flow[run_id]

        # If specific step requested, use that
        if step_name:
            try:
                step = run[step_name]
                for task in step:
                    cards = get_cards(task)
                    if cards and len(cards) > card_index:
                        return cards[card_index].get()
            except Exception:
                pass
            return None

        # Find a step with a card (typically 'train', 'evaluate', 'register', etc.)
        card_steps = ['train', 'evaluate', 'register', 'build_and_write']
        for step in card_steps:
            try:
                step_obj = run[step]
                for task in step_obj:
                    cards = get_cards(task)
                    if cards and len(cards) > card_index:
                        return cards[card_index].get()
            except Exception:
                continue

        return None
    except Exception as e:
        print(f"[DEBUG] get_card_html error: {e}")
        return None


def get_flow_cards_info(flow_name: str, run_id: str = "latest") -> list:
    """
    Get information about all cards in a flow run.

    Returns list of dicts with step_name, card_index, card_type info.
    """
    try:
        from metaflow import Flow
        from metaflow.cards import get_cards

        flow = Flow(flow_name)

        if run_id == "latest":
            run = flow.latest_successful_run
            if run is None:
                return []
        else:
            run = flow[run_id]

        cards_info = []
        for step in run.steps():
            step_name = step.id
            for task in step:
                cards = get_cards(task)
                for i, card in enumerate(cards):
                    cards_info.append({
                        "step_name": step_name,
                        "card_index": i,
                        "card_type": getattr(card, 'type', 'default'),
                    })
        return cards_info
    except Exception as e:
        print(f"[DEBUG] get_flow_cards_info error: {e}")
        return []


@app.get("/cards/{flow_alias}/latest", response_class=HTMLResponse)
async def card_latest(flow_alias: str):
    """
    Get the card HTML for the latest run of a flow.

    Args:
        flow_alias: Short flow name (train, evaluate, promote, ingest, build_dataset)

    Returns:
        HTML content of the card, or 404 if not found
    """
    flow_name = FLOW_NAME_MAP.get(flow_alias)
    if not flow_name:
        raise HTTPException(status_code=404, detail=f"Unknown flow alias: {flow_alias}")

    html = get_card_html(flow_name, "latest")
    if html is None:
        raise HTTPException(status_code=404, detail=f"No card found for {flow_name}")

    return HTMLResponse(content=html)


@app.get("/cards/{flow_alias}/{run_id}", response_class=HTMLResponse)
async def card_by_run(flow_alias: str, run_id: str):
    """
    Get the card HTML for a specific run of a flow.

    Args:
        flow_alias: Short flow name (train, evaluate, promote, ingest, build_dataset)
        run_id: Metaflow run ID

    Returns:
        HTML content of the card, or 404 if not found
    """
    flow_name = FLOW_NAME_MAP.get(flow_alias)
    if not flow_name:
        raise HTTPException(status_code=404, detail=f"Unknown flow alias: {flow_alias}")

    html = get_card_html(flow_name, run_id)
    if html is None:
        raise HTTPException(status_code=404, detail=f"No card found for {flow_name}/{run_id}")

    return HTMLResponse(content=html)


@app.get("/cards", response_class=HTMLResponse)
async def cards_page(request: Request):
    """
    Cards overview page - list available cards with iframe previews.
    """
    branch = get_branch_from_request(request)

    # Get recent runs with cards for each flow
    cards_info = []
    for alias, flow_name in FLOW_NAME_MAP.items():
        try:
            from metaflow import Flow
            flow = Flow(flow_name)
            run = flow.latest_successful_run
            if run:
                # Get detailed card info for this run
                step_cards = get_flow_cards_info(flow_name, "latest")
                cards_info.append({
                    "alias": alias,
                    "flow_name": flow_name,
                    "run_id": run.id,
                    "created_at": run.created_at.isoformat() if run.created_at else None,
                    "url": f"/cards/{alias}/latest",
                    "step_cards": step_cards,  # List of {step_name, card_index}
                    "card_count": len(step_cards),
                })
        except Exception as e:
            print(f"[DEBUG] cards_page error for {flow_name}: {e}")
            pass

    return templates.TemplateResponse("cards.html", get_template_context(
        request, branch,
        cards=cards_info,
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
