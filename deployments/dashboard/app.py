"""
Crypto Anomaly Detection Dashboard

A web dashboard for monitoring the ML pipeline and model registry.

Pages:
- / (Overview) - Pipeline health, latest model, latest evaluation
- /data - Data explorer (snapshots, datasets)
- /models - Model registry (versions, compare)
- /cards - Flow cards viewer

Simplified model lifecycle:
- latest = candidate (just trained, under evaluation)
- latest-1 = champion (previous version, proven)
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
    """Format long version strings to a truncated form.

    Shows a shortened version for display, with full version available via tooltip.
    """
    if not version_str:
        return "-"
    v = str(version_str)

    # If it's a simple version like "v5" or "5", return as-is
    if len(v) < 25:
        return f"v{v}" if not v.startswith("v") else v

    # For task pathspec format, extract Flow/run_id
    if "_task_" in v:
        parts = v.split("_")
        try:
            task_idx = parts.index("task")
            flow_name = parts[task_idx + 1] if len(parts) > task_idx + 1 else ""
            run_id = parts[task_idx + 2] if len(parts) > task_idx + 2 else ""

            # For Argo runs, find a unique identifier
            if run_id == "argo":
                # Look for UUID-like part
                uuid_parts = [p for p in parts if len(p) >= 8 and "-" in p]
                if uuid_parts:
                    return f"{flow_name[:15]}/{uuid_parts[0][:8]}"
                return f"{flow_name[:15]}/argo"
            else:
                return f"{flow_name[:15]}/{run_id}"
        except (ValueError, IndexError):
            pass

    # Fallback: just truncate
    return v[:25] + "..."


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


def get_pipeline_status(asset: Asset, branch: str):
    """Get pipeline health status for display."""
    from src import registry

    status = {
        "project": PROJECT,
        "branch": branch,
        "model": None,
        "data": None,
        "champion": None,
        "overall": "unknown",
    }

    # Check model (latest = candidate)
    try:
        ref = registry.load_model(asset, "anomaly_detector", instance="latest")
        status["model"] = {
            "status": "healthy",
            "version": ref.version,
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

    # Check champion (latest-1)
    try:
        ref = registry.load_model(asset, "anomaly_detector", instance="latest-1")
        status["champion"] = {
            "status": "available",
            "version": ref.version,
        }
    except Exception:
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


def get_model_versions(asset: Asset, branch: str, limit=20):
    """Get list of model versions from Metaflow training runs.

    Uses Metaflow Client API to fetch run artifacts directly.
    This provides complete version history with all training metadata.

    Args:
        asset: Asset client (for fallback)
        branch: Project branch to filter runs by (e.g., 'test.feature_prediction')
        limit: Maximum number of versions to return
    """
    from metaflow import Flow, namespace

    versions = []

    # Look up champion by alias tag (scoped to this branch)
    current_champion_run = get_champion_run_id(branch=branch)

    # Set project namespace to see all runs (dev + argo) for this project,
    # then filter by project_branch tag to get branch-specific runs
    namespace(f"project:{PROJECT}")

    # Get version history from Metaflow runs, filtered by project_branch tag
    try:
        flow = Flow('TrainDetectorFlow')
        branch_tag = f"project_branch:{branch}"
        for i, run in enumerate(flow.runs(branch_tag)):
            if i >= limit:
                break

            # Only include successful runs
            if not run.successful:
                continue

            run_id = run.id

            # Get run artifacts from the train step
            try:
                train_step = run['train']
                for task in train_step:
                    # Access artifacts
                    config = dict(task.data.model_config)
                    prediction = task.data.prediction
                    feature_set = task.data.feature_set
                    data_source = task.data.data_source

                    hp = config.get('hyperparameters', {})

                    versions.append({
                        "version": f"Train/{run_id}",
                        "alias": "champion" if str(run_id) == str(current_champion_run) else None,
                        "algorithm": config.get("algorithm", "isolation_forest"),
                        "anomaly_rate": _float(prediction.anomaly_rate) if hasattr(prediction, 'anomaly_rate') else None,
                        "training_samples": _int(feature_set.n_samples) if hasattr(feature_set, 'n_samples') else None,
                        "silhouette_score": None,  # Computed in evaluate flow
                        "score_gap": None,
                        "training_timestamp": run.created_at.isoformat() if run.created_at else None,
                        "training_run_id": run_id,
                        "data_source": data_source,
                        "contamination": _float(hp.get("contamination")),
                        "n_estimators": _int(hp.get("n_estimators")),
                    })
                    break  # Only need first task
            except Exception as e:
                print(f"[DEBUG] Failed to get artifacts for run {run_id}: {e}")
                # Add minimal entry for run without detailed artifacts
                versions.append({
                    "version": f"Train/{run_id}",
                    "alias": "champion" if str(run_id) == str(current_champion_run) else None,
                    "algorithm": "isolation_forest",
                    "anomaly_rate": None,
                    "training_samples": None,
                    "silhouette_score": None,
                    "score_gap": None,
                    "training_timestamp": run.created_at.isoformat() if run.created_at else None,
                    "training_run_id": run_id,
                    "data_source": None,
                    "contamination": None,
                    "n_estimators": None,
                })

    except Exception as e:
        print(f"[DEBUG] Failed to get Metaflow runs: {e}")
        # Fall back to just the latest model from asset
        try:
            ref = registry.load_model(asset, "anomaly_detector", version="latest")
            ann = ref.annotations
            versions.append({
                "version": ref.version,
                "alias": ref.alias,
                "algorithm": ann.get("algorithm"),
                "anomaly_rate": _float(ann.get("anomaly_rate")),
                "training_samples": _int(ann.get("training_samples")),
                "silhouette_score": _float(ann.get("silhouette_score")) if ann.get("silhouette_score") else None,
                "score_gap": _float(ann.get("score_gap")) if ann.get("score_gap") else None,
                "training_timestamp": ann.get("training_timestamp"),
                "training_run_id": ann.get("training_run_id"),
                "data_source": ann.get("data_source"),
            })
        except Exception:
            pass

    return versions


def get_latest_evaluation(asset: Asset):
    """Get latest evaluation result from evaluation_results data asset."""
    try:
        ref = asset.consume_data_asset("evaluation_results", instance="latest")
        props = ref.get("data_properties", {})
        ann = props.get("annotations", {})

        passed = ann.get("passed", False)
        if isinstance(passed, str):
            passed = passed.lower() == "true"

        return {
            "passed": passed,
            "model_name": ann.get("model_name"),
            "model_version": ann.get("model_version"),
            "compared_to_version": ann.get("compared_to_version"),
            "eval_dataset": ann.get("eval_dataset"),
            "anomaly_rate": _float(ann.get("eval_anomaly_rate")),
            "silhouette_score": _float(ann.get("silhouette_score")),
            "score_gap": _float(ann.get("score_gap")),
            "evaluated_by_run_id": ann.get("evaluated_by_run_id"),
        }
    except Exception:
        return None


def get_outcome_metrics_for_model(model_version: str):
    """Get outcome validation metrics for a model from ValidateOutcomesFlow runs.

    These are the ground-truth metrics that answer: "Did this model's predictions
    actually predict crashes?"

    Returns dict with crash_recall, anomaly_precision, false_alarm_rate, etc.
    """
    try:
        from metaflow import Flow

        flow = Flow('ValidateOutcomesFlow')

        # Search for validation runs that validated this model version
        for run in flow.runs():
            if not run.successful:
                continue

            try:
                # Check if this run validated the model we're interested in
                end_step = run['end']
                for task in end_step:
                    if not hasattr(task.data, 'metrics') or task.data.metrics is None:
                        continue

                    metrics = task.data.metrics

                    # Check if this validation included our model version
                    # (ValidateOutcomesFlow validates predictions from multiple models)
                    if hasattr(task.data, 'outcomes'):
                        outcomes = task.data.outcomes
                        model_versions = set(o.model_version for o in outcomes if hasattr(o, 'model_version'))

                        # Look for matching model version (e.g., "TrainDetectorFlow/186089")
                        version_match = any(model_version in v or v in model_version for v in model_versions)
                        if not version_match:
                            continue

                    return {
                        "crash_recall": _float(metrics.crash_recall) if hasattr(metrics, 'crash_recall') else None,
                        "anomaly_precision": _float(metrics.anomaly_precision) if hasattr(metrics, 'anomaly_precision') else None,
                        "false_alarm_rate": _float(metrics.false_alarm_rate) if hasattr(metrics, 'false_alarm_rate') else None,
                        "total_predictions": _int(metrics.total_predictions) if hasattr(metrics, 'total_predictions') else None,
                        "total_crashes": _int(metrics.total_crashes) if hasattr(metrics, 'total_crashes') else None,
                        "validated_at": run.created_at.isoformat() if run.created_at else None,
                        "validation_run_id": run.id,
                    }
            except Exception:
                continue

        return None
    except Exception as e:
        print(f"[DEBUG] get_outcome_metrics error: {e}")
        return None


def get_model_details(run_id: str, alias: str = None):
    """Get detailed model info from a training run.

    Args:
        run_id: The Metaflow run ID (e.g., "186088")
        alias: Optional alias label for display (e.g., "champion", "candidate")

    Returns dict with:
    - Model version and hyperparameters
    - Training data source (resolved version, not alias)
    - Data snapshot range and count
    - Outcome validation metrics if available
    """
    from metaflow import Flow

    try:
        flow = Flow('TrainDetectorFlow')
        run = flow[run_id]

        if not run.successful:
            return None

        train_step = run['train']
        for task in train_step:
            config = dict(task.data.model_config)
            prediction = task.data.prediction
            feature_set = task.data.feature_set
            data_source = task.data.data_source

            hp = config.get('hyperparameters', {})

            details = {
                "version": f"Train/{run_id}",
                "alias": alias,
                "algorithm": config.get("algorithm"),
                "n_estimators": _int(hp.get("n_estimators")),
                "contamination": _float(hp.get("contamination")),
                "anomaly_rate": _float(prediction.anomaly_rate) if hasattr(prediction, 'anomaly_rate') else None,
                "training_samples": _int(feature_set.n_samples) if hasattr(feature_set, 'n_samples') else None,
                "training_run_id": run_id,
                "data_source": data_source,
                "training_timestamp": run.created_at.isoformat() if run.created_at else None,
            }

            # Get outcome validation metrics if available
            outcome_metrics = get_outcome_metrics_for_model(f"TrainDetectorFlow/{run_id}")
            if outcome_metrics:
                details["outcome_metrics"] = outcome_metrics

            return details

    except Exception as e:
        print(f"[DEBUG] get_model_details error for run {run_id}: {e}")
        return None


def get_champion_run_id(branch: str = None) -> str:
    """Get the training run ID of the current champion model.

    Uses Metaflow's native tagging to find the run tagged 'champion',
    optionally scoped to a specific project branch.

    Args:
        branch: Project branch to filter by (e.g., 'test.feature_prediction').
                If None, returns champion across all branches.

    Returns the run ID string, or None if no champion exists.
    """
    from metaflow import Flow, namespace

    try:
        # Set project namespace to see all runs (dev + argo)
        namespace(f"project:{PROJECT}")
        flow = Flow("TrainDetectorFlow")

        # Build tag filter: always require 'champion', optionally filter by branch
        if branch:
            # Find champion within this branch
            branch_tag = f"project_branch:{branch}"
            for run in flow.runs(branch_tag):
                if "champion" in run.tags:
                    return run.id
        else:
            # Find champion across all branches
            for run in flow.runs("champion"):
                return run.id
    except Exception:
        pass

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
    """Overview page - pipeline health, latest model, latest evaluation."""
    branch = get_branch_from_request(request)
    asset = get_asset_client(branch)

    status = get_pipeline_status(asset, branch)
    evaluation = get_latest_evaluation(asset)

    # Get champion details (latest-1)
    champion_details = None
    try:
        from src import registry
        ref = registry.load_model(asset, "anomaly_detector", instance="latest-1")
        ann = ref.annotations
        champion_details = {
            "version": ref.version,
            "algorithm": ann.get("algorithm"),
            "anomaly_rate": _float(ann.get("anomaly_rate")),
            "training_samples": _int(ann.get("training_samples")),
            "silhouette_score": _float(ann.get("silhouette_score")),
        }
    except Exception:
        pass

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

    return templates.TemplateResponse("data.html", get_template_context(
        request, branch,
        data_assets=data_assets,
    ))


@app.get("/models", response_class=HTMLResponse)
async def model_registry(request: Request):
    """Model registry page - versions, compare."""
    branch = get_branch_from_request(request)
    asset = get_asset_client(branch)

    # Get versions with error handling (filtered by branch)
    versions = []
    error_message = None
    try:
        versions = get_model_versions(asset, branch=branch, limit=20)
    except Exception as e:
        error_message = f"Failed to load model versions: {str(e)}"
        print(f"[ERROR] {error_message}")

    # Get champion details with full lineage (scoped to branch)
    champion_run_id = get_champion_run_id(branch=branch)
    champion = get_model_details(champion_run_id, alias="champion") if champion_run_id else None

    # Count champions to validate uniqueness
    champion_count = sum(1 for v in versions if v.get("alias") == "champion")

    return templates.TemplateResponse("models.html", get_template_context(
        request, branch,
        versions=versions,
        error_message=error_message,
        champion=champion,
        champion_count=champion_count,
    ))




# =============================================================================
# Card Embedding Endpoints
# =============================================================================

# Flow name mapping for card lookups
FLOW_NAME_MAP = {
    "train": "TrainDetectorFlow",
    "evaluate": "EvaluateDetectorFlow",
    "ingest": "IngestMarketDataFlow",
    "build_dataset": "BuildDatasetFlow",
    "validate_outcomes": "ValidateOutcomesFlow",
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
        card_steps = ['train', 'evaluate', 'register', 'build_and_write', 'compute_metrics']
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
        flow_alias: Short flow name (train, evaluate, ingest, build_dataset, validate_outcomes)

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
        flow_alias: Short flow name (train, evaluate, ingest, build_dataset)
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
    from metaflow import Flow, namespace

    branch = get_branch_from_request(request)

    # Set project namespace to see all runs (dev + argo) for this project
    namespace(f"project:{PROJECT}")

    # Get recent runs with cards for each flow, filtered by branch
    cards_info = []
    branch_tag = f"project_branch:{branch}"

    for alias, flow_name in FLOW_NAME_MAP.items():
        try:
            flow = Flow(flow_name)
            # Find latest successful run for this branch
            run = None
            for r in flow.runs(branch_tag):
                if r.successful:
                    run = r
                    break

            if run:
                # Get detailed card info for this run
                step_cards = get_flow_cards_info(flow_name, run.id)
                cards_info.append({
                    "alias": alias,
                    "flow_name": flow_name,
                    "run_id": run.id,
                    "created_at": run.created_at.isoformat() if run.created_at else None,
                    "url": f"/cards/{alias}/{run.id}?branch={branch}",
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
async def api_compare(request: Request, v1: str, v2: str, branch: str = None):
    """Compare two model versions using Metaflow run data."""
    from metaflow import Flow, namespace
    import numpy as np

    # Set project namespace to see all runs
    namespace(f"project:{PROJECT}")

    def get_version_data(version_str: str) -> dict:
        """Extract full model data from a version string like 'vTrain/186089'.

        Returns all lineage info useful for comparison:
        - Model hyperparameters
        - Training data source (resolved version)
        - Data snapshot range and count
        - Quality metrics (score gap, anomaly rate)
        - Outcome validation metrics (if available)
        """
        # Parse version string (e.g., "vTrain/186089" -> run_id=186089)
        v = version_str.lstrip('v')
        if '/' in v:
            _, run_id = v.split('/', 1)
        else:
            run_id = v

        flow = Flow('TrainDetectorFlow')
        run = flow[run_id]

        train_step = run['train']
        for task in train_step:
            config = dict(task.data.model_config)
            prediction = task.data.prediction
            feature_set = task.data.feature_set
            data_source = task.data.data_source

            hp = config.get('hyperparameters', {})

            # Compute quality metrics from scores
            scores = prediction.anomaly_scores
            preds = prediction.predictions
            normal_scores = scores[preds == 1]
            anomaly_scores = scores[preds == -1]
            score_gap = float(normal_scores.mean() - anomaly_scores.mean()) if len(anomaly_scores) > 0 else None

            # Get training timestamp
            created_at = run.created_at.isoformat() if run.created_at else None

            # Get data timestamp (when the training data was created)
            data_timestamp = feature_set.timestamp if hasattr(feature_set, 'timestamp') else None

            # Get data snapshot range/count if available (from register step artifacts)
            data_snapshot_start = None
            data_snapshot_end = None
            data_snapshot_count = None
            try:
                register_step = run['register']
                for reg_task in register_step:
                    if hasattr(reg_task.data, 'data_snapshot_range'):
                        snapshot_range = reg_task.data.data_snapshot_range
                        if snapshot_range:
                            data_snapshot_start = snapshot_range[0]
                            data_snapshot_end = snapshot_range[1]
                    if hasattr(reg_task.data, 'data_snapshot_count'):
                        data_snapshot_count = reg_task.data.data_snapshot_count
                    break
            except Exception:
                pass

            result = {
                "version": f"Train/{run_id}",
                "alias": None,
                "algorithm": config.get("algorithm", "isolation_forest"),
                "training_run_id": run_id,
                "data_source": data_source,
                "data_timestamp": data_timestamp,
                "data_snapshot_start": data_snapshot_start,
                "data_snapshot_end": data_snapshot_end,
                "data_snapshot_count": data_snapshot_count,
                "training_samples": feature_set.n_samples if hasattr(feature_set, 'n_samples') else None,
                "anomaly_rate": float(prediction.anomaly_rate) if hasattr(prediction, 'anomaly_rate') else None,
                "score_gap": score_gap,
                "score_std": float(scores.std()) if len(scores) > 0 else None,
                "contamination": float(hp.get("contamination")) if hp.get("contamination") else None,
                "n_estimators": int(hp.get("n_estimators")) if hp.get("n_estimators") else None,
                "created_at": created_at,
            }

            # Get outcome validation metrics if available
            outcome_metrics = get_outcome_metrics_for_model(f"TrainDetectorFlow/{run_id}")
            if outcome_metrics:
                result["outcome_metrics"] = outcome_metrics

            return result

        raise ValueError(f"No data found for run {run_id}")

    try:
        data1 = get_version_data(v1)
        data2 = get_version_data(v2)

        return {
            "v1": data1,
            "v2": data2,
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/models/promote")
async def api_promote(request: Request, version: str, alias: str = "champion"):
    """Promote a model version to an alias (e.g., champion).

    NOTE: The Outerbounds Asset SDK does not currently support updating tags
    on existing asset versions. In production, this would trigger the PromoteFlow
    which re-registers the model with the alias tag set.
    """
    # The Asset SDK doesn't support tag updates after registration.
    # To promote a model, use the PromoteFlow:
    #   python flows/promote/flow.py run --version=<run_id> --alias=champion
    run_id = version.split('/')[-1]
    command = f"python flows/promote/flow.py run --version={run_id} --alias={alias}"
    return {
        "success": False,
        "message": (
            f"Tag updates not supported via API. "
            f"To promote Train/{run_id} to {alias}, run:\n\n"
            f"{command}"
        ),
        "action": "run_flow",
        "command": command,
    }


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
