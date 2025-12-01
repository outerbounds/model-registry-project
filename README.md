# Crypto Market Anomaly Detection

Real-time anomaly detection on live cryptocurrency market data from CoinGecko.

## Model Lifecycle

This project uses the **champion/challenger** pattern for model lifecycle:

| Status | Meaning | How it's set |
|--------|---------|--------------|
| `candidate` | Newly trained, not yet evaluated | TrainAnomalyFlow |
| `evaluated` | Passed point-in-time quality gates | EvaluateAnomalyFlow |
| `challenger` | Running in parallel with champion | Deployment config |
| `champion` | Primary model, blessed for serving | PromoteAnomalyFlow |
| `retired` | Previous champion, replaced | (when new champion is promoted) |

The idea that will become important is to view these states as ML Asset quality designations, and not confuse such status tags with deployment environments.
- A `champion` model can be deployed to dev, staging, OR prod
- Deployment configs can then reference models by alias (`champion`) or version (`v5`)
- The intent is clean separation of "is this model performant enough to serve production traffic?" from "which code/project branch is this model on?"

### Evaluated vs Challenger

The distinction matters:

- **`evaluated`**: Point-in-time assessment. A single evaluation run passed quality gates. This is necessary but may not be sufficient in most real-world systems.
- **`challenger`**: Trial period. The model runs in parallel with the champion over time, allowing comparison of Evals.log streams. This applies to BOTH:
  - **Batch workflows**: Parallel daily/hourly runs with different model refs
  - **API deployments**: Traffic split between deployment variants

```
candidate ──► evaluated ──► challenger ──► champion
                │               │
          Point-in-time    Trial period
          (single run)     (parallel runs)
```
## Project Structure

```
├── src/                   # Core operational logic (reusable)
│   ├── data.py            # Data fetching, feature engineering
│   ├── model.py           # Model training, inference
│   ├── registry.py        # Asset registration, versioning
│   └── eval.py            # Evaluation, quality gates
│
├── flows/                 # Thin orchestration layers
│   ├── ingest/flow.py     # Orchestrates: fetch → extract → register DataAsset
│   ├── train/flow.py      # Orchestrates: load data → train → register ModelAsset
│   ├── evaluate/flow.py   # Orchestrates: load → predict → gates → update
│   └── promote/flow.py    # Orchestrates: load → promote → publish
│
├── deployments/
│   └── api/app.py         # FastAPI service (uses src modules)
│
├── data/                   # DataAsset configs
│   └── market_snapshot/
└── models/                 # ModelAsset configs
    └── anomaly_detector/
```

### Why `/src` at project root?

Separating operational logic from orchestration enables:

1. **Dependency injection** - Config files can specify which code paths to use
2. **Dynamic loading** - Different asset states trigger different behaviors
3. **Testability** - Test workflows independently of flow orchestration
4. **Reusability** - Same logic in flows, API, and notebooks

```python
# In a flow
from src import data, model, registry
snapshot = data.fetch_market_data()
feature_set = data.extract_features(snapshot)
trained, result = model.train(feature_set)
registry.register_model(prj, "anomaly_detector", ...)

# In API
from src import data, model
snapshot = data.fetch_market_data()
prediction = model.predict_fresh(config, data.extract_features(snapshot))

# In notebook
from src import data
snapshot = data.fetch_market_data()
data.get_top_movers(snapshot)
```

## ML Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                         ML LIFECYCLE                             │
│                                                                  │
│   TrainFlow ──────► EvaluateFlow ──────► PromoteFlow            │
│       │                  │                    │                  │
│       ▼                  ▼                    ▼                  │
│   candidate          evaluated            champion              │
│   (new model)     (gates passed)       (blessed for            │
│                                          serving)               │
│                                                                  │
│   Optional: Deploy as 'challenger' for trial period             │
│   before promoting to champion                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT CONFIGS                            │
│                    (in git branches)                             │
│                                                                  │
│   git:main     →  model_ref: "anomaly_detector:latest"          │
│   git:staging  →  model_ref: "anomaly_detector:champion"        │
│   git:prod     →  model_ref: "anomaly_detector:v5"  (pinned)    │
└─────────────────────────────────────────────────────────────────┘
```

## Flows

### IngestMarketDataFlow

Fetches live data, registers versioned `market_snapshot` DataAsset:

```bash
python flows/ingest/flow.py run
python flows/ingest/flow.py run --num_coins 200
```

### TrainAnomalyFlow

Trains model, registers as `candidate`. Can use registered data or fetch fresh:

```bash
# Quick testing (fetch fresh data)
python flows/train/flow.py run --contamination 0.10

# Production (use registered data asset for lineage)
python flows/train/flow.py run --data_version latest
```

### EvaluateAnomalyFlow

Evaluates candidate, applies quality gates, updates status to `evaluated`:

```bash
python flows/evaluate/flow.py run --max_anomaly_rate 0.20
```

### PromoteAnomalyFlow

Promotes model to `champion` status:

```bash
python flows/promote/flow.py run --version latest
```

## API

The API tries to load `champion` first, falls back to `latest`:

| Endpoint | Description |
|----------|-------------|
| `POST /scan` | Scan market for anomalies |
| `GET /anomalies` | Get detected anomalies |
| `GET /market` | Market overview |
| `GET /model/info` | Model info (status: candidate/evaluated/champion) |
| `POST /model/reload` | Hot-reload model |
| `GET /versions` | List all versions |

## Data Source

**CoinGecko API** - Free, no API key, updates every few minutes.

## Scheduling & Event Triggering

When deployed to Argo Workflows, flows trigger automatically:

```
@schedule(hourly)          @trigger_on_finish         @trigger_on_finish
      │                           │                          │
      ▼                           ▼                          ▼
 IngestFlow ──────────────► TrainFlow ──────────────► EvaluateFlow
                                                            │
                                               publish_event("approval_requested")
                                                            │
                                                     [HUMAN REVIEW]
                                                            │
                                              @project_trigger("model_approved")
                                                            ▼
                                                      PromoteFlow
```

### Deploy to Argo

```bash
# Deploy all flows (one-time)
python flows/ingest/flow.py --with retry argo-workflows create
python flows/train/flow.py --with retry argo-workflows create
python flows/evaluate/flow.py --with retry argo-workflows create
python flows/promote/flow.py --with retry argo-workflows create
```

### Trigger Promotion (after human review)

```bash
python -c "from obproject.project_events import ProjectEvent; \
    ProjectEvent('model_approved', 'crypto_anomaly', 'main').publish()"
```

### Making Promotion Fully Automated

To skip human approval, edit `flows/promote/flow.py`:

```python
# Replace:
@project_trigger(event="model_approved")

# With:
@trigger_on_finish(flow='EvaluateAnomalyFlow')
```

### Making Promotion Manual-Only

Remove the `@project_trigger` decorator entirely from PromoteFlow.

## Human-in-the-Loop Stages

Two stages require human judgment (not automated by default):

| Stage | Who | Decision |
|-------|-----|----------|
| **Promote to Champion** | ML Engineer | "Is this model ready?" (quality/business judgment) |
| **Challenger Deployment** | Platform/MLOps | "Deploy for A/B testing?" (infrastructure concern) |

These are intentionally separate from flow automation because:
- Champion promotion is a **business decision** requiring domain context
- Challenger deployment is an **infrastructure task** (traffic splitting, monitoring)—not a flow

## System Extensions (Quick-Start Notes)

For teams looking to extend this system:

### Approval Queue/Dashboard

**Simplest approach:** Use Slack workflow or a shared spreadsheet.

1. EvaluateFlow's `approval_requested` event triggers a Slack message (via Argo notification)
2. Message includes: model version, anomaly rate, link to UI
3. Reviewer clicks "Approve" button → runs a Slack workflow that calls:
   ```bash
   curl -X POST your-webhook/approve?version=v5
   ```
4. Webhook publishes `model_approved` event

**Why not build a queue?** Events are fire-and-forget. Building state tracking adds complexity. Slack/JIRA already handle approval workflows well.

### Challenger Traffic Splitting

**Simplest approach:** Two deployment configs, manual traffic routing.

1. Create `deployments/api-challenger/` with `model_ref: evaluated`
2. Deploy both APIs to separate endpoints
3. Use nginx/envoy/cloud load balancer to split traffic (e.g., 90/10)
4. Monitor both via existing `/model/info` endpoint
5. After trial period, promote challenger and remove the deployment

**Why not automate?** Traffic splitting is infrastructure-specific (K8s Istio, AWS ALB, etc.). Keep flows focused on ML logic.

### Rollback Automation

**Simplest approach:** Manual PromoteFlow with previous version.

```bash
# Rollback to previous champion
python flows/promote/flow.py run --version v4
```

For automated rollback on drift detection:
1. Create `MonitorFlow` that runs hourly, checks model performance
2. If degraded, publish `rollback_requested` event with `--version` in payload
3. PromoteFlow listens for this event (add second trigger)

**Why start manual?** Automated rollback requires clear rollback criteria. Start with manual, learn what triggers rollback, then automate.

## Customer Requirements Mapping

| Requirement | Implementation |
|-------------|----------------|
| Create model | `models/anomaly_detector/asset_config.toml` |
| Create version | `prj.register_model()` in TrainFlow |
| Link to training job | Annotations: `training_run_id`, `training_flow` |
| Link to training metrics | Annotations: `anomaly_rate`, etc. |
| Link to training dataset | Annotations: `data_source`, `training_timestamp` |
| Link to evaluation job | `Evals.log()` or evaluation_results asset |
| Link to evaluation dataset | Evaluation annotations |
| **Assign status** | `tags: {status: "candidate/evaluated/champion"}` |
| Download by alias | `consume_model_asset(instance="champion")` |
| Request approval | `publish_event("approval_requested")` |
| Compare metrics | Evals comparison / EvaluateFlow card |
| Trial period | Deploy as challenger with parallel execution |

## Why Champion, Not Production?

The word "production" conflates two concepts:

1. **ML Quality**: "Is this model good enough?"
2. **Deployment Environment**: "Where is this code running?"

By using `champion`:
- It's clear this is an ML quality designation
- Deployment configs can reference `champion` from any environment
- No confusion between model status and deploy environment
- Clean mental model for MLOps professionals
