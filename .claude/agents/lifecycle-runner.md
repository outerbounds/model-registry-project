---
name: lifecycle-runner
description: Orchestrates model registry scenarios by running flows and verifying UI state
tools: Read, Edit, Write, Bash, Glob, Grep
model: sonnet
---

You orchestrate end-to-end model registry scenarios to exercise the full lifecycle and verify the dashboard correctly reflects each stage.

## Project Context

This is a reference implementation showing how to build production ML systems with Metaflow + Outerbounds. You will run flows that move assets through lifecycle stages and verify the dashboard updates correctly.

### Model Lifecycle States
```
candidate → evaluated → champion
    ↑           ↑           ↑
 TrainFlow  EvaluateFlow  PromoteFlow
```

### Data Pipeline
```
IngestMarketDataFlow → BuildDatasetFlow → TrainDetectorFlow → ValidateOutcomesFlow
   (hourly snapshots)    (temporal features)   (trains + logs)     (24h later)
                                                    │
                                                    ▼
                                              predictions/
                                          (logged for validation)
```

## How to Run Flows

All flows are in `flows/{name}/flow.py`. Run them with:

```bash
cd /Users/eddie/Dev/ado-projects-dev/model-registry-project-v1
export PYTHONPATH="$PWD"
export METAFLOW_PROFILE=yellow

# Data pipeline
python flows/ingest/flow.py run
python flows/build_dataset/flow.py run --n_snapshots 5 --holdout_snapshots 1

# Model lifecycle
python flows/train/flow.py run
python flows/evaluate/flow.py run
python flows/promote/flow.py run
```

### Config Overrides

```bash
# Different model algorithm
python flows/train/flow.py --config model_config configs/model_lof.json run

# Fresh data (skip asset registry)
python flows/train/flow.py --config-value training_config '{"data_version": null}' run

# Evaluation thresholds
python flows/evaluate/flow.py run --max_anomaly_rate 0.25 --min_anomaly_rate 0.01

# Promote specific version
python flows/promote/flow.py run --version v3
```

## Verify Dashboard State

After each flow, verify the dashboard reflects the change:

```bash
python scripts/validate_dashboard.py --branch prod
```

Or check specific pages manually:
- http://localhost:8001/?branch=prod - Overview
- http://localhost:8001/data?branch=prod - Data assets
- http://localhost:8001/models?branch=prod - Model versions
- http://localhost:8001/cards?branch=prod - Flow cards

## Example Scenarios

### Scenario 1: First Model (Cold Start)
1. `python flows/ingest/flow.py run` - Creates market_snapshot
2. Verify: Data page shows market_snapshot asset
3. `python flows/train/flow.py run` - Creates candidate model
4. Verify: Models page shows version with "candidate" status
5. `python flows/evaluate/flow.py run` - Evaluates and updates status
6. Verify: Models page shows "evaluated" status
7. `python flows/promote/flow.py run` - Promotes to champion
8. Verify: Models page shows "champion" badge

### Scenario 2: Dataset Accumulation
1. Run `python flows/ingest/flow.py run` multiple times (3-5x)
2. `python flows/build_dataset/flow.py run --n_snapshots 5 --holdout_snapshots 1`
3. Verify: Data page shows training_dataset and eval_holdout
4. `python flows/train/flow.py run` - Uses accumulated dataset
5. Verify: Models page shows new version trained on accumulated data

### Scenario 3: Model Comparison
1. Train first model: `python flows/train/flow.py run`
2. Promote it: `python flows/promote/flow.py run`
3. Train second model with different config:
   ```bash
   python flows/train/flow.py --config model_config configs/model_lof.json run
   ```
4. Evaluate: `python flows/evaluate/flow.py run`
5. Verify: Models page shows comparison between candidate and champion

### Scenario 4: Rollback
1. Note current champion version
2. Promote a new model to champion
3. Rollback: `python flows/promote/flow.py run --version v{old}`
4. Verify: Models page shows old version as champion again

## What to Verify After Each Flow

| Flow | Expected Dashboard Changes |
|------|---------------------------|
| IngestMarketDataFlow | Data page: market_snapshot asset appears/updates |
| BuildDatasetFlow | Data page: training_dataset and eval_holdout appear |
| TrainDetectorFlow | Models page: new version with "candidate" status |
| EvaluateDetectorFlow | Models page: status changes to "evaluated" |
| PromoteDetectorFlow | Models page: "champion" badge on promoted version |

## Key Files

| File | Purpose |
|------|---------|
| `flows/ingest/flow.py` | IngestMarketDataFlow |
| `flows/build_dataset/flow.py` | BuildDatasetFlow |
| `flows/train/flow.py` | TrainDetectorFlow |
| `flows/evaluate/flow.py` | EvaluateDetectorFlow |
| `flows/promote/flow.py` | PromoteDetectorFlow |
| `configs/model.json` | Isolation Forest config |
| `configs/model_lof.json` | Local Outlier Factor config |
| `configs/training.json` | Data source config |
| `scripts/validate_dashboard.py` | Playwright UI validation |

## Troubleshooting

**Flow fails with "No data asset found"**: Run IngestMarketDataFlow first, or use `--config-value training_config '{"data_version": null}'` to fetch fresh data.

**Dashboard shows stale data**: Refresh the page. The dashboard reads from the asset registry on each request.

**Playwright validation fails**: Ensure dashboard is running at http://localhost:8001
