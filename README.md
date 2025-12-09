<img style="display: block; max-width: 100%; height: auto; margin: auto;" alt="system" src="https://github.com/user-attachments/assets/6b4ed419-bc10-482e-9969-3dcf236f0e37" />


An end-to-end ML project demonstrating **model registry patterns** with Outerbounds Projects. Detects anomalies in live cryptocurrency market data from CoinGecko.

## Quick Start

```bash
# Setup
cd model-registry-project-v1
export PYTHONPATH="$PWD"

# Train a model (fetches fresh data)
python flows/train/flow.py run

# Evaluate the model
python flows/evaluate/flow.py run

# Promote to champion
python flows/promote/flow.py run --run_id <TRAIN_RUN_ID>
```

## Project Structure

```
├── src/                        # Reusable ML logic
│   ├── data.py                 # Data fetching from CoinGecko
│   ├── features.py             # Feature engineering
│   ├── model.py                # Isolation Forest training/inference
│   ├── registry.py             # Asset registration, champion management
│   └── eval.py                 # Evaluation metrics, quality gates
│
├── flows/                      # Metaflow orchestration
│   ├── ingest/flow.py          # Fetch → register DataAsset
│   ├── build_dataset/flow.py   # Accumulate snapshots → train/eval split
│   ├── train/flow.py           # Train → register ModelAsset
│   ├── evaluate/flow.py        # Evaluate → quality gates
│   ├── promote/flow.py         # Promote → champion tag
│   └── validate_outcomes/      # Production validation
│
├── deployments/
│   └── dashboard/              # FastAPI dashboard for monitoring
│
├── configs/                    # Environment-specific configs
│   ├── model.json
│   ├── training.json
│   └── evaluation.json
│
└── obproject.toml              # Project configuration
```

## Flows

### IngestMarketDataFlow

Fetches live crypto prices and registers a `market_snapshot` DataAsset.

```bash
python flows/ingest/flow.py run
```

### BuildDatasetFlow

Accumulates snapshots into train/eval datasets with time-based splits.

```bash
# Default: 96hr history, 7hr holdout
python flows/build_dataset/flow.py run

# Quick test (smaller holdout)
python flows/build_dataset/flow.py run --holdout_hours 1
```

**Trigger Form Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `holdout_hours` | `7` | Hours of recent data reserved for evaluation holdout |
| `max_history_hours` | `96` | Maximum history window to include |
| `min_snapshots_per_coin` | `3` | Minimum snapshots required per coin (filters incomplete data) |
| `add_targets` | `false` | Add target columns for supervised learning |

### TrainDetectorFlow

Trains an Isolation Forest model and registers it as a ModelAsset.

```bash
# Use accumulated dataset
python flows/train/flow.py run

# Quick test with fresh data (no dataset lineage)
python flows/train/flow.py run --fresh_data
```

### EvaluateDetectorFlow

Evaluates a model against quality gates using the holdout dataset.

```bash
# Default: evaluate latest model on holdout set
python flows/evaluate/flow.py run

# Use fresh data instead of holdout
python flows/evaluate/flow.py run --eval_data_source fresh

# Evaluate specific model version
python flows/evaluate/flow.py run --candidate_version v5
```

**Trigger Form Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `candidate_version` | `latest` | Model to evaluate. Use `latest`, `latest-N`, or version ID |
| `compare_to` | `latest-1` | Model to compare against. Use `none` to skip comparison |
| `eval_data_source` | *(empty)* | Leave empty for `eval_holdout` (from config). Set `fresh` for live data |
| `eval_data_version` | *(empty)* | Leave empty for `latest` (from config). Set `latest-1` for previous holdout |

Empty fields use defaults from `configs/evaluation.json`. Parameters override config at runtime.

**Version specifiers:** `latest`, `latest-1`, `latest-2`, etc. resolve to actual version IDs at runtime.

### PromoteModelFlow

Promotes a model to champion using Metaflow run tags.

```bash
# Promote by training run ID
python flows/promote/flow.py run --version 186362

# Promote latest model
python flows/promote/flow.py run --version latest
```

**Trigger Form Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `version` | *(required)* | Training run ID to promote (e.g., `186362`) or `latest` |
| `alias` | `champion` | Tag to apply (typically `champion`) |
| `model_name` | `anomaly_detector` | Model asset name |

## Model Lifecycle

```
TrainFlow ──────► EvaluateFlow ──────► PromoteFlow
    │                  │                    │
    ▼                  ▼                    ▼
candidate          evaluated            champion
(new model)     (gates passed)    (tagged for serving)
```

### Champion Management

Champions are tracked via **Metaflow run tags**, not Asset API tags:

```python
from metaflow import Flow, Run

# Find current champion
for run in Flow("TrainDetectorFlow").runs("champion"):
    print(f"Champion: {run.id}")
    break

# Promote a new champion
Run(f"TrainDetectorFlow/{run_id}").add_tag("champion")
```

This enables efficient server-side queries to find the current champion model.

## Configuration

### obproject.toml

```toml
project = "crypto_anomaly"
title = "Crypto Market Anomaly Detection"

# Map branches to environments
[branch_to_environment]
"main" = "production"
"*" = "dev"

# Environment-specific configs
[environments.production.flow_configs]
model_config = "configs/model.json"
training_config = "configs/training.json"
```

### Local Development

#### Cross-Branch Data Access

By default, local runs read and write to your user branch. To use production data from `main` while writing to your branch:

```toml
# obproject.toml
[dev-assets]
branch = "main"
```

This is essential for **new branches** that lack historical data for training. The `[dev-assets]` config enables:
- **Read** from `main` (production datasets, snapshots)
- **Write** to `user.yourname` (isolated experiments)

| Context | Write Branch | Read Branch |
|---------|--------------|-------------|
| Deployed (Argo) | Deployment branch | Same as write |
| Local dev with `[dev-assets]` | User branch | `main` |
| Local dev without `[dev-assets]` | User branch | Same as write |

#### Inspecting Datasets Across Branches

Use the `inspect_datasets.py` script to check data availability:

```bash
# Compare all branches
python scripts/inspect_datasets.py

# Detailed analysis of main branch
python scripts/inspect_datasets.py --branch main --detailed

# Compare specific branches
python scripts/inspect_datasets.py --compare main test.feature_prediction
```

Example output:
```
training_dataset:
  Branch                       Rows  Snapshots  Range
  main                         6695         67  2025-12-05 to 2025-12-07
  test.feature_prediction       N/A        N/A  N/A
```

## Deployment

### Deploy Flows to Argo

```bash
python flows/ingest/flow.py argo-workflows create
python flows/train/flow.py argo-workflows create
python flows/evaluate/flow.py argo-workflows create
python flows/promote/flow.py argo-workflows create
```

### Dashboard

The FastAPI dashboard in `deployments/dashboard/` displays model performance and anomaly predictions.

## Data Source

**CoinGecko API** - Free tier, no API key required, updates every few minutes.
