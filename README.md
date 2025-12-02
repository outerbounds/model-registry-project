<img width="960" height="540" alt="System designs (1)" src="https://github.com/user-attachments/assets/6206ae5a-e33d-4832-9c80-523c75a86da1" />

An end-to-end ML project demonstrating **model registry patterns** with Outerbounds Projects. Detects anomalies in live cryptocurrency market data from CoinGecko.

## Quick Start

```bash
# Setup
cd model-registry-project-v1
export PYTHONPATH="$PWD"
export METAFLOW_PROFILE=yellow  # or your profile

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
# Default: 168hr history, 24hr holdout
python flows/build_dataset/flow.py run

# Quick test (smaller holdout)
python flows/build_dataset/flow.py run --holdout_hours 1
```

### TrainDetectorFlow

Trains an Isolation Forest model and registers it as a ModelAsset.

```bash
# Use accumulated dataset
python flows/train/flow.py run

# Quick test with fresh data (no dataset lineage)
python flows/train/flow.py run --fresh_data
```

### EvaluateDetectorFlow

Evaluates the latest model against quality gates.

```bash
python flows/evaluate/flow.py run
```

### PromoteModelFlow

Promotes a model to champion using Metaflow run tags.

```bash
python flows/promote/flow.py run --run_id <TRAIN_RUN_ID>
```

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

For local testing, read and write use your user branch. For production-like behavior (read from main, write to your branch):

```toml
[dev-assets]
branch = "main"
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
