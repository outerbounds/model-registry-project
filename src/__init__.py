"""
Crypto Anomaly Detection - Core Operations

This module contains the operational logic separated from orchestration:
- data: Data fetching and feature engineering
- model: Model training and inference
- registry: Asset registration and versioning
- eval: Evaluation and quality gates

These can be invoked from:
- Flows (offline batch processing)
- API deployments (online serving)
- Notebooks (interactive development)
- Local testing

NOTE: Submodules are NOT imported at package level to avoid triggering
dependency imports (numpy, sklearn, etc.) before the task environment
is set up. Use explicit imports: `from src import data` or `from src.data import ...`
"""

# IMPORTANT: Must be defined before any imports for Metaflow to detect it
# and include this package in the code package for remote execution
METAFLOW_PACKAGE_POLICY = 'include'

__all__ = ["data", "model", "registry", "eval", "storage", "cards", "features", "predictions"]
