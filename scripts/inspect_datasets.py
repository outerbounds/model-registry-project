#!/usr/bin/env python
"""
Inspect datasets across branches.

This script helps you understand what data is available across different
branches of your project. Useful for:
- Checking if a branch has enough data for training
- Comparing dataset sizes across branches
- Debugging data pipeline issues
- Validating [dev-assets] configuration

Usage:
    # Check all branches
    python scripts/inspect_datasets.py

    # Check specific branch
    python scripts/inspect_datasets.py --branch main

    # Detailed analysis of a dataset
    python scripts/inspect_datasets.py --branch main --dataset training_dataset --detailed

    # Compare two branches
    python scripts/inspect_datasets.py --compare main test.feature_prediction

Requirements:
    - METAFLOW_PROFILE set appropriately
    - PYTHONPATH includes project root
"""

import argparse
import sys
from datetime import datetime
from typing import Optional, List

# Add project root to path
sys.path.insert(0, ".")

from src.storage import DatasetStore, SnapshotStore


def inspect_branch(project: str, branch: str, detailed: bool = False) -> dict:
    """Inspect datasets and snapshots on a branch."""
    result = {
        "branch": branch,
        "datasets": {},
        "snapshots": {"count": 0, "paths": []},
    }

    # Check datasets
    dataset_store = DatasetStore(project, branch)
    for name in ["training_dataset", "eval_holdout"]:
        try:
            table, meta = dataset_store.load_dataset(name, version="latest")
            if table and meta:
                dataset_info = {
                    "rows": table.num_rows,
                    "columns": len(table.column_names),
                    "snapshot_count": meta.snapshot_count,
                    "range_start": meta.snapshot_range[0][:19] if meta.snapshot_range[0] else None,
                    "range_end": meta.snapshot_range[1][:19] if meta.snapshot_range[1] else None,
                    "builder_run_id": meta.builder_run_id,
                }

                if detailed and table.num_rows > 0:
                    import pandas as pd
                    df = table.to_pandas()

                    # Add detailed stats
                    dataset_info["column_names"] = list(df.columns)
                    dataset_info["memory_mb"] = df.memory_usage(deep=True).sum() / 1024 / 1024

                    # Check for duplicates if snapshot_timestamp exists
                    if "snapshot_timestamp" in df.columns:
                        df["snapshot_timestamp"] = pd.to_datetime(df["snapshot_timestamp"])
                        dataset_info["unique_timestamps"] = df["snapshot_timestamp"].nunique()
                        dataset_info["timestamp_range_hours"] = (
                            df["snapshot_timestamp"].max() - df["snapshot_timestamp"].min()
                        ).total_seconds() / 3600

                    # Check for coin diversity if coin_id exists
                    if "coin_id" in df.columns:
                        dataset_info["unique_coins"] = df["coin_id"].nunique()
                        dataset_info["rows_per_coin"] = table.num_rows / df["coin_id"].nunique()

                result["datasets"][name] = dataset_info
            else:
                result["datasets"][name] = {"status": "not_found"}
        except Exception as e:
            result["datasets"][name] = {"status": "error", "error": str(e)}

    # Check snapshots
    try:
        snapshot_store = SnapshotStore(project, branch)
        paths = snapshot_store.list_snapshots()
        result["snapshots"]["count"] = len(paths)
        if detailed:
            result["snapshots"]["paths"] = paths[-10:]  # Last 10
    except Exception as e:
        result["snapshots"]["error"] = str(e)

    return result


def print_branch_summary(result: dict, detailed: bool = False):
    """Pretty print branch inspection results."""
    print(f"\n{'='*60}")
    print(f"Branch: {result['branch']}")
    print(f"{'='*60}")

    print(f"\nSnapshots: {result['snapshots'].get('count', 'error')}")
    if detailed and result["snapshots"].get("paths"):
        print("  Recent paths:")
        for p in result["snapshots"]["paths"][-5:]:
            print(f"    {p.split('/')[-1]}")

    for name, info in result["datasets"].items():
        print(f"\n{name}:")
        if info.get("status") == "not_found":
            print("  Status: NOT FOUND")
        elif info.get("status") == "error":
            print(f"  Status: ERROR - {info.get('error')}")
        else:
            print(f"  Rows: {info['rows']:,}")
            print(f"  Columns: {info['columns']}")
            print(f"  Snapshots: {info['snapshot_count']}")
            print(f"  Range: {info.get('range_start', 'N/A')} to {info.get('range_end', 'N/A')}")
            print(f"  Builder: {info.get('builder_run_id', 'N/A')}")

            if detailed:
                if "memory_mb" in info:
                    print(f"  Memory: {info['memory_mb']:.2f} MB")
                if "unique_timestamps" in info:
                    print(f"  Unique timestamps: {info['unique_timestamps']}")
                    print(f"  Time span: {info['timestamp_range_hours']:.1f} hours")
                if "unique_coins" in info:
                    print(f"  Unique coins: {info['unique_coins']}")
                    print(f"  Rows per coin: {info['rows_per_coin']:.1f}")


def compare_branches(project: str, branches: List[str]):
    """Compare datasets across branches."""
    results = {}
    for branch in branches:
        results[branch] = inspect_branch(project, branch)

    print(f"\n{'='*70}")
    print(f"Dataset Comparison: {' vs '.join(branches)}")
    print(f"{'='*70}")

    # Compare training_dataset
    print("\ntraining_dataset:")
    print(f"  {'Branch':<35} {'Rows':>10} {'Snapshots':>10} {'Range'}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*20}")
    for branch, result in results.items():
        info = result["datasets"].get("training_dataset", {})
        rows = info.get("rows", "N/A")
        snaps = info.get("snapshot_count", "N/A")
        range_str = f"{info.get('range_start', '')[:10]} to {info.get('range_end', '')[:10]}" if info.get("range_start") else "N/A"
        print(f"  {branch:<35} {str(rows):>10} {str(snaps):>10} {range_str}")

    # Compare eval_holdout
    print("\neval_holdout:")
    print(f"  {'Branch':<35} {'Rows':>10} {'Snapshots':>10} {'Range'}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*20}")
    for branch, result in results.items():
        info = result["datasets"].get("eval_holdout", {})
        rows = info.get("rows", "N/A")
        snaps = info.get("snapshot_count", "N/A")
        range_str = f"{info.get('range_start', '')[:10]} to {info.get('range_end', '')[:10]}" if info.get("range_start") else "N/A"
        print(f"  {branch:<35} {str(rows):>10} {str(snaps):>10} {range_str}")


def main():
    parser = argparse.ArgumentParser(description="Inspect datasets across branches")
    parser.add_argument("--project", default="crypto_anomaly", help="Project name")
    parser.add_argument("--branch", help="Specific branch to inspect")
    parser.add_argument("--dataset", help="Specific dataset to inspect")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    parser.add_argument("--compare", nargs="+", help="Compare multiple branches")

    args = parser.parse_args()

    if args.compare:
        compare_branches(args.project, args.compare)
    elif args.branch:
        result = inspect_branch(args.project, args.branch, detailed=args.detailed)
        print_branch_summary(result, detailed=args.detailed)
    else:
        # Default: check common branches
        branches = ["main", "test.feature_prediction", "user.eddie_at_outerbounds.co"]
        compare_branches(args.project, branches)


if __name__ == "__main__":
    main()
