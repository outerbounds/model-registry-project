"""
Reusable Metaflow card components for anomaly detection flows.

This module provides simple, clean visualization functions that return
card-ready components. The goal is minimal code in flows:

    from src.cards import score_distribution_chart, gates_summary
    current.card.append(score_distribution_chart(prediction.scores, prediction.labels))
    current.card.append(gates_summary(eval_result))

Components:
- score_distribution_chart: Dual histogram showing normal vs anomaly score separation
- feature_histograms: Small multiples for feature distributions
- gates_summary: Quality gates with visual pass/fail indicators
- model_comparison_table: Side-by-side candidate vs champion metrics
- top_anomalies_table: Rich table of detected anomalies
- historical_metrics_chart: Cross-run metrics trend using Client API
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass


def score_distribution_chart(
    scores: List[float],
    labels: List[int],
    title: str = "Anomaly Score Distribution",
) -> "VegaChart":
    """
    Create a dual histogram showing score distribution for normal vs anomaly.

    This visualization answers: "How well does the model separate anomalies?"
    A good model shows two distinct peaks with minimal overlap.

    Args:
        scores: Anomaly scores from model (more negative = more anomalous)
        labels: Prediction labels (-1 = anomaly, 1 = normal)
        title: Chart title

    Returns:
        VegaChart component ready to append to card
    """
    from metaflow.cards import VegaChart

    # Build data for Vega
    data = []
    for score, label in zip(scores, labels):
        data.append({
            "score": float(score),
            "type": "Anomaly" if label == -1 else "Normal"
        })

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title,
        "width": 500,
        "height": 250,
        "data": {"values": data},
        "mark": {"type": "bar", "opacity": 0.7},
        "encoding": {
            "x": {
                "field": "score",
                "type": "quantitative",
                "bin": {"maxbins": 30},
                "title": "Anomaly Score"
            },
            "y": {
                "aggregate": "count",
                "title": "Count"
            },
            "color": {
                "field": "type",
                "type": "nominal",
                "scale": {"domain": ["Normal", "Anomaly"], "range": ["#4c78a8", "#e45756"]},
                "title": "Class"
            }
        },
        "config": {
            "legend": {"orient": "top-right"}
        }
    }

    return VegaChart(spec)


def feature_histograms(
    features: List[List[float]],
    feature_names: List[str],
    max_features: int = 6,
) -> "VegaChart":
    """
    Create small multiple histograms for feature distributions.

    This visualization answers: "What do the input features look like?"
    Useful for spotting data quality issues or drift.

    Args:
        features: Feature matrix (samples x features)
        feature_names: Names of features
        max_features: Maximum number of features to show

    Returns:
        VegaChart component with faceted histograms
    """
    from metaflow.cards import VegaChart

    # Build data for Vega (long format for faceting)
    data = []
    for sample in features:
        for i, (name, value) in enumerate(zip(feature_names[:max_features], sample)):
            data.append({
                "feature": name.replace("_", " ").title()[:20],  # Clean name
                "value": float(value)
            })

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Feature Distributions",
        "width": 150,
        "height": 100,
        "data": {"values": data},
        "mark": "bar",
        "encoding": {
            "x": {
                "field": "value",
                "type": "quantitative",
                "bin": {"maxbins": 15},
                "title": None
            },
            "y": {
                "aggregate": "count",
                "title": None
            },
            "color": {"value": "#4c78a8"}
        },
        "facet": {
            "field": "feature",
            "type": "nominal",
            "columns": 3,
            "title": None
        },
        "resolve": {"scale": {"x": "independent"}}
    }

    return VegaChart(spec)


def gates_summary(eval_result: "EvaluationResult") -> "Markdown":
    """
    Create a formatted markdown summary of quality gates with visual indicators.

    This visualization answers: "Did the model pass evaluation?"
    Shows each gate with clear pass/fail status.

    Args:
        eval_result: EvaluationResult from src.eval.run_quality_gates()

    Returns:
        Markdown component with gates summary
    """
    from metaflow.cards import Markdown

    lines = ["## Quality Gates\n"]

    # Overall status banner
    if eval_result.all_passed:
        lines.append("**Status: ALL GATES PASSED**\n")
    else:
        failed_count = eval_result.gates_failed
        lines.append(f"**Status: {failed_count} GATE(S) FAILED**\n")

    # Individual gates
    lines.append("| Gate | Threshold | Actual | Status |")
    lines.append("|------|-----------|--------|--------|")

    for gate in eval_result.gates:
        status = "PASS" if gate.passed else "FAIL"
        lines.append(f"| {gate.name} | {gate.threshold} | {gate.actual} | {status} |")

    # Summary metrics
    lines.append("\n### Metrics Summary\n")
    lines.append(f"- **Anomaly Rate:** {eval_result.anomaly_rate:.1%}")
    lines.append(f"- **Anomalies Detected:** {eval_result.n_anomalies} / {eval_result.n_samples}")

    if eval_result.rate_diff is not None:
        lines.append(f"- **Rate Drift from Training:** {eval_result.rate_diff:.1%}")

    if eval_result.metrics.silhouette_score is not None:
        lines.append(f"- **Silhouette Score:** {eval_result.metrics.silhouette_score:.3f}")

    if eval_result.metrics.score_gap is not None:
        lines.append(f"- **Score Gap:** {eval_result.metrics.score_gap:.3f}")

    return Markdown("\n".join(lines))


def model_comparison_table(
    candidate: Dict[str, Any],
    champion: Optional[Dict[str, Any]] = None,
) -> "Markdown":
    """
    Create a side-by-side comparison table of candidate vs champion.

    This visualization answers: "Is the new model better than the current one?"

    Args:
        candidate: Dict with candidate model metrics (version, anomaly_rate, etc.)
        champion: Optional dict with champion model metrics

    Returns:
        Markdown component with comparison table
    """
    from metaflow.cards import Markdown

    lines = ["## Model Comparison\n"]

    if champion is None:
        lines.append("*No champion model to compare against (first model)*\n")
        lines.append("### Candidate Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key, value in candidate.items():
            if isinstance(value, float):
                lines.append(f"| {key} | {value:.4f} |")
            else:
                lines.append(f"| {key} | {value} |")
    else:
        lines.append("| Metric | Candidate | Champion | Delta |")
        lines.append("|--------|-----------|----------|-------|")

        # Common metrics to compare
        metrics = ["version", "anomaly_rate", "silhouette_score", "score_gap", "training_samples"]

        for metric in metrics:
            cand_val = candidate.get(metric)
            champ_val = champion.get(metric)

            if cand_val is None:
                continue

            if isinstance(cand_val, float):
                cand_str = f"{cand_val:.4f}"
                if champ_val is not None:
                    champ_str = f"{champ_val:.4f}"
                    delta = cand_val - champ_val
                    delta_str = f"{delta:+.4f}"
                else:
                    champ_str = "-"
                    delta_str = "-"
            else:
                cand_str = str(cand_val)
                champ_str = str(champ_val) if champ_val is not None else "-"
                delta_str = "-"

            lines.append(f"| {metric} | {cand_str} | {champ_str} | {delta_str} |")

    return Markdown("\n".join(lines))


def top_anomalies_table(
    anomalies: List["AnomalyResult"],
    limit: int = 10,
) -> "Table":
    """
    Create a rich table of top detected anomalies.

    This visualization answers: "What specific anomalies were detected?"
    Shows coin details, price changes, and anomaly scores.

    Args:
        anomalies: List of AnomalyResult from src.model.get_anomalies()
        limit: Maximum number of anomalies to show

    Returns:
        Table component with anomaly details
    """
    from metaflow.cards import Table

    rows = []
    for a in anomalies[:limit]:
        c = a.coin_info
        rows.append([
            c.get("symbol", "?"),
            c.get("name", "Unknown")[:20],
            f"${c.get('current_price', 0):,.2f}",
            f"{c.get('price_change_24h', 0):+.1f}%",
            f"{c.get('price_change_7d', 0):+.1f}%",
            f"{a.anomaly_score:.3f}"
        ])

    return Table(
        rows,
        headers=["Symbol", "Name", "Price", "24h", "7d", "Score"]
    )


def historical_metrics_chart(
    flow_name: str,
    metric_name: str = "anomaly_rate",
    limit: int = 20,
) -> "VegaChart":
    """
    Create a line chart of metrics across recent runs using Metaflow Client API.

    This visualization answers: "How has this metric changed over time?"
    Useful for detecting drift or model degradation.

    Args:
        flow_name: Name of the flow to query (e.g., 'EvaluateDetectorFlow')
        metric_name: Artifact name to plot (e.g., 'anomaly_rate')
        limit: Maximum number of runs to include

    Returns:
        VegaChart component with historical trend
    """
    from metaflow import Flow
    from metaflow.cards import VegaChart

    data = []

    try:
        flow = Flow(flow_name)

        for i, run in enumerate(flow.runs()):
            if i >= limit:
                break

            if not run.successful:
                continue

            # Try to extract the metric from run data
            try:
                # Check common step names for the metric
                for step_name in ['evaluate', 'train', 'end']:
                    try:
                        step = run[step_name]
                        for task in step:
                            task_data = task.data

                            # Look for the metric in various forms
                            value = None
                            if hasattr(task_data, metric_name):
                                value = getattr(task_data, metric_name)
                            elif hasattr(task_data, 'prediction') and hasattr(task_data.prediction, metric_name):
                                value = getattr(task_data.prediction, metric_name)
                            elif hasattr(task_data, 'eval_result') and hasattr(task_data.eval_result, metric_name):
                                value = getattr(task_data.eval_result, metric_name)

                            if value is not None:
                                data.append({
                                    "run_id": run.id,
                                    "timestamp": run.created_at.isoformat(),
                                    "value": float(value)
                                })
                                break
                        if data and data[-1]["run_id"] == run.id:
                            break
                    except Exception:
                        continue
            except Exception:
                continue

    except Exception as e:
        # Return empty chart with error message
        pass

    # Build chart even if data is empty
    if not data:
        data = [{"run_id": "no_data", "timestamp": "2024-01-01", "value": 0}]

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": f"{metric_name.replace('_', ' ').title()} Over Time",
        "width": 500,
        "height": 200,
        "data": {"values": data},
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {
                "field": "timestamp",
                "type": "temporal",
                "title": "Run Time"
            },
            "y": {
                "field": "value",
                "type": "quantitative",
                "title": metric_name.replace("_", " ").title()
            },
            "tooltip": [
                {"field": "run_id", "title": "Run ID"},
                {"field": "value", "title": "Value", "format": ".3f"}
            ]
        }
    }

    return VegaChart(spec)


def price_change_heatmap(
    coin_info: List[Dict],
    limit: int = 20,
) -> "VegaChart":
    """
    Create a heatmap of price changes across timeframes for top coins.

    This visualization answers: "Which coins are moving most across timeframes?"

    Args:
        coin_info: List of coin info dicts from FeatureSet.coin_info
        limit: Number of top coins to show

    Returns:
        VegaChart component with heatmap
    """
    from metaflow.cards import VegaChart

    # Build data for heatmap
    data = []
    for coin in coin_info[:limit]:
        symbol = coin.get("symbol", "?")
        for tf, key in [("1h", "price_change_1h"), ("24h", "price_change_24h"), ("7d", "price_change_7d")]:
            value = coin.get(key, 0)
            data.append({
                "coin": symbol,
                "timeframe": tf,
                "change": float(value)
            })

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Price Changes Heatmap",
        "width": 200,
        "height": 400,
        "data": {"values": data},
        "mark": "rect",
        "encoding": {
            "x": {
                "field": "timeframe",
                "type": "nominal",
                "title": "Timeframe",
                "sort": ["1h", "24h", "7d"]
            },
            "y": {
                "field": "coin",
                "type": "nominal",
                "title": "Coin"
            },
            "color": {
                "field": "change",
                "type": "quantitative",
                "scale": {"scheme": "redyellowgreen", "domain": [-20, 0, 20]},
                "title": "% Change"
            },
            "tooltip": [
                {"field": "coin", "title": "Coin"},
                {"field": "timeframe", "title": "Timeframe"},
                {"field": "change", "title": "Change %", "format": ".1f"}
            ]
        }
    }

    return VegaChart(spec)


def training_summary_card(
    model_algorithm: str,
    hyperparameters: Dict[str, Any],
    n_samples: int,
    n_features: int,
    anomaly_rate: float,
    data_source: str,
) -> "Markdown":
    """
    Create a summary markdown for training results.

    Args:
        model_algorithm: Algorithm name (e.g., 'isolation_forest')
        hyperparameters: Model hyperparameters dict
        n_samples: Number of training samples
        n_features: Number of features
        anomaly_rate: Detected anomaly rate
        data_source: Description of data source

    Returns:
        Markdown component with training summary
    """
    from metaflow.cards import Markdown

    hp_str = ", ".join(f"{k}={v}" for k, v in hyperparameters.items())

    lines = [
        "## Training Summary\n",
        f"**Algorithm:** {model_algorithm}",
        f"**Hyperparameters:** {hp_str}",
        f"**Data Source:** {data_source}",
        "",
        "### Dataset",
        f"- **Samples:** {n_samples:,}",
        f"- **Features:** {n_features}",
        "",
        "### Results",
        f"- **Anomaly Rate:** {anomaly_rate:.1%}",
        f"- **Anomalies Detected:** {int(n_samples * anomaly_rate):,}",
    ]

    return Markdown("\n".join(lines))
