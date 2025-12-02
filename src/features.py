"""
Temporal feature engineering for crash prediction.

This module transforms raw snapshots into ML-ready datasets with:
- Lag features (how did this coin behave N hours ago?)
- Rolling statistics (mean/std over past N hours)
- Future target labels (did this coin crash within 24h?)

The goal is to predict price crashes BEFORE they happen, not just
detect "anomalies" in a static snapshot.

Feature Engineering Philosophy:
------------------------------
Raw snapshots capture point-in-time market state. But for prediction:
1. We need TEMPORAL context (what happened before?)
2. We need a TARGET to predict (what happens after?)
3. We need to preserve ALL observations (not deduplicate)

Schema after feature engineering:
- Base features: price_change_pct_1h, price_change_pct_24h, etc.
- Lag features: price_change_pct_24h_lag_1h, _lag_6h, _lag_24h
- Rolling features: price_change_pct_24h_rolling_mean_24h, _rolling_std_24h
- Target: future_drop_15pct_24h (1 if coin dropped >15% in next 24h)
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import pyarrow as pa


@dataclass
class DatasetConfig:
    """Configuration for dataset building."""
    # Window settings
    max_history_hours: int = 168  # 7 days of history (rolling window)
    min_snapshots_per_coin: int = 6  # Need at least 6 hours of history

    # Lag features (hours ago)
    lag_hours: List[int] = None  # Default: [1, 6, 24]

    # Rolling window features
    rolling_windows_hours: List[int] = None  # Default: [6, 24]

    # Target definition
    target_lookahead_hours: int = 24
    crash_threshold_pct: float = -15.0  # -15% = crash

    def __post_init__(self):
        if self.lag_hours is None:
            self.lag_hours = [1, 6, 24]
        if self.rolling_windows_hours is None:
            self.rolling_windows_hours = [6, 24]


def add_temporal_features(
    table: pa.Table,
    config: Optional[DatasetConfig] = None,
) -> pa.Table:
    """
    Add lag features and rolling statistics to a snapshot table.

    This transforms raw snapshots into ML-ready features by adding
    temporal context for each observation.

    Args:
        table: PyArrow table with columns: coin_id, snapshot_timestamp, features...
        config: DatasetConfig with lag/rolling settings

    Returns:
        PyArrow table with additional lag and rolling columns
    """
    import pandas as pd

    if config is None:
        config = DatasetConfig()

    df = table.to_pandas()

    # Ensure timestamp is datetime
    df['snapshot_timestamp'] = pd.to_datetime(df['snapshot_timestamp'])

    # Sort by coin and time for proper lag calculation
    df = df.sort_values(['coin_id', 'snapshot_timestamp'])

    # Base feature columns to create lags for
    base_features = [
        'price_change_pct_1h',
        'price_change_pct_24h',
        'price_change_pct_7d',
        'market_cap_to_volume',
        'sparkline_volatility',
    ]

    # Add lag features per coin
    for lag_hours in config.lag_hours:
        for col in base_features:
            if col in df.columns:
                df[f'{col}_lag_{lag_hours}h'] = df.groupby('coin_id')[col].shift(lag_hours)

    # Add rolling statistics per coin
    for window_hours in config.rolling_windows_hours:
        for col in ['price_change_pct_24h', 'sparkline_volatility']:
            if col in df.columns:
                grouped = df.groupby('coin_id')[col]
                df[f'{col}_rolling_mean_{window_hours}h'] = grouped.transform(
                    lambda x: x.rolling(window=window_hours, min_periods=1).mean()
                )
                df[f'{col}_rolling_std_{window_hours}h'] = grouped.transform(
                    lambda x: x.rolling(window=window_hours, min_periods=2).std()
                )

    # Add time-based features
    df['hour_of_day'] = df['snapshot_timestamp'].dt.hour
    df['day_of_week'] = df['snapshot_timestamp'].dt.dayofweek

    # Convert back to PyArrow
    return pa.Table.from_pandas(df, preserve_index=False)


def add_future_targets(
    table: pa.Table,
    config: Optional[DatasetConfig] = None,
) -> pa.Table:
    """
    Add future price movement targets for supervised learning.

    IMPORTANT: This creates data leakage if used naively!
    Only use for EVALUATION, not for training production models.
    For training, you must use properly time-split data.

    Args:
        table: PyArrow table with coin_id, snapshot_timestamp, current_price
        config: DatasetConfig with target settings

    Returns:
        Table with added target columns
    """
    import pandas as pd

    if config is None:
        config = DatasetConfig()

    df = table.to_pandas()
    df['snapshot_timestamp'] = pd.to_datetime(df['snapshot_timestamp'])
    df = df.sort_values(['coin_id', 'snapshot_timestamp'])

    # Calculate future price (lookahead_hours later)
    df['future_price'] = df.groupby('coin_id')['current_price'].shift(-config.target_lookahead_hours)

    # Calculate future price change
    df['future_price_change_pct'] = (
        (df['future_price'] - df['current_price']) / df['current_price'] * 100
    )

    # Binary crash target
    df['future_crash'] = (df['future_price_change_pct'] <= config.crash_threshold_pct).astype(int)

    # Binary pump target (for completeness)
    df['future_pump'] = (df['future_price_change_pct'] >= abs(config.crash_threshold_pct)).astype(int)

    return pa.Table.from_pandas(df, preserve_index=False)


def filter_by_history(
    table: pa.Table,
    config: Optional[DatasetConfig] = None,
) -> pa.Table:
    """
    Filter to rows that have sufficient history for lag features.

    Removes early observations for each coin that don't have enough
    prior snapshots to compute lag features.

    Args:
        table: PyArrow table with temporal features
        config: DatasetConfig with min_snapshots_per_coin

    Returns:
        Filtered table
    """
    import pandas as pd

    if config is None:
        config = DatasetConfig()

    df = table.to_pandas()
    df['snapshot_timestamp'] = pd.to_datetime(df['snapshot_timestamp'])

    # Count observations per coin and rank by time
    df['obs_rank'] = df.groupby('coin_id')['snapshot_timestamp'].rank(method='first')

    # Keep only rows with sufficient history
    df = df[df['obs_rank'] >= config.min_snapshots_per_coin]

    # Drop helper column
    df = df.drop(columns=['obs_rank'])

    return pa.Table.from_pandas(df, preserve_index=False)


def apply_rolling_window(
    table: pa.Table,
    max_hours: int = 168,
) -> pa.Table:
    """
    Apply rolling window to limit dataset to recent history.

    This prevents unbounded dataset growth while keeping temporal diversity.

    Args:
        table: PyArrow table with snapshot_timestamp
        max_hours: Maximum hours of history to keep

    Returns:
        Table filtered to recent window
    """
    import pandas as pd
    from datetime import timedelta

    df = table.to_pandas()
    df['snapshot_timestamp'] = pd.to_datetime(df['snapshot_timestamp'])

    # Find the most recent timestamp
    max_ts = df['snapshot_timestamp'].max()
    cutoff = max_ts - timedelta(hours=max_hours)

    # Filter to recent window
    df = df[df['snapshot_timestamp'] >= cutoff]

    return pa.Table.from_pandas(df, preserve_index=False)


def build_ml_dataset(
    raw_table: pa.Table,
    config: Optional[DatasetConfig] = None,
    include_targets: bool = False,
) -> Tuple[pa.Table, List[str]]:
    """
    Full pipeline: raw snapshots → ML-ready dataset.

    Args:
        raw_table: Concatenated raw snapshots
        config: DatasetConfig
        include_targets: Whether to add future targets (for evaluation only!)

    Returns:
        Tuple of (processed table, list of feature column names)
    """
    if config is None:
        config = DatasetConfig()

    # Step 1: Apply rolling window to bound dataset size
    table = apply_rolling_window(raw_table, config.max_history_hours)

    # Step 2: Add temporal features
    table = add_temporal_features(table, config)

    # Step 3: Add targets if requested (for evaluation/backtesting)
    if include_targets:
        table = add_future_targets(table, config)

    # Step 4: Filter to rows with sufficient history
    table = filter_by_history(table, config)

    # Identify feature columns
    df = table.to_pandas()
    feature_cols = [
        col for col in df.columns
        if col not in [
            'coin_id', 'symbol', 'name', 'current_price', 'market_cap',
            'snapshot_timestamp', 'future_price', 'future_price_change_pct',
            'future_crash', 'future_pump'
        ]
        and not col.startswith('future_')
    ]

    return table, feature_cols


def get_feature_columns(config: Optional[DatasetConfig] = None) -> List[str]:
    """
    Get list of all feature columns that will be created.

    Useful for model training to know expected input shape.
    """
    if config is None:
        config = DatasetConfig()

    base_features = [
        'price_change_pct_1h',
        'price_change_pct_24h',
        'price_change_pct_7d',
        'market_cap_to_volume',
        'ath_change_pct',
        'sparkline_volatility',
    ]

    feature_cols = list(base_features)

    # Lag features
    for lag in config.lag_hours:
        for col in base_features:
            feature_cols.append(f'{col}_lag_{lag}h')

    # Rolling features
    for window in config.rolling_windows_hours:
        for col in ['price_change_pct_24h', 'sparkline_volatility']:
            feature_cols.append(f'{col}_rolling_mean_{window}h')
            feature_cols.append(f'{col}_rolling_std_{window}h')

    # Time features
    feature_cols.extend(['hour_of_day', 'day_of_week'])

    return feature_cols
