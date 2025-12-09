"""
Temporal feature engineering for crash prediction.

This module transforms raw snapshots into ML-ready datasets with:
- Lag features (how did this coin behave N hours ago?)
- Rolling statistics (mean/std over past N hours)
- Future target labels (did this coin crash within 24h?)
- Gap-aware processing (handles missing snapshots gracefully)
- Feature scaling (log transform for skewed distributions)

The goal is to predict price crashes BEFORE they happen, not just
detect "anomalies" in a static snapshot.

Feature Engineering Philosophy:
------------------------------
Raw snapshots capture point-in-time market state. But for prediction:
1. We need TEMPORAL context (what happened before?)
2. We need a TARGET to predict (what happens after?)
3. We need to preserve ALL observations (not deduplicate)

Data Quality Handling:
---------------------
1. Late-appearing coins: Require minimum history before including
2. Missing snapshots (gaps): Flag and optionally impute
3. Skewed features: Log-transform high-variance columns
4. Missing lag values: Forward-fill within reasonable windows

Schema after feature engineering:
- Base features: price_change_pct_1h, price_change_pct_24h, etc.
- Lag features: price_change_pct_24h_lag_1h, _lag_6h, _lag_24h
- Rolling features: price_change_pct_24h_rolling_mean_24h, _rolling_std_24h
- Scaled features: market_cap_to_volume_log (log-transformed)
- Quality flags: has_gap_in_history (boolean)
- Target: future_drop_15pct_24h (1 if coin dropped >15% in next 24h)
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import pyarrow as pa
import numpy as np


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

    # Data quality settings
    max_gap_hours: float = 1.5  # Gaps larger than this are flagged
    impute_lag_features: bool = True  # Forward-fill missing lag values
    max_impute_hours: int = 3  # Max hours to forward-fill

    # Feature scaling
    log_transform_features: List[str] = None  # Features to log-transform
    clip_outliers: bool = True  # Clip extreme values
    outlier_std: float = 5.0  # Clip beyond N standard deviations

    def __post_init__(self):
        if self.lag_hours is None:
            self.lag_hours = [1, 6, 24]
        if self.rolling_windows_hours is None:
            self.rolling_windows_hours = [6, 24]
        if self.log_transform_features is None:
            # These features typically have extreme outliers
            self.log_transform_features = ['market_cap_to_volume']


def detect_gaps(
    df,
    config: DatasetConfig,
) -> "pd.DataFrame":
    """
    Detect gaps in snapshot history per coin.

    Returns DataFrame with 'has_gap_in_history' flag and 'hours_since_last' column.
    """
    import pandas as pd

    df = df.sort_values(['coin_id', 'snapshot_timestamp'])

    # Calculate time since previous snapshot for each coin
    df['hours_since_last'] = df.groupby('coin_id')['snapshot_timestamp'].diff().dt.total_seconds() / 3600

    # Flag if there's a large gap in recent history (within max lag window)
    max_lag = max(config.lag_hours) if config.lag_hours else 24

    def has_recent_gap(group):
        """Check if any gap > threshold in the last max_lag observations."""
        recent = group.tail(max_lag)
        return (recent['hours_since_last'] > config.max_gap_hours).any()

    gap_flags = df.groupby('coin_id').apply(has_recent_gap, include_groups=False).reset_index()
    gap_flags.columns = ['coin_id', 'has_gap_in_history']

    df = df.merge(gap_flags, on='coin_id', how='left')

    return df


def apply_log_transform(
    df,
    config: DatasetConfig,
) -> "pd.DataFrame":
    """
    Apply log transform to highly skewed features.

    Creates new columns with '_log' suffix. Uses log1p for stability with zeros.
    """
    for col in config.log_transform_features:
        if col in df.columns:
            # Use log1p(abs(x)) * sign(x) to handle negatives and zeros
            df[f'{col}_log'] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

    return df


def clip_outliers(
    df,
    config: DatasetConfig,
    numeric_cols: List[str],
) -> "pd.DataFrame":
    """
    Clip extreme outliers to reduce impact on model training.

    Clips values beyond N standard deviations from mean.
    """
    if not config.clip_outliers:
        return df

    for col in numeric_cols:
        if col in df.columns and df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                lower = mean - config.outlier_std * std
                upper = mean + config.outlier_std * std
                clipped = df[col].clip(lower, upper)
                # Only report if we actually clipped something
                n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
                if n_clipped > 0:
                    df[col] = clipped

    return df


def impute_lag_features(
    df,
    config: DatasetConfig,
) -> "pd.DataFrame":
    """
    Impute missing lag features using forward-fill within limits.

    Only imputes if the gap is within max_impute_hours.
    """
    import pandas as pd

    if not config.impute_lag_features:
        return df

    lag_cols = [c for c in df.columns if '_lag_' in c]

    for col in lag_cols:
        # Forward-fill within each coin, limited by max_impute_hours
        # Since we have hourly data, max_impute_hours = max rows to fill
        df[col] = df.groupby('coin_id')[col].transform(
            lambda x: x.ffill(limit=config.max_impute_hours)
        )

    return df


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

    # Detect gaps in history
    df = detect_gaps(df, config)

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

    # Impute missing lag features (forward-fill within limits)
    df = impute_lag_features(df, config)

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

    # Apply log transform to skewed features
    df = apply_log_transform(df, config)

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
) -> Tuple[pa.Table, dict]:
    """
    Filter to rows that have sufficient history for lag features.

    Removes early observations for each coin that don't have enough
    prior snapshots to compute lag features. Also removes coins that
    never accumulate enough history.

    Args:
        table: PyArrow table with temporal features
        config: DatasetConfig with min_snapshots_per_coin

    Returns:
        Tuple of (filtered table, stats dict with filtering info)
    """
    import pandas as pd

    if config is None:
        config = DatasetConfig()

    df = table.to_pandas()
    df['snapshot_timestamp'] = pd.to_datetime(df['snapshot_timestamp'])

    initial_rows = len(df)
    initial_coins = df['coin_id'].nunique()

    # Count total observations per coin
    coin_obs_count = df.groupby('coin_id').size()

    # Identify coins with insufficient total history
    insufficient_coins = coin_obs_count[coin_obs_count < config.min_snapshots_per_coin].index.tolist()

    # Remove coins with insufficient history entirely
    df = df[~df['coin_id'].isin(insufficient_coins)]

    # For remaining coins, rank observations by time
    df['obs_rank'] = df.groupby('coin_id')['snapshot_timestamp'].rank(method='first')

    # Keep only rows with sufficient history (removes early observations)
    df = df[df['obs_rank'] >= config.min_snapshots_per_coin]

    # Drop helper column
    df = df.drop(columns=['obs_rank'])

    # Compute stats
    stats = {
        'initial_rows': initial_rows,
        'initial_coins': initial_coins,
        'final_rows': len(df),
        'final_coins': df['coin_id'].nunique(),
        'removed_coins': insufficient_coins,
        'removed_coin_count': len(insufficient_coins),
        'early_obs_removed': initial_rows - len(df) - sum(coin_obs_count[c] for c in insufficient_coins if c in coin_obs_count.index),
    }

    return pa.Table.from_pandas(df, preserve_index=False), stats


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
    verbose: bool = True,
) -> Tuple[pa.Table, List[str]]:
    """
    Full pipeline: raw snapshots → ML-ready dataset.

    Pipeline steps:
    1. Apply rolling window to bound dataset size
    2. Add temporal features (lags, rolling stats, gap detection)
    3. Apply log transforms to skewed features
    4. Impute missing lag features (forward-fill)
    5. Add targets if requested (for evaluation only!)
    6. Filter to rows with sufficient history
    7. Clip outliers

    Args:
        raw_table: Concatenated raw snapshots
        config: DatasetConfig
        include_targets: Whether to add future targets (for evaluation only!)
        verbose: Print processing stats

    Returns:
        Tuple of (processed table, list of feature column names)
    """
    if config is None:
        config = DatasetConfig()

    if verbose:
        print(f"Building ML dataset with config:")
        print(f"  min_snapshots_per_coin: {config.min_snapshots_per_coin}")
        print(f"  lag_hours: {config.lag_hours}")
        print(f"  impute_lag_features: {config.impute_lag_features}")
        print(f"  log_transform_features: {config.log_transform_features}")

    # Step 1: Apply rolling window to bound dataset size
    table = apply_rolling_window(raw_table, config.max_history_hours)
    if verbose:
        print(f"\nAfter rolling window: {table.num_rows} rows")

    # Step 2: Add temporal features (includes gap detection, lags, rolling stats, log transforms)
    table = add_temporal_features(table, config)
    if verbose:
        print(f"After temporal features: {table.num_rows} rows, {len(table.column_names)} columns")

    # Step 3: Add targets if requested (for evaluation/backtesting)
    if include_targets:
        table = add_future_targets(table, config)

    # Step 4: Filter to rows with sufficient history
    table, filter_stats = filter_by_history(table, config)
    if verbose:
        print(f"\nFiltering stats:")
        print(f"  Initial: {filter_stats['initial_rows']} rows, {filter_stats['initial_coins']} coins")
        print(f"  Final: {filter_stats['final_rows']} rows, {filter_stats['final_coins']} coins")
        if filter_stats['removed_coin_count'] > 0:
            print(f"  Removed {filter_stats['removed_coin_count']} coins with insufficient history: {filter_stats['removed_coins'][:5]}{'...' if filter_stats['removed_coin_count'] > 5 else ''}")

    # Step 5: Identify feature columns
    df = table.to_pandas()
    feature_cols = [
        col for col in df.columns
        if col not in [
            'coin_id', 'symbol', 'name', 'current_price', 'market_cap',
            'snapshot_timestamp', 'future_price', 'future_price_change_pct',
            'future_crash', 'future_pump', 'hours_since_last'
        ]
        and not col.startswith('future_')
    ]

    # Step 6: Clip outliers on numeric feature columns
    numeric_feature_cols = [c for c in feature_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    df = clip_outliers(df, config, numeric_feature_cols)

    if verbose:
        # Report on data quality
        lag_cols = [c for c in feature_cols if '_lag_' in c]
        if lag_cols:
            missing_pct = df[lag_cols].isna().mean().mean() * 100
            print(f"\nData quality:")
            print(f"  Lag feature missing rate: {missing_pct:.1f}%")
        if 'has_gap_in_history' in df.columns:
            gap_pct = df['has_gap_in_history'].mean() * 100
            print(f"  Rows with gaps in history: {gap_pct:.1f}%")
        if 'market_cap_to_volume_log' in df.columns:
            print(f"  Log-transformed features: {[c for c in df.columns if '_log' in c]}")

    table = pa.Table.from_pandas(df, preserve_index=False)

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
    lag_base = [
        'price_change_pct_1h',
        'price_change_pct_24h',
        'price_change_pct_7d',
        'market_cap_to_volume',
        'sparkline_volatility',
    ]
    for lag in config.lag_hours:
        for col in lag_base:
            feature_cols.append(f'{col}_lag_{lag}h')

    # Rolling features
    for window in config.rolling_windows_hours:
        for col in ['price_change_pct_24h', 'sparkline_volatility']:
            feature_cols.append(f'{col}_rolling_mean_{window}h')
            feature_cols.append(f'{col}_rolling_std_{window}h')

    # Log-transformed features
    for col in config.log_transform_features:
        feature_cols.append(f'{col}_log')

    # Time features
    feature_cols.extend(['hour_of_day', 'day_of_week'])

    # Data quality flags
    feature_cols.append('has_gap_in_history')

    return feature_cols
