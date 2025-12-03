"""
Data fetching and feature engineering for crypto anomaly detection.

This module handles:
- Fetching live market data from CoinGecko API
- Extracting features for anomaly detection
- Data validation and cleaning
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
import statistics


@dataclass
class MarketSnapshot:
    """Container for fetched market data with metadata."""
    coins: List[Dict]
    timestamp: str
    source: str = "coingecko"

    @property
    def count(self) -> int:
        return len(self.coins)


@dataclass
class FeatureSet:
    """Container for extracted features with coin metadata."""
    features: List[List[float]]
    coin_info: List[Dict]
    feature_names: List[str]
    timestamp: str

    @property
    def n_samples(self) -> int:
        return len(self.features)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


# Feature columns for anomaly detection
FEATURE_COLS = [
    "price_change_pct_1h",
    "price_change_pct_24h",
    "price_change_pct_7d",
    "market_cap_to_volume",
    "ath_change_pct",
    "sparkline_volatility",
]


def fetch_market_data(num_coins: int = 100, timeout: int = 30) -> MarketSnapshot:
    """
    Fetch live market data from CoinGecko API.

    Args:
        num_coins: Number of top coins by market cap to fetch (max 250)
        timeout: Request timeout in seconds

    Returns:
        MarketSnapshot with raw coin data

    Raises:
        requests.HTTPError: If API request fails
    """
    import requests

    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": min(num_coins, 250),
        "page": 1,
        "sparkline": "true",
        "price_change_percentage": "1h,24h,7d,30d"
    }

    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()

    return MarketSnapshot(
        coins=response.json(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def extract_features(snapshot: MarketSnapshot) -> FeatureSet:
    """
    Extract features from market snapshot for anomaly detection.

    Args:
        snapshot: MarketSnapshot from fetch_market_data()

    Returns:
        FeatureSet with extracted features and coin metadata
    """
    features = []
    coin_info = []

    for coin in snapshot.coins:
        feature_values, info = _extract_coin_features(coin)
        features.append(feature_values)
        coin_info.append(info)

    return FeatureSet(
        features=features,
        coin_info=coin_info,
        feature_names=FEATURE_COLS,
        timestamp=snapshot.timestamp,
    )


def _extract_coin_features(coin: Dict) -> Tuple[List[float], Dict]:
    """
    Extract features from a single coin's data.

    Returns:
        Tuple of (feature_values, coin_info_dict)
    """
    # Calculate sparkline volatility (coefficient of variation)
    sparkline_vol = 0.0
    if coin.get("sparkline_in_7d"):
        prices = coin["sparkline_in_7d"].get("price", [])
        if prices and len(prices) > 1:
            mean_price = statistics.mean(prices)
            if mean_price > 0:
                sparkline_vol = statistics.stdev(prices) / mean_price

    # Calculate market cap to volume ratio
    mc_to_vol = 0.0
    total_vol = coin.get("total_volume") or 0
    if total_vol > 0:
        mc_to_vol = (coin.get("market_cap") or 0) / total_vol

    # Extract features (handle nulls with 0)
    feature_values = [
        coin.get("price_change_percentage_1h_in_currency") or 0,
        coin.get("price_change_percentage_24h") or 0,
        coin.get("price_change_percentage_7d_in_currency") or 0,
        mc_to_vol,
        coin.get("ath_change_percentage") or 0,
        sparkline_vol,
    ]

    # Coin metadata for reporting
    info = {
        "coin_id": coin["id"],
        "symbol": coin["symbol"].upper(),
        "name": coin["name"],
        "current_price": coin.get("current_price", 0),
        "market_cap": coin.get("market_cap", 0),
        "market_cap_rank": coin.get("market_cap_rank"),
        "total_volume": coin.get("total_volume", 0),
        "price_change_1h": coin.get("price_change_percentage_1h_in_currency") or 0,
        "price_change_24h": coin.get("price_change_percentage_24h") or 0,
        "price_change_7d": coin.get("price_change_percentage_7d_in_currency") or 0,
        "volatility_7d": sparkline_vol,
    }

    return feature_values, info


def get_feature_stats(feature_set: FeatureSet) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics for each feature.

    Args:
        feature_set: FeatureSet from extract_features()

    Returns:
        Dict mapping feature name to stats (mean, stdev, min, max)
    """
    stats = {}
    for i, name in enumerate(feature_set.feature_names):
        values = [row[i] for row in feature_set.features]
        stats[name] = {
            "mean": statistics.mean(values) if values else 0.0,
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }
    return stats
