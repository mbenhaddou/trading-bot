"""
Unified Data Provider System
Handles both simulation (Yahoo Finance with caching) and live trading (Alpaca IEX)
Respects Yahoo Finance time limitations with intelligent chunked downloading
"""

import os
import pickle
import json
import hashlib
import re
import time
import numpy as np
import pandas as pd
import pytz
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque, defaultdict

import logging

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:
    yf = None
    logger.warning("yfinance not available - simulation mode will be limited")

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import APIError
except ImportError:
    tradeapi = None
    APIError = Exception
    logger.warning("alpaca_trade_api not available - live trading will be limited")

# ============================================================================
# YAHOO FINANCE LIMITATIONS
# ============================================================================


YFINANCE_LIMITS = {
    '1m': {'max_days': 7, 'display': '1 minute'},
    '2m': {'max_days': 60, 'display': '2 minute'},
    '5m': {'max_days': 60, 'display': '5 minute'},
    '15m': {'max_days': 60, 'display': '15 minute'},
    '30m': {'max_days': 60, 'display': '30 minute'},
    '60m': {'max_days': 730, 'display': 'hourly'},
    '1h': {'max_days': 730, 'display': 'hourly'},
    '1d': {'max_days': None, 'display': 'daily'},
    '1wk': {'max_days': None, 'display': 'weekly'},
    '1mo': {'max_days': None, 'display': 'monthly'}
}

INTERVAL_MAPPINGS = {
    'm':'m',
    'min': 'm', 'minute': 'm', 'minutes': 'm',
    'h': 'h', 'hour': 'h', 'hours': 'h',
    'd': 'd', 'day': 'd', 'days': 'd',
    'w': 'w', 'wk': 'w', 'week': 'w', 'weeks': 'w',
    'mo': 'mo', 'month': 'mo', 'months': 'mo'
}


def normalize_interval(interval: str) -> str:
    """
    Normalize interval string to standard format (e.g., '1m', '5m', '1h', '1d')

    Handles various formats:
    - Yahoo Finance: '1m', '5m', '1h', '1d'
    - Alpaca: '1Min', '5Min', '1Hour', '1Day'
    - Other: '1minute', '5minutes', '1hour', '1day', etc.
    """
    if not interval or not isinstance(interval, str):
        raise ValueError(f"Invalid interval: {interval}")

    interval_lower = interval.strip().lower()
    match = re.match(r'^(\d+)\s*([a-z]+)$', interval_lower)

    if not match:
        raise ValueError(
            f"Invalid interval format: '{interval}'. "
            f"Expected format like '1m', '5min', '1Hour', '1Day', etc."
        )

    number, unit = match.groups()

    if unit not in INTERVAL_MAPPINGS:
        raise ValueError(
            f"Unknown interval unit: '{unit}'. "
            f"Supported units: {', '.join(INTERVAL_MAPPINGS.keys())}"
        )

    normalized_unit = INTERVAL_MAPPINGS[unit]
    return f"{number}{normalized_unit}"


from dataclasses import dataclass

@dataclass
class EpisodeConfig:
    """Define clear temporal boundaries with interval awareness"""

    # Temporal boundaries
    train_start: str = "2020-01-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2023-06-30"
    test_start: str = "2023-07-01"
    test_end: str = "2023-12-31"

    # Interval-specific parameters
    interval: str = "1Min"
    lookback_bars: int = 100
    episode_length_bars: int = 390

    def __post_init__(self):
        """Validate configuration"""
        train_end = pd.Timestamp(self.train_end)
        val_start = pd.Timestamp(self.val_start)
        val_end = pd.Timestamp(self.val_end)
        test_start = pd.Timestamp(self.test_start)

        assert train_end < val_start, f"Train/Val overlap!"
        assert val_end < test_start, f"Val/Test overlap!"

        logger.info(f"Episode config validated for interval: {self.interval}")
        logger.info(f"  Lookback: {self.lookback_bars} bars")
        logger.info(f"  Episode length: {self.episode_length_bars} bars")

    def get_boundaries(self, mode: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get start/end for given mode"""
        mode_map = {
            'train': (self.train_start, self.train_end),
            'training': (self.train_start, self.train_end),
            'validation': (self.val_start, self.val_end),
            'val': (self.val_start, self.val_end),
            'test': (self.test_start, self.test_end),
        }
        start, end = mode_map[mode.lower()]
        return pd.Timestamp(start), pd.Timestamp(end)

    def get_estimated_episodes_available(self, mode: str = 'train') -> int:
        """
        Calculate how many different episode starting positions are available

        This tells you how much variety you have in your training data
        """
        start, end = self.get_boundaries(mode)
        days = (end - start).days
        trading_days = days * (5 / 7) * 0.95  # Account for weekends/holidays

        bars_per_day = self._calculate_bars_per_day(self.interval)
        total_bars = int(trading_days * bars_per_day)

        # Available starting positions
        available_positions = max(1, total_bars - self.episode_length_bars)

        return available_positions

    @staticmethod
    def _calculate_bars_per_day(interval: str) -> float:
        """Calculate bars per trading day for given interval"""
        interval_lower = interval.lower()

        if 'min' in interval_lower or interval_lower.endswith('m'):
            if interval_lower.endswith('min'):
                minutes = int(interval_lower.replace('min', ''))
            else:
                minutes = int(interval_lower.replace('m', ''))
            return 390.0 / minutes

        elif 'hour' in interval_lower or interval_lower.endswith('h'):
            if 'hour' in interval_lower:
                hours = int(interval_lower.replace('hour', '').replace('s', ''))
            else:
                hours = int(interval_lower.replace('h', ''))
            return 6.5 / hours

        elif 'day' in interval_lower or interval_lower.endswith('d'):
            return 1.0

        elif 'week' in interval_lower or interval_lower.endswith('w'):
            return 0.2

        elif 'month' in interval_lower or interval_lower.endswith('mo'):
            return 0.05

        else:
            logger.warning(f"Unknown interval {interval}, assuming 1 bar/day")
            return 1.0

    @staticmethod
    def _calculate_episode_length_from_data(train_start: str, train_end: str,
                                            interval: str) -> int:
        """
        Calculate appropriate episode length based on training period and interval

        The goal is to make episodes:
        1. Long enough to capture meaningful patterns
        2. Short enough to have multiple different episodes (variety)
        3. Appropriate scale for the interval

        Args:
            train_start: Training start date
            train_end: Training end date
            interval: Data interval

        Returns:
            Episode length in bars
        """
        # Calculate training period duration
        start = pd.Timestamp(train_start)
        end = pd.Timestamp(train_end)
        days = (end - start).days
        trading_days = days * (5 / 7) * 0.95  # Account for weekends/holidays

        # Calculate total bars in training period
        bars_per_day = EpisodeConfig._calculate_bars_per_day(interval)
        total_bars = int(trading_days * bars_per_day)

        interval_lower = interval.lower()

        # Calculate episode length as a percentage of total data
        # We want 10-20 different episode starting positions for variety
        # So episode_length should be 80-90% of total, or use predefined ranges

        if '1min' in interval_lower or interval_lower == '1m':
            # 1-minute: 1 trading day (390 bars)
            episode_length = 390

        elif '5min' in interval_lower or interval_lower == '5m':
            # 5-minute: 1 trading day (78 bars) to 1 week (390 bars)
            episode_length = min(390, max(78, int(total_bars * 0.05)))

        elif '15min' in interval_lower or interval_lower == '15m':
            # 15-minute: 1 trading day (26 bars) to 2 weeks (260 bars)
            episode_length = min(260, max(26, int(total_bars * 0.05)))

        elif '30min' in interval_lower or interval_lower == '30m':
            # 30-minute: 1 trading day (13 bars) to 2 weeks (130 bars)
            episode_length = min(130, max(13, int(total_bars * 0.05)))

        elif '1hour' in interval_lower or interval_lower == '1h':
            # 1-hour: 1 week (30-35 bars) to 1 month (130 bars)
            episode_length = min(130, max(30, int(total_bars * 0.08)))

        elif '1day' in interval_lower or interval_lower == '1d':
            # Daily: Use 5-10% of training data
            # With 3 years (756 bars): 38-76 bars (2-4 months)
            # With 10 years (2520 bars): 126-252 bars (6-12 months)
            min_length = 20  # 1 month minimum
            max_length = 252  # 1 year maximum
            calculated = int(total_bars * 0.10)  # 10% of data
            episode_length = max(min_length, min(calculated, max_length))

        elif 'week' in interval_lower or interval_lower.endswith('w'):
            # Weekly: 3-6 months (12-26 bars)
            min_length = 12
            max_length = 52
            calculated = int(total_bars * 0.15)
            episode_length = max(min_length, min(calculated, max_length))

        else:
            # Default: 10% of total data, min 10, max 100
            episode_length = max(10, min(100, int(total_bars * 0.10)))

        # Calculate how many different episodes we can sample
        available_positions = max(1, total_bars - episode_length)

        logger.info(f"Episode length calculation for {interval}:")
        logger.info(f"  Training period: {train_start} to {train_end}")
        logger.info(f"  Total bars: {total_bars}")
        logger.info(f"  Episode length: {episode_length} bars")
        logger.info(f"  Episode duration: {episode_length / bars_per_day:.1f} trading days")
        logger.info(f"  Available episode positions: {available_positions}")

        return episode_length

    @staticmethod
    def _calculate_lookback(interval: str, episode_length: int) -> int:
        """
        Calculate appropriate lookback based on interval

        Lookback should be enough for technical indicators (typically 20-100 bars)
        """
        interval_lower = interval.lower()

        if '1min' in interval_lower or interval_lower == '1m':
            # 1-minute: 1-2 hours
            lookback = 100

        elif '5min' in interval_lower or interval_lower == '5m':
            # 5-minute: 2-4 hours (24-48 bars)
            lookback = 100

        elif '15min' in interval_lower or interval_lower == '15m':
            # 15-minute: 1-2 days (26-52 bars)
            lookback = 60

        elif '1hour' in interval_lower or interval_lower == '1h':
            # 1-hour: 3-5 days (20-35 bars)
            lookback = 100

        elif '1day' in interval_lower or interval_lower == '1d':
            # Daily: 2-3 months (40-60 bars)
            lookback = 60

        elif 'week' in interval_lower:
            # Weekly: 6-12 months (26-52 bars)
            lookback = 52

        else:
            # Default: 1/3 of episode length, max 100
            lookback = min(100, max(20, episode_length // 3))

        return lookback

    @staticmethod
    def from_interval(interval: str,
                      train_start: str = "2020-01-01",
                      train_end: str = "2022-12-31",
                      val_start: str = "2023-01-01",
                      val_end: str = "2023-06-30",
                      test_start: str = "2023-07-01",
                      test_end: str = "2023-12-31") -> 'EpisodeConfig':
        """
        Create EpisodeConfig with episode length calculated from training data

        Args:
            interval: Data interval (1Min, 1Day, etc.)
            train_start, train_end: Training period
            val_start, val_end: Validation period
            test_start, test_end: Test period

        Returns:
            EpisodeConfig with calculated episode_length_bars and lookback_bars
        """
        # Calculate episode length based on training data and interval
        episode_length_bars = EpisodeConfig._calculate_episode_length_from_data(
            train_start, train_end, interval
        )

        # Calculate appropriate lookback
        lookback_bars = EpisodeConfig._calculate_lookback(interval, episode_length_bars)

        logger.info(f"Created EpisodeConfig for {interval}:")
        logger.info(f"  Training: {train_start} to {train_end}")
        logger.info(f"  Episode length: {episode_length_bars} bars")
        logger.info(f"  Lookback: {lookback_bars} bars")

        config = EpisodeConfig(
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
            interval=interval,
            lookback_bars=lookback_bars,
            episode_length_bars=episode_length_bars
        )

        # Show how many episodes can be sampled
        available = config.get_estimated_episodes_available('train')
        logger.info(f"  Can sample {available} different episodes from training data")

        return config

    def get_estimated_episodes_available(self, mode: str = 'train') -> int:
        """
        Calculate how many UNIQUE episode starting positions are available

        IMPORTANT: This tells you diversity, NOT the limit!
        You can run unlimited episodes by revisiting positions.

        Returns:
            Number of unique starting positions (for diversity tracking)
        """
        start, end = self.get_boundaries(mode)
        days = (end - start).days
        trading_days = days * (5 / 7) * 0.95

        bars_per_day = self._calculate_bars_per_day(self.interval)
        total_bars = int(trading_days * bars_per_day)

        # Available UNIQUE starting positions
        available_positions = max(1, total_bars - self.episode_length_bars)

        return available_positions

    def get_expected_segment_reuse(self, total_episodes: int, mode: str = 'train') -> float:
        """
        Calculate expected reuse rate for given number of episodes

        Args:
            total_episodes: Total episodes you plan to run
            mode: Training mode

        Returns:
            Expected number of times each segment will be visited on average

        Example:
            If you have 250 unique segments and run 10,000 episodes,
            each segment will be visited ~40 times on average.
        """
        unique_positions = self.get_estimated_episodes_available(mode)
        return total_episodes / max(1, unique_positions)


# ============================================================================
# BASE INTERFACE
# ============================================================================

class BaseDataProvider(ABC):
    """Unified interface for all data providers"""

    def __init__(self):
        # Initialize attributes that subclasses can override
        self._current_index = 0
        self._max_cache_index = 0
        self._cache_mode = False

    @abstractmethod
    def initialize_cache_for_episode(self, lookback_bars: int = 1000) -> bool:
        """Initialize data cache for episode (simulation only)"""
        pass

    @abstractmethod
    def get_historical_data(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data"""
        pass

    @abstractmethod
    def get_latest_data(self, symbols: List[str] = None) -> Dict:
        """Get latest market data for symbols"""
        pass

    def get_latest_data_for_symbol(self, symbol: str) -> Dict:
        """Convenience method for single symbol"""
        return self.get_latest_data([symbol]).get(symbol, {})

    def advance_cache(self) -> bool:
        """Advance to next timestep (simulation only, returns True for live)"""
        return True

    def reset_cache_position(self, start_index: int = 100):
        """Reset cache position (simulation only)"""
        pass

    @property
    def cache_mode(self) -> bool:
        """Whether provider is in cache mode"""
        return self._cache_mode

    @property
    def current_index(self) -> int:
        """Current cache index (simulation only)"""
        return self._current_index

    @current_index.setter
    def current_index(self, value: int):
        """Set current cache index"""
        self._current_index = value

    @property
    def max_cache_index(self) -> int:
        """Maximum cache index (simulation only)"""
        return self._max_cache_index

    @max_cache_index.setter
    def max_cache_index(self, value: int):
        """Set maximum cache index"""
        self._max_cache_index = value


# ============================================================================
# CHUNKED DATA DOWNLOADER
# ============================================================================

class ChunkedDataDownloader:
    """
    Handles chunked downloading of data respecting Yahoo Finance limitations
    Automatically splits requests into valid chunks and stitches results
    """

    def __init__(self):
        self.download_stats = {
            'total_chunks': 0,
            'successful_chunks': 0,
            'failed_chunks': 0,
            'total_rows': 0
        }

    def download_with_chunks(self, symbol: str, interval: str,
                             start_date: datetime, end_date: datetime,
                             progress_callback=None) -> Optional[pd.DataFrame]:
        """
        Download data in chunks respecting Yahoo Finance limitations

        Args:
            symbol: Stock symbol
            interval: Data interval (1m, 5m, 1h, 1d, etc.)
            start_date: Start date
            end_date: End date
            progress_callback: Optional callback function(current, total, message)

        Returns:
            Combined DataFrame or None if failed
        """
        # Get limitation
        limit_info = YFINANCE_LIMITS.get(interval, {'max_days': None})
        max_days = limit_info['max_days']

        # Calculate total duration
        total_days = (end_date - start_date).days

        # If no limit or within limit, download directly
        if max_days is None or total_days <= max_days:
            logger.info(f"Downloading {symbol} {interval} from {start_date.date()} to {end_date.date()} (single chunk)")
            return self._download_single_chunk(symbol, interval, start_date, end_date)

        # Need chunked download
        logger.info(f"Downloading {symbol} {interval} from {start_date.date()} to {end_date.date()} in chunks")
        logger.info(f"Limitation: {limit_info['display']} data limited to {max_days} days")

        # Calculate chunks
        chunks = self._calculate_chunks(start_date, end_date, max_days)
        total_chunks = len(chunks)

        logger.info(f"Splitting into {total_chunks} chunks...")

        # Download each chunk
        all_dataframes = []

        for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
            if progress_callback:
                progress_callback(i, total_chunks, f"Downloading chunk {i}/{total_chunks}")

            logger.info(f"Chunk {i}/{total_chunks}: {chunk_start.date()} to {chunk_end.date()}")

            df = self._download_single_chunk(symbol, interval, chunk_start, chunk_end)

            if df is not None and not df.empty:
                all_dataframes.append(df)
                self.download_stats['successful_chunks'] += 1
                self.download_stats['total_rows'] += len(df)
                logger.info(f"  ✓ Downloaded {len(df)} rows")
            else:
                self.download_stats['failed_chunks'] += 1
                logger.warning(f"  ✗ Chunk failed or empty")

            # Rate limiting
            if i < total_chunks:
                time.sleep(0.5)  # Be nice to Yahoo Finance

        self.download_stats['total_chunks'] = total_chunks

        # Combine all chunks
        if not all_dataframes:
            logger.error(f"All chunks failed for {symbol}")
            return None

        logger.info(f"Combining {len(all_dataframes)} chunks...")
        combined_df = pd.concat(all_dataframes)

        # Remove duplicates (overlap between chunks)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        combined_df = combined_df.sort_index()

        logger.info(f"✓ Combined result: {len(combined_df)} rows total")

        return combined_df

    def _calculate_chunks(self, start_date: datetime, end_date: datetime,
                          max_days: int) -> List[Tuple[datetime, datetime]]:
        """Calculate optimal chunk boundaries"""
        chunks = []

        # Use slightly smaller chunks to account for weekends/holidays
        safe_chunk_days = int(max_days * 0.9)  # 90% of limit

        current_start = start_date

        while current_start < end_date:
            current_end = min(
                current_start + timedelta(days=safe_chunk_days),
                end_date
            )
            chunks.append((current_start, current_end))
            current_start = current_end

        return chunks

    def _download_single_chunk(self, symbol: str, interval: str,
                               start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Download a single chunk of data"""
        try:
            df = yf.download(
                tickers=symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval,
                progress=False,
                threads=False,
                auto_adjust=False
            )

            if df is None or df.empty:
                return None

            # Normalize columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = [str(c).lower() for c in df.columns]

            # Ensure required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            for col in required:
                if col not in df.columns:
                    df[col] = 0

            df = df[required].copy()
            df.index = pd.to_datetime(df.index)

            return df

        except Exception as e:
            logger.error(f"Error downloading chunk: {e}")
            return None

    def get_stats(self) -> Dict:
        """Get download statistics"""
        return self.download_stats.copy()


# ============================================================================
# SIMULATION DATA CACHE
# ============================================================================

class SimulationDataCache:
    """Persistent cache for simulation data with memory + disk storage"""

    def __init__(self, cache_dir: str = 'simulation_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.memory_cache = {}
        self.metadata_file = self.cache_dir / 'metadata.json'
        self.metadata = self._load_metadata()

        logger.info(f"Simulation cache initialized at {self.cache_dir}")

    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        return {}

    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save metadata: {e}")

    def _get_cache_key(self, symbol: str, start_date: str, end_date: str, interval: str) -> str:
        """Generate cache key"""
        key_str = f"{symbol}_{start_date}_{end_date}_{interval}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.pkl"

    def has_data(self, symbol: str, start_date: str, end_date: str, interval: str) -> bool:
        """Check if data is cached"""
        cache_key = self._get_cache_key(symbol, start_date, end_date, interval)
        return cache_key in self.memory_cache or self._get_cache_path(cache_key).exists()

    def get_data(self, symbol: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """Retrieve cached data"""
        cache_key = self._get_cache_key(symbol, start_date, end_date, interval)

        # Check memory
        if cache_key in self.memory_cache:
            logger.debug(f"Memory cache hit for {symbol}")
            return self.memory_cache[cache_key].copy()

        # Check disk
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)
                self.memory_cache[cache_key] = df
                logger.debug(f"Disk cache hit for {symbol}")
                return df.copy()
            except Exception as e:
                logger.warning(f"Failed to load cache for {symbol}: {e}")

        return None

    def save_data(self, symbol: str, start_date: str, end_date: str, interval: str, data: pd.DataFrame):
        """Save data to cache"""
        cache_key = self._get_cache_key(symbol, start_date, end_date, interval)

        # Save to memory
        self.memory_cache[cache_key] = data.copy()

        # Save to disk
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)

            # Update metadata
            self.metadata[cache_key] = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'timeframe': interval,
                'cached_at': datetime.now().isoformat(),
                'rows': len(data)
            }
            self._save_metadata()
            logger.debug(f"Cached {len(data)} rows for {symbol}")
        except Exception as e:
            logger.error(f"Failed to cache {symbol}: {e}")

    def clear_cache(self, older_than_days: int = None):
        """Clear cache"""
        if older_than_days is None:
            self.memory_cache.clear()
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()
            self.metadata.clear()
            self._save_metadata()
            logger.info("Cleared all cache")
        else:
            cutoff = datetime.now() - timedelta(days=older_than_days)
            keys_to_remove = []

            for key, meta in self.metadata.items():
                if datetime.fromisoformat(meta['cached_at']) < cutoff:
                    keys_to_remove.append(key)
                    cache_path = self._get_cache_path(key)
                    if cache_path.exists():
                        cache_path.unlink()

            for key in keys_to_remove:
                self.metadata.pop(key, None)
                self.memory_cache.pop(key, None)

            self._save_metadata()
            logger.info(f"Cleared {len(keys_to_remove)} old cache entries")

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        disk_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
        return {
            'memory_entries': len(self.memory_cache),
            'disk_entries': len(self.metadata),
            'disk_size_mb': disk_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }


# ============================================================================
# SIMULATION DATA PROVIDER (YAHOO FINANCE)
# ============================================================================

class SimulationDataProvider(BaseDataProvider):
    """Simulation data provider using Yahoo Finance with chunked downloading and caching"""

    def __init__(self, config: Dict, cache_dir: str = 'simulation_cache', episode_config: EpisodeConfig = None):
        super().__init__()
        self.config = config
        self.episode_config = episode_config or EpisodeConfig()  # NEW
        self.cache = SimulationDataCache(cache_dir)
        self.downloader = ChunkedDataDownloader()

        self.episode_count = 0
        self.segment_usage = defaultdict(int)
        # Cache mode tracking (using parent's properties)
        self.cached_data = {}

        # Statistics
        self.download_count = 0
        self.cache_hit_count = 0

        # Market data cache
        self.last_market_data = {}

        if yf is None:
            raise ImportError("yfinance required for simulation mode: pip install yfinance")

        logger.info("SimulationDataProvider initialized with chunked downloading support")

    def initialize_cache_for_episode(self, mode: str = 'train',
                                     lookback_bars: int = None,
                                     random_start: bool = None,
                                     start_offset: int = None) -> bool:
        """
        Initialize cache with explicit temporal boundaries

        UNLIMITED EPISODES: Can be called unlimited times with different offsets.
        Same segment can be used multiple times with different policies.

        Args:
            mode: 'train', 'validation', or 'test'
            lookback_bars: Override default lookback
            random_start: Random start for variety
            start_offset: EXPLICIT offset (NEW - for controlled sampling)
        """

        # Use config values
        if lookback_bars is None:
            lookback_bars = self.episode_config.lookback_bars


        if random_start is None:
            random_start = (mode == 'train')  # Random for train, fixed for val/test

        episode_length = self.episode_config.episode_length_bars

        # Get interval from config
        interval = self.config.get('simulation', {}).get('timeframe', '1Min')

        logger.info(f"\n{'=' * 70}")
        logger.info(f"INITIALIZING CACHE - {mode.upper()} MODE")
        logger.info(f"{'=' * 70}")

        # Get temporal boundaries for this mode (the actual train/val/test period)
        period_start, period_end = self.episode_config.get_boundaries(mode)

        logger.info(f"Target period: {period_start.date()} to {period_end.date()}")
        logger.info(f"Interval: {interval}")
        logger.info(f"Lookback: {lookback_bars} bars (BEFORE period start)")
        logger.info(f"Episode length: {episode_length} bars")
        logger.info(f"Random start: {random_start}")

        symbols = self.config.get('symbols_to_trade', [])
        self.cached_data = {}
        min_length = float('inf')

        # Calculate bars per day based on interval
        bars_per_day = self._calculate_bars_per_day(interval)
        logger.info(f"Bars per day: {bars_per_day}")

        # Calculate how much data to fetch
        # We need: lookback_bars BEFORE period_start + episode_length within period
        lookback_days = int(lookback_bars / bars_per_day * 1.5) + 5  # 1.5x buffer for gaps
        episode_days = int(episode_length / bars_per_day * 1.5) + 5

        # CRITICAL FIX: Start fetching from BEFORE the period to include lookback
        data_start = period_start - timedelta(days=lookback_days)

        # For random starts, we need extra data within the period
        if random_start:
            # Need enough data in the period to sample different start points
            # Extra data = potential random offset
            extra_days = int((episode_length * 2) / bars_per_day * 1.5)
            data_end = period_end + timedelta(days=extra_days)
        else:
            data_end = period_end + timedelta(days=episode_days)

        logger.info(f"\nFetching data range:")
        logger.info(f"  From: {data_start.date()} (includes {lookback_days} days for lookback)")
        logger.info(f"  To:   {data_end.date()}")

        # Load data for all symbols
        for symbol in symbols:
            logger.info(f"\nLoading {symbol}...")

            # Fetch data - includes lookback period
            df = self.get_historical_data_range(symbol, interval, data_start, data_end)

            if df is None or df.empty:
                logger.error(f"  Failed to load data for {symbol}")
                return False

            logger.info(f"  Fetched {len(df)} bars from {df.index[0]} to {df.index[-1]}")

            # DON'T filter to period_start/period_end - we need the lookback data!
            # Just keep all the data we fetched

            # Verify we have enough bars
            total_bars_in_period = len(df[df.index >= period_start])

            if total_bars_in_period < episode_length:
                logger.warning(f"  Insufficient bars in period for {symbol}: {total_bars_in_period} "
                               f"(need {episode_length})")

                # Try fetching more data
                data_end = period_end + timedelta(days=episode_days * 2)
                df = self.get_historical_data_range(symbol, interval, data_start, data_end)

                if df is None or len(df[df.index >= period_start]) < episode_length:
                    logger.error(f"  Still insufficient data for {symbol}")
                    return False

            self.cached_data[symbol] = df
            min_length = min(min_length, len(df))
            logger.info(f"  Cached {symbol}: {len(df)} bars total")

        # Find where period_start is in the cached data
        first_symbol = symbols[0]
        df = self.cached_data[first_symbol]

        # Find the index where period_start begins
        period_data = df[df.index >= period_start]
        if period_data.empty:
            logger.error("No data within the target period!")
            return False

        # The first index in the period
        period_start_idx = df.index.get_loc(period_data.index[0])

        # We want to start trading at period_start_idx
        # But we need lookback_bars before it
        min_start_idx = period_start_idx  # Can't start before period begins

        # How many bars do we have before period_start?
        available_lookback = period_start_idx

        if available_lookback < lookback_bars:
            logger.warning(f"Only {available_lookback} lookback bars available, need {lookback_bars}")
            # Adjust or fail
            if available_lookback < lookback_bars * 0.8:  # At least 80% of requested lookback
                logger.error("Insufficient lookback data")
                return False
            else:
                logger.info(f"Continuing with {available_lookback} lookback bars")
                actual_lookback = available_lookback
        else:
            actual_lookback = lookback_bars

        # Calculate how many episodes can fit in the period
        period_end_idx = len(df) - 1  # Last index in data

        # Available bars in period for episodes
        available_for_episodes = period_end_idx - period_start_idx + 1

        if available_for_episodes < episode_length:
            logger.error(f"Not enough bars in period: {available_for_episodes} < {episode_length}")
            return False

        # Calculate possible start positions for random sampling
        # We can start anywhere from period_start_idx to (period_end_idx - episode_length)
        available_starts = available_for_episodes - episode_length

        if available_starts < 1:
            logger.error(f"Not enough data for episodes")
            return False

        logger.info(f"\nData organization:")
        logger.info(f"  Total cached bars: {len(df)}")
        logger.info(f"  Period starts at index: {period_start_idx}")
        logger.info(f"  Available lookback: {actual_lookback} bars")
        logger.info(f"  Available for episodes: {available_for_episodes} bars")
        logger.info(f"  Possible start positions: {available_starts}")

        # Set start offset for episode
        if start_offset is not None:
            # EXPLICIT OFFSET (for controlled sampling)
            offset = min(start_offset, available_starts - 1)
            logger.info(f"  Using explicit offset: {offset}")

            # Track segment usage (NEW)
            self.segment_usage[offset] += 1
            reuse_count = self.segment_usage[offset]
            logger.info(f"  ✓ Segment {offset} used {reuse_count} time(s)")

        elif random_start and available_starts > 0:
            # RANDOM OFFSET
            offset = np.random.randint(0, available_starts)
            logger.info(f"  Random offset: {offset}")

            # Track segment usage (NEW)
            self.segment_usage[offset] += 1
            reuse_count = self.segment_usage[offset]

        else:
            # FIXED OFFSET
            offset = 0
            logger.info(f"  Fixed offset: {offset}")
            self.segment_usage[offset] += 1

        # Increment episode counter (NEW)
        self.episode_count += 1

        # Set indices
        # Episode starts at period_start_idx + offset
        self.episode_start_index = period_start_idx + offset
        self.episode_end_index = self.episode_start_index + episode_length
        self.current_index = self.episode_start_index
        self.max_cache_index = len(df) - 1
        self._cache_mode = True

        # Log episode info with timestamps
        start_ts = df.index[self.episode_start_index]
        end_ts = df.index[min(self.episode_end_index - 1, self.max_cache_index)]

        # Show lookback range
        lookback_start_idx = max(0, self.episode_start_index - actual_lookback)
        lookback_start_ts = df.index[lookback_start_idx]

        logger.info(f"\nEpisode configuration:")
        logger.info(f"  Lookback range: index {lookback_start_idx} to {self.episode_start_index - 1}")
        logger.info(f"                  time {lookback_start_ts} to {df.index[self.episode_start_index - 1]}")
        logger.info(f"  Episode range:  index {self.episode_start_index} to {self.episode_end_index - 1}")
        logger.info(f"                  time {start_ts} to {end_ts}")
        logger.info(f"  Episode length: {episode_length} bars")
        logger.info(f"  Current index: {self.current_index}")
        logger.info(f"  Max index: {self.max_cache_index}")

        # Log diversity statistics (NEW)
        unique_segments_used = len(self.segment_usage)
        avg_reuse = sum(self.segment_usage.values()) / max(1, len(self.segment_usage))
        coverage = unique_segments_used / max(1, available_starts)

        logger.info(f"\nEpisode diversity stats:")
        logger.info(f"  Total episodes: {self.episode_count}")
        logger.info(f"  Unique segments: {unique_segments_used}/{available_starts}")
        logger.info(f"  Coverage: {coverage:.1%}")
        logger.info(f"  Avg reuse: {avg_reuse:.1f}x")
        # Verify lookback is available
        if self.episode_start_index < actual_lookback:
            logger.warning(f"⚠️  Episode starts at index {self.episode_start_index}, "
                           f"but need {actual_lookback} bars of lookback!")
            logger.warning("  Features may not have full history at start")

        logger.info(f"{'=' * 70}\n")

        return True

    # Also update get_historical_data to ensure it respects the full cached range
    def get_historical_data(self, symbol: str, timeframe: str = '1Day', limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Get historical data - cache-aware

        Returns bars from (current_index - limit) to current_index (exclusive)
        This gives you 'limit' bars of history BEFORE current position
        """
        # CACHE MODE: Return historical window from cache
        if self._cache_mode and symbol in self.cached_data:
            df = self.cached_data[symbol]

            # Historical data should be BEFORE current_index
            start_idx = max(0, self.current_index - limit)
            end_idx = self.current_index  # Exclusive - don't include current bar

            if start_idx >= end_idx:
                logger.warning(f"Invalid indices for {symbol}: [{start_idx}:{end_idx}]")
                return pd.DataFrame()

            historical = df.iloc[start_idx:end_idx].copy()

            logger.debug(f"Historical data for {symbol}: "
                         f"indices [{start_idx}:{end_idx}] = {len(historical)} bars, "
                         f"from {historical.index[0] if len(historical) > 0 else 'N/A'} "
                         f"to {historical.index[-1] if len(historical) > 0 else 'N/A'}")

            return historical

        # NOT IN CACHE MODE: Fetch from network
        return self._fetch_historical_uncached(symbol, timeframe, limit)

    def _calculate_bars_per_day(self, interval: str) -> int:
        """
        Calculate approximate number of bars per trading day for given interval

        Args:
            interval: Interval string (1Min, 5Min, 1Hour, 1Day, etc.)

        Returns:
            Approximate bars per trading day
        """
        # Normalize interval to lowercase
        interval_lower = interval.lower()

        # Trading day is typically 6.5 hours (390 minutes)
        # For most US markets: 9:30 AM - 4:00 PM ET

        if 'min' in interval_lower or 'm' == interval_lower[-1]:
            # Extract number of minutes
            if interval_lower.endswith('min'):
                minutes = int(interval_lower.replace('min', ''))
            else:
                minutes = int(interval_lower.replace('m', ''))

            bars_per_day = 390 // minutes

        elif 'hour' in interval_lower or 'h' == interval_lower[-1]:
            # Extract number of hours
            if interval_lower.endswith('hour') or interval_lower.endswith('hours'):
                hours = int(interval_lower.replace('hour', '').replace('s', ''))
            else:
                hours = int(interval_lower.replace('h', ''))

            bars_per_day = int(6.5 / hours)

        elif 'day' in interval_lower or 'd' == interval_lower[-1]:
            # Daily bars - 1 bar per day
            bars_per_day = 1

        elif 'week' in interval_lower or 'w' == interval_lower[-1]:
            # Weekly bars - ~0.2 bars per day (5 trading days per week)
            bars_per_day = 0.2

        elif 'month' in interval_lower or 'mo' == interval_lower[-1]:
            # Monthly bars - ~0.05 bars per day (20 trading days per month)
            bars_per_day = 0.05

        else:
            # Unknown interval, default to 1 bar per day
            logger.warning(f"Unknown interval '{interval}', defaulting to 1 bar/day")
            bars_per_day = 1

        return max(1, int(bars_per_day))

    def validate_no_lookahead(self) -> bool:
        """
        Validate that no future data is accessed
        Run this during development to verify safety
        """
        if not self._cache_mode:
            logger.warning("Not in cache mode, skipping validation")
            return True

        logger.info("\n🔍 VALIDATING NO LOOK-AHEAD BIAS...")

        symbols = list(self.cached_data.keys())
        if not symbols:
            return True

        test_symbol = symbols[0]
        original_index = self.current_index

        # FIX: Get interval from config instead of hardcoding '1Min'
        interval = self.config.get('simulation', {}).get('timeframe', '1Min')

        # Test at multiple positions
        test_positions = [
            self.episode_start_index,
            self.episode_start_index + 50,
            min(self.episode_start_index + 100, self.episode_end_index - 1)
        ]

        all_valid = True

        for test_idx in test_positions:
            self.current_index = test_idx

            # FIX: Use interval from config, not hardcoded '1Min'
            hist = self.get_historical_data(test_symbol, interval, 100)

            if hist is None or hist.empty:
                continue

            # Get current bar info
            cached = self.cached_data[test_symbol]
            current_ts = cached.index[test_idx]
            hist_last_ts = hist.index[-1]

            # Find indices
            hist_last_idx = cached.index.get_loc(hist_last_ts)

            logger.info(f"  Position {test_idx}:")
            logger.info(f"    Current time: {current_ts}")
            logger.info(f"    Historical ends: {hist_last_ts} (index {hist_last_idx})")

            # Critical check: historical shouldn't go beyond current
            if hist_last_idx > test_idx:
                logger.error(f"    ❌ LOOK-AHEAD! Historical uses index {hist_last_idx} > {test_idx}")
                all_valid = False
            else:
                logger.info(f"    ✓ No future access")

        # Restore position
        self.current_index = original_index

        if all_valid:
            logger.info("✅ NO LOOK-AHEAD BIAS DETECTED\n")
        else:
            logger.error("❌ LOOK-AHEAD BIAS FOUND\n")

        return all_valid

    def advance_cache(self) -> bool:
        """Move to next timestep"""
        if not self._cache_mode:
            logger.warning("advance_cache() called but cache_mode is False")
            return False

        self.current_index += 1

        if self.current_index >= self.max_cache_index:
            logger.info(f"Reached end of cached data at index {self.current_index}")
            return False

        return True

    def reset_cache_position(self, start_index: int = 100):
        """Reset to beginning of cache"""
        if self._cache_mode:
            self.current_index = start_index
            logger.debug(f"Cache position reset to {start_index}")

    def get_historical_data_range(self, symbol: str, interval: str,
                                  start_date: datetime, end_date: datetime,
                                  progress_callback=None) -> Optional[pd.DataFrame]:
        """
        Get historical data for specific date range (supports any duration)

        Args:
            symbol: Stock symbol
            interval: Data interval (1m, 5m, 1h, 1d, etc.)
            start_date: Start date
            end_date: End date
            progress_callback: Optional progress callback

        Returns:
            DataFrame with OHLCV data
        """

        # NORMALIZE INTERVAL FIRST
        try:
            interval = normalize_interval(interval)
            logger.debug(f"Using normalized interval: {interval}")
        except ValueError as e:
            logger.error(f"Invalid interval format: {e}")
            return None

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # Check cache
        cached_data = self.cache.get_data(symbol, start_str, end_str, interval)
        if cached_data is not None:
            self.cache_hit_count += 1
            logger.info(f"Cache hit: {symbol} {interval} from {start_str} to {end_str}")
            return cached_data

        # Download with chunks
        logger.info(f"Downloading {symbol} {interval} from {start_str} to {end_str}")
        df = self.downloader.download_with_chunks(
            symbol, interval, start_date, end_date, progress_callback
        )

        if df is not None and not df.empty:
            # Cache the result
            self.cache.save_data(symbol, start_str, end_str, interval, df)
            self.download_count += 1

            # Log download stats
            stats = self.downloader.get_stats()
            logger.info(f"Download complete: {stats['successful_chunks']}/{stats['total_chunks']} chunks, "
                        f"{stats['total_rows']} total rows")

        return df

    def get_latest_data(self, symbols: List[str] = None) -> Dict:
        """Get latest market data"""
        if symbols is None:
            symbols = self.config.get('symbols_to_trade', [])

        # CACHE MODE: Return cached data at current index
        if self._cache_mode and self.cached_data:
            latest_data = {}
            for symbol in symbols:
                if symbol not in self.cached_data:
                    continue

                df = self.cached_data[symbol]
                if self.current_index >= len(df):
                    continue

                row = df.iloc[self.current_index]
                latest_data[symbol] = {
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": int(row['volume']),
                    "timestamp": row.name.isoformat() if hasattr(row.name, 'isoformat') else str(row.name),
                    "bid": float(row['close']),
                    "ask": float(row['close']),
                    "last_trade": float(row['close'])
                }

            self.last_market_data = latest_data
            return latest_data

        # NOT IN CACHE MODE: Fetch latest bar
        latest_data = {}
        for symbol in symbols:
            try:
                df = self._fetch_historical_uncached(symbol, '1Day', limit=1)
                if df is not None and not df.empty:
                    last_row = df.iloc[-1]
                    latest_data[symbol] = {
                        'open': float(last_row['open']),
                        'high': float(last_row['high']),
                        'low': float(last_row['low']),
                        'close': float(last_row['close']),
                        'volume': int(last_row['volume']),
                        'timestamp': df.index[-1].isoformat(),
                        'bid': float(last_row['close']),
                        'ask': float(last_row['close']),
                        'last_trade': float(last_row['close'])
                    }
            except Exception as e:
                logger.error(f"Error getting latest data for {symbol}: {e}")

        self.last_market_data = latest_data
        return latest_data

    def _fetch_historical_uncached(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Internal method to fetch from Yahoo Finance (respects limits)"""
        # Map timeframe
        try:
            interval = normalize_interval(timeframe)
        except ValueError:
            # Fallback to '1d' if normalization fails
            logger.warning(f"Could not normalize timeframe '{timeframe}', using '1d'")
            interval = '1d'

        # Get limitation info
        limit_info = YFINANCE_LIMITS.get(interval, {'max_days': None})
        max_days = limit_info['max_days']

        # Calculate date range
        end_date = datetime.now()

        # Estimate days needed based on interval and limit
        if interval == '1m':
            # ~390 minutes per trading day
            days_needed = max(7, (limit // 390) + 2)
        elif interval in ['5m', '15m', '30m']:
            days_needed = max(30, (limit // 78) + 2)
        elif interval in ['1h', '60m']:
            days_needed = max(60, (limit // 6) + 2)
        else:  # daily
            days_needed = limit + 2

        # Respect maximum limit
        if max_days is not None:
            days_needed = min(days_needed, max_days)

        start_date = end_date - timedelta(days=days_needed)

        # Use chunked download if needed
        df = self.get_historical_data_range(symbol, interval, start_date, end_date)

        if df is not None and not df.empty:
            return df.tail(limit)

        return None

    def get_cache_stats(self) -> Dict:
        """Get comprehensive statistics"""
        cache_stats = self.cache.get_stats()
        download_stats = self.downloader.get_stats()

        return {
            **cache_stats,
            'downloads': self.download_count,
            'cache_hits': self.cache_hit_count,
            'hit_rate': self.cache_hit_count / max(1, self.download_count + self.cache_hit_count),
            'download_stats': download_stats
        }

    def get_episode_diversity_stats(self) -> Dict:
        """Get statistics about episode diversity and data reuse"""

        available_positions = self.episode_config.get_estimated_episodes_available('train')

        if not self.segment_usage:
            return {
                'total_episodes': self.episode_count,
                'unique_segments_visited': 0,
                'total_possible_segments': available_positions,
                'coverage': 0.0,
                'avg_reuse': 0.0,
                'max_reuse': 0,
                'min_reuse': 0
            }

        reuse_counts = list(self.segment_usage.values())

        return {
            'total_episodes': self.episode_count,
            'unique_segments_visited': len(self.segment_usage),
            'total_possible_segments': available_positions,
            'coverage': len(self.segment_usage) / max(1, available_positions),
            'avg_reuse': np.mean(reuse_counts),
            'max_reuse': max(reuse_counts),
            'min_reuse': min(reuse_counts),
            'most_used_segment': max(self.segment_usage.items(), key=lambda x: x[1])[0],
            'least_used_segment': min(self.segment_usage.items(), key=lambda x: x[1])[0]
        }
# ============================================================================
# LIVE DATA PROVIDER (ALPACA IEX)
# ============================================================================

class LiveDataProvider(BaseDataProvider):
    """Live data provider using Alpaca IEX feed (free tier compatible)"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.market_cache = {}
        self.last_known_prices = {}
        self.last_update = {}

        if tradeapi is None:
            raise ImportError("alpaca_trade_api required for live trading: pip install alpaca-trade-api")

        try:
            self.api = tradeapi.REST(
                os.getenv("ALPACA_API_KEY"),
                os.getenv("ALPACA_API_SECRET"),
                config["base_url"],
                api_version='v2'
            )
            logger.info("LiveDataProvider initialized with Alpaca IEX feed")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            self.api = None

    def initialize_cache_for_episode(self, lookback_bars: int = 1000) -> bool:
        """Not used in live mode"""
        return True

    def get_historical_data(self, symbol: str, timeframe: str = '1Day', limit: int = 100) -> Optional[pd.DataFrame]:
        """Get historical data using IEX feed"""
        if self.api is None:
            return self._create_synthetic_data(symbol, limit)

        try:
            # NORMALIZE INTERVAL FIRST
            try:
                normalized_timeframe = normalize_interval(timeframe)
            except ValueError:
                logger.warning(f"Could not normalize timeframe '{timeframe}', using as-is")
                normalized_timeframe = timeframe

            # Map normalized interval to Alpaca timeframes
            timeframe_map = {
                '1m': tradeapi.TimeFrame.Minute,
                '5m': tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute),
                '15m': tradeapi.TimeFrame(15, tradeapi.TimeFrameUnit.Minute),
                '30m': tradeapi.TimeFrame(30, tradeapi.TimeFrameUnit.Minute),
                '1h': tradeapi.TimeFrame.Hour,
                '1d': tradeapi.TimeFrame.Day,
            }
            tf = timeframe_map.get(normalized_timeframe, tradeapi.TimeFrame.Day)

            # Calculate time range
            eastern = pytz.timezone('America/New_York')
            end = datetime.now(eastern)

            if timeframe == '1Min':
                start = end - timedelta(days=5)
            elif timeframe == '1Day':
                start = end - timedelta(days=limit * 2)
            else:
                start = end - timedelta(days=10)

            logger.info(f"Fetching {symbol} IEX data: {timeframe}")

            # Use IEX feed for free tier
            bars = self.api.get_bars(
                symbol,
                tf,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                feed='iex',
                adjustment='raw',
                limit=1000
            )

            # Convert to DataFrame
            if hasattr(bars, 'df'):
                df = bars.df
            else:
                rows = []
                for bar in bars:
                    rows.append({
                        'timestamp': getattr(bar, 't', getattr(bar, 'time', None)),
                        'open': float(getattr(bar, 'o', 0)),
                        'high': float(getattr(bar, 'h', 0)),
                        'low': float(getattr(bar, 'l', 0)),
                        'close': float(getattr(bar, 'c', 0)),
                        'volume': int(getattr(bar, 'v', 0))
                    })
                df = pd.DataFrame(rows)
                df.set_index('timestamp', inplace=True)

            if not df.empty:
                df.columns = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap'][:len(df.columns)]
                df = df[['open', 'high', 'low', 'close', 'volume']].fillna(method='ffill')
                df.index = pd.to_datetime(df.index)
                return df.tail(min(limit, len(df)))

            logger.warning(f"No IEX data for {symbol}, using synthetic")
            return self._create_synthetic_data(symbol, limit)

        except Exception as e:
            logger.error(f"Error fetching IEX data for {symbol}: {e}")
            return self._create_synthetic_data(symbol, limit)

    def get_latest_data(self, symbols: List[str] = None) -> Dict:
        """Get latest quotes using IEX feed"""
        if symbols is None:
            symbols = self.config.get('symbols_to_trade', [])

        if self.api is None:
            return {s: self._create_synthetic_quote(s) for s in symbols}

        latest_data = {}

        for symbol in symbols:
            try:
                # Get latest trade from IEX
                latest_trade = self.api.get_latest_trade(symbol, feed='iex')

                if latest_trade:
                    # Try to get quote for bid/ask
                    try:
                        latest_quote = self.api.get_latest_quote(symbol, feed='iex')
                        bid = float(latest_quote.bid_price) if latest_quote.bid_price else latest_trade.price * 0.999
                        ask = float(latest_quote.ask_price) if latest_quote.ask_price else latest_trade.price * 1.001
                    except:
                        bid = latest_trade.price * 0.999
                        ask = latest_trade.price * 1.001

                    latest_data[symbol] = {
                        'symbol': symbol,
                        'open': float(latest_trade.price),
                        'high': float(latest_trade.price),
                        'low': float(latest_trade.price),
                        'close': float(latest_trade.price),
                        'bid': float(bid),
                        'ask': float(ask),
                        'volume': float(latest_trade.size),
                        'last_trade': float(latest_trade.price),
                        'timestamp': latest_trade.timestamp.isoformat()
                    }

                    self.market_cache[symbol] = latest_data[symbol]
                    self.last_known_prices[symbol] = latest_trade.price
                    self.last_update[symbol] = datetime.now()

            except Exception as e:
                logger.warning(f"Could not get IEX data for {symbol}: {e}")
                # Use cached or synthetic
                if symbol in self.market_cache:
                    latest_data[symbol] = self.market_cache[symbol]
                else:
                    latest_data[symbol] = self._create_synthetic_quote(symbol)

        return latest_data

    def _create_synthetic_data(self, symbol: str, bars: int) -> pd.DataFrame:
        """Create synthetic data when real data unavailable"""
        logger.warning(f"Creating synthetic data for {symbol}")

        dates = pd.date_range(end=datetime.now(), periods=bars, freq='1Min')
        base_price = self.last_known_prices.get(symbol, 100.0)

        returns = np.random.normal(0, 0.001, bars)
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'open': prices * np.random.uniform(0.998, 1.002, bars),
            'high': prices * np.random.uniform(1.001, 1.01, bars),
            'low': prices * np.random.uniform(0.99, 0.999, bars),
            'close': prices,
            'volume': np.random.uniform(1e6, 1e7, bars)
        }, index=dates)

    def _create_synthetic_quote(self, symbol: str) -> Dict:
        """Create synthetic quote"""
        price = self.last_known_prices.get(symbol, 100.0)
        return {
            'symbol': symbol,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'bid': price * 0.999,
            'ask': price * 1.001,
            'volume': 1000000,
            'last_trade': price,
            'timestamp': datetime.now().isoformat()
        }


# ============================================================================
# UNIFIED DATA PROVIDER FACTORY
# ============================================================================

def create_data_provider(config: Dict) -> BaseDataProvider:
    """
    Factory function to create appropriate data provider based on config

    Args:
        config: Configuration dictionary with 'mode' key

    Returns:
        BaseDataProvider instance (SimulationDataProvider or LiveDataProvider)
    """
    mode = str(config.get('mode', 'live')).lower()

    if mode in ['simulation', 'rl_training', 'backtesting']:
        cache_dir = config.get('simulation', {}).get('cache_dir', 'simulation_cache')
        return SimulationDataProvider(config, cache_dir)
    else:
        return LiveDataProvider(config)


# ============================================================================
# LIVE MARKET STATE PROVIDER
# ============================================================================

class LiveMarketStateProvider:
    """
    Provides current market state for live trading
    Works with any BaseDataProvider (simulation or live)
    """

    def __init__(self, data_provider: BaseDataProvider, portfolio_manager, config: Dict):
        self.data_provider = data_provider
        self.portfolio_manager = portfolio_manager
        self.config = config

        # Import here to avoid circular dependency
        from temporal_rl_system import AggregatedFeatureExtractor
        self.feature_extractor = AggregatedFeatureExtractor()

        # State tracking
        self.last_update = None
        self.update_interval = 60  # seconds
        self._last_portfolio_value = None
        self._recent_returns = []
        self._peak_value = None

    def get_current_market_state(self, symbols: List[str] = None) -> np.ndarray:
        """Get current market state vector for inference (60-dimensional)"""
        if symbols is None:
            symbols = self.config.get('symbols_to_trade', [])

        # FIX: Get interval from config
        interval = self.config.get('simulation', {}).get('timeframe', '1Min')

        # For live trading, always use 1Min regardless of config
        # (Alpaca provides minute-level data for live trading)
        if self.config.get('mode', 'simulation') == 'live':
            interval = '1Min'

        try:
            # Get data
            historical_data = {}
            for symbol in symbols:
                # FIX: Use interval from config, not hardcoded '1Min'
                hist = self.data_provider.get_historical_data(symbol, interval, 100)
                if hist is not None and not hist.empty:
                    historical_data[symbol] = hist

            current_market_data = self.data_provider.get_latest_data(symbols)
            portfolio_state = self._get_portfolio_state(current_market_data)

            # Extract features
            state = self.feature_extractor.extract_features(
                historical_data, current_market_data, portfolio_state
            )

            # Ensure correct shape
            if state.shape[0] != 60:
                state = self._ensure_dimension(state, 60)

            return state

        except Exception as e:
            logger.error(f"Error getting market state: {e}")
            return np.zeros(60, dtype=np.float32)


    def _get_portfolio_state(self, current_market_data: Dict) -> Dict:
        """Get current portfolio state with metrics"""
        try:
            current_value = self.portfolio_manager.get_portfolio_value(current_market_data)
            cash = self.portfolio_manager.get_available_cash()
            positions = self.portfolio_manager.get_current_holdings()

            # Position ratio
            position_values = [
                qty * current_market_data.get(sym, {}).get('close', 0)
                for sym, qty in positions.items()
            ]
            total_position_value = sum(position_values) if position_values else 0
            position_ratio = total_position_value / current_value if current_value > 0 else 0

            # Daily return
            if self._last_portfolio_value:
                daily_return = (current_value - self._last_portfolio_value) / self._last_portfolio_value
            else:
                daily_return = 0
                self._last_portfolio_value = current_value

            # Track returns for Sharpe
            self._recent_returns.append(daily_return)
            if len(self._recent_returns) > 100:
                self._recent_returns.pop(0)

            # Sharpe estimate
            if len(self._recent_returns) > 10:
                returns_array = np.array(self._recent_returns)
                sharpe = np.mean(returns_array) / (np.std(returns_array) + 1e-8)
            else:
                sharpe = 0

            # Drawdown
            if self._peak_value is None:
                self._peak_value = current_value
            else:
                self._peak_value = max(self._peak_value, current_value)

            drawdown = (self._peak_value - current_value) / self._peak_value if self._peak_value > 0 else 0

            return {
                'cash': cash,
                'total_value': current_value,
                'unrealized_pnl': current_value - self.portfolio_manager.initial_capital,
                'positions': positions,
                'position_ratio': position_ratio,
                'daily_return': daily_return,
                'drawdown': drawdown,
                'sharpe_estimate': sharpe
            }

        except Exception as e:
            logger.error(f"Error getting portfolio state: {e}")
            return {
                'cash': self.portfolio_manager.get_available_cash(),
                'total_value': self.portfolio_manager.initial_capital,
                'unrealized_pnl': 0,
                'positions': {},
                'position_ratio': 0,
                'daily_return': 0,
                'drawdown': 0,
                'sharpe_estimate': 0
            }

    def _ensure_dimension(self, state: np.ndarray, target_dim: int) -> np.ndarray:
        """Ensure state has correct dimension"""
        state = state.flatten()
        if state.shape[0] < target_dim:
            state = np.pad(state, (0, target_dim - state.shape[0]), 'constant')
        elif state.shape[0] > target_dim:
            state = state[:target_dim]
        return state.astype(np.float32)


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_market_data(market_data: Dict) -> bool:
    """Validate market data quality"""
    if not market_data:
        return False

    required_fields = ['close', 'volume']
    valid_symbols = 0

    for symbol, data in market_data.items():
        if not isinstance(data, dict):
            continue

        if not all(field in data for field in required_fields):
            continue

        close = data.get('close', 0)
        volume = data.get('volume', 0)

        if not isinstance(close, (int, float)) or close <= 0:
            continue

        if not isinstance(volume, (int, float)) or volume < 0:
            continue

        valid_symbols += 1

    return (valid_symbols / len(market_data)) >= 0.5 if market_data else False


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import sys

    # Example 1: Download large date range respecting limits
    print("=" * 80)
    print("Example 1: Downloading 90 days of 1-minute data (respects 7-day limit)")
    print("=" * 80)

    config = {
        'mode': 'simulation',
        'symbols_to_trade': ['AAPL'],
        'simulation': {'cache_dir': 'test_cache'}
    }

    provider = create_data_provider(config)

    # This will automatically chunk the request
    start = datetime.now() - timedelta(days=90)
    end = datetime.now()

    df = provider.get_historical_data_range(
        'AAPL', '1m', start, end,
        progress_callback=lambda curr, total, msg: print(f"  Progress: {curr}/{total} - {msg}")
    )

    if df is not None:
        print(f"\n✓ Downloaded {len(df)} rows")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"\n{df.head()}")

        # Show stats
        stats = provider.get_cache_stats()
        print(f"\nCache Stats:")
        print(f"  Downloads: {stats['downloads']}")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(
            f"  Download chunks: {stats['download_stats']['successful_chunks']}/{stats['download_stats']['total_chunks']}")

    # Example 2: Different intervals
    print("\n" + "=" * 80)
    print("Example 2: Different intervals")
    print("=" * 80)

    intervals = [
        ('5m', 60, "60 days of 5-minute"),
        ('1h', 365, "365 days of hourly"),
        ('1d', 1000, "1000 days of daily")
    ]

    for interval, days, description in intervals:
        print(f"\n{description} data:")
        start = datetime.now() - timedelta(days=days)
        df = provider.get_historical_data_range('MSFT', interval, start, datetime.now())
        if df is not None:
            print(f"  ✓ {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")