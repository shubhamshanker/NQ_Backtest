"""
Precomputed Cumulative Sums for O(1) Moving Averages
===================================================
Task 2: Generate precomputed cumsum arrays for instant indicator calculations
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from numba import njit


class PrecomputeEngine:
    """Generate and manage precomputed data for ultra-fast indicator calculations."""

    def __init__(self, cache_dir: str = "backtesting/cache"):
        self.cache_dir = Path(cache_dir)
        self.precomputed_dir = self.cache_dir / "precomputed"
        self.precomputed_dir.mkdir(parents=True, exist_ok=True)

    def create_precomputed_data(self, arrays: Dict[str, np.ndarray], cache_key: str) -> Dict[str, str]:
        """
        Create precomputed data structures from memory-mapped arrays.

        Args:
            arrays: Memory-mapped arrays from data_loader
            cache_key: Unique identifier for this dataset

        Returns:
            Dict with paths to precomputed files
        """
        print(f"🔄 Creating precomputed data for {cache_key}...")
        start_time = time.time()

        n_bars = len(arrays['close'])
        price_multiplier = arrays['metadata']['price_multiplier']

        precomputed_files = {
            'cumsum_close': self.precomputed_dir / f"{cache_key}_cumsum_close.dat",
            'cumsum_high': self.precomputed_dir / f"{cache_key}_cumsum_high.dat",
            'cumsum_low': self.precomputed_dir / f"{cache_key}_cumsum_low.dat",
            'cumsum_volume': self.precomputed_dir / f"{cache_key}_cumsum_volume.dat",
            'session_bounds': self.precomputed_dir / f"{cache_key}_session_bounds.npy",
            'metadata': self.precomputed_dir / f"{cache_key}_precompute_meta.json"
        }

        # Create cumulative sum arrays (int64 to prevent overflow)
        cumsum_close = np.memmap(precomputed_files['cumsum_close'], dtype='int64', mode='w+', shape=(n_bars,))
        cumsum_high = np.memmap(precomputed_files['cumsum_high'], dtype='int64', mode='w+', shape=(n_bars,))
        cumsum_low = np.memmap(precomputed_files['cumsum_low'], dtype='int64', mode='w+', shape=(n_bars,))

        # Generate cumulative sums
        cumsum_close[:] = arrays['close'].astype('int64').cumsum()
        cumsum_high[:] = arrays['high'].astype('int64').cumsum()
        cumsum_low[:] = arrays['low'].astype('int64').cumsum()

        # Volume cumsum if available
        if 'volume' in arrays:
            cumsum_volume = np.memmap(precomputed_files['cumsum_volume'], dtype='int64', mode='w+', shape=(n_bars,))
            cumsum_volume[:] = arrays['volume'].astype('int64').cumsum()
            del cumsum_volume

        # Generate session boundaries (9:30-16:00 ET detection)
        session_bounds = self._detect_session_bounds(arrays['timestamps'])
        np.save(precomputed_files['session_bounds'], session_bounds)

        # Force write to disk
        del cumsum_close, cumsum_high, cumsum_low

        # Save metadata
        metadata = {
            'source_cache_key': cache_key,
            'n_bars': n_bars,
            'n_sessions': len(session_bounds),
            'price_multiplier': price_multiplier,
            'created_at': time.time(),
            'file_size_mb': sum(f.stat().st_size for f in precomputed_files.values() if f.exists() and f.suffix == '.dat') / 1024 / 1024
        }

        with open(precomputed_files['metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)

        computation_time = time.time() - start_time
        print(f"✅ Precomputation completed in {computation_time:.2f}s")
        print(f"📦 Precomputed cache size: {metadata['file_size_mb']:.1f} MB")
        print(f"🎯 {len(session_bounds)} trading sessions detected")

        return {k: str(v) for k, v in precomputed_files.items()}

    def load_precomputed_data(self, precomputed_files: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Load precomputed data for fast access."""

        # Load metadata
        with open(precomputed_files['metadata'], 'r') as f:
            metadata = json.load(f)

        n_bars = metadata['n_bars']

        # Load precomputed arrays
        precomputed = {
            'cumsum_close': np.memmap(precomputed_files['cumsum_close'], dtype='int64', mode='r', shape=(n_bars,)),
            'cumsum_high': np.memmap(precomputed_files['cumsum_high'], dtype='int64', mode='r', shape=(n_bars,)),
            'cumsum_low': np.memmap(precomputed_files['cumsum_low'], dtype='int64', mode='r', shape=(n_bars,)),
            'session_bounds': np.load(precomputed_files['session_bounds']),
            'metadata': metadata
        }

        # Load volume cumsum if exists
        cumsum_volume_path = Path(precomputed_files['cumsum_volume'])
        if cumsum_volume_path.exists():
            precomputed['cumsum_volume'] = np.memmap(cumsum_volume_path, dtype='int64', mode='r', shape=(n_bars,))

        return precomputed

    def _detect_session_bounds(self, timestamps: np.ndarray) -> np.ndarray:
        """
        Detect trading session boundaries (9:30-16:00 ET).

        Returns array of [session_start, session_end) indices.
        """
        # Convert Unix timestamps to datetime for session detection
        # Simplified: detect gaps > 16 hours as session boundaries
        time_diffs = np.diff(timestamps.astype('int64'))
        session_gap_threshold = 16 * 3600  # 16 hours in seconds

        # Find session breaks
        session_breaks = np.where(time_diffs > session_gap_threshold)[0] + 1

        # Create session bounds array
        session_starts = np.concatenate([[0], session_breaks])
        session_ends = np.concatenate([session_breaks, [len(timestamps)]])

        session_bounds = np.column_stack([session_starts, session_ends])
        return session_bounds


@njit(cache=True, fastmath=True)
def sma_instant(cumsum: np.ndarray, period: int, index: int) -> int:
    """
    Calculate Simple Moving Average instantly using precomputed cumsum.

    Args:
        cumsum: Precomputed cumulative sum array
        period: SMA period
        index: Current bar index

    Returns:
        SMA value as integer (multiply by PRICE_MULTIPLIER)
    """
    if index < period - 1:
        # Not enough data, return simple average
        return cumsum[index] // (index + 1)

    # O(1) calculation using cumsum difference
    return (cumsum[index] - cumsum[index - period]) // period


@njit(cache=True, fastmath=True)
def ema_update(prev_ema: int, current_price: int, alpha_int: int, alpha_scale: int = 10000) -> int:
    """
    Update EMA using integer arithmetic for speed.

    Args:
        prev_ema: Previous EMA value (integer)
        current_price: Current price (integer)
        alpha_int: Alpha multiplied by alpha_scale for integer math
        alpha_scale: Scale factor for alpha

    Returns:
        Updated EMA value as integer
    """
    # EMA = α * price + (1-α) * prev_ema
    # Using integer math: EMA = (alpha_int * price + (alpha_scale - alpha_int) * prev_ema) / alpha_scale
    return (alpha_int * current_price + (alpha_scale - alpha_int) * prev_ema) // alpha_scale


@njit(cache=True, fastmath=True)
def rolling_high_low(high_array: np.ndarray, low_array: np.ndarray,
                    index: int, period: int) -> tuple:
    """
    Calculate rolling high/low over a period.

    Args:
        high_array: High price array
        low_array: Low price array
        index: Current index
        period: Lookback period

    Returns:
        (rolling_high, rolling_low) as integers
    """
    start_idx = max(0, index - period + 1)
    end_idx = index + 1

    rolling_high = high_array[start_idx:end_idx].max()
    rolling_low = low_array[start_idx:end_idx].min()

    return rolling_high, rolling_low


@njit(cache=True, fastmath=True)
def session_high_low(high_array: np.ndarray, low_array: np.ndarray,
                    session_start: int, current_index: int) -> tuple:
    """
    Calculate session high/low from session start to current index.

    Args:
        high_array: High price array
        low_array: Low price array
        session_start: Session start index
        current_index: Current bar index

    Returns:
        (session_high, session_low) as integers
    """
    if current_index < session_start:
        return high_array[current_index], low_array[current_index]

    session_high = high_array[session_start:current_index + 1].max()
    session_low = low_array[session_start:current_index + 1].min()

    return session_high, session_low


@njit(cache=True, fastmath=True)
def compute_all_indicators(cumsum_close: np.ndarray, cumsum_high: np.ndarray,
                          cumsum_low: np.ndarray, high_array: np.ndarray,
                          low_array: np.ndarray, index: int,
                          session_start: int) -> tuple:
    """
    Compute all common indicators at once for maximum efficiency.

    Returns:
        (sma_20, sma_50, session_high, session_low, rolling_high_10, rolling_low_10)
    """
    sma_20 = sma_instant(cumsum_close, 20, index)
    sma_50 = sma_instant(cumsum_close, 50, index)

    session_high, session_low = session_high_low(high_array, low_array, session_start, index)
    rolling_high_10, rolling_low_10 = rolling_high_low(high_array, low_array, index, 10)

    return sma_20, sma_50, session_high, session_low, rolling_high_10, rolling_low_10


def benchmark_indicator_speed(arrays: Dict[str, np.ndarray],
                             precomputed: Dict[str, np.ndarray],
                             n_calculations: int = 10000) -> Dict[str, float]:
    """Benchmark indicator calculation speed: Traditional vs Precomputed."""

    print("🏁 Benchmarking Indicator Speed")
    print("=" * 40)

    close_prices = arrays['close']
    high_prices = arrays['high']
    low_prices = arrays['low']
    cumsum_close = precomputed['cumsum_close']
    cumsum_high = precomputed['cumsum_high']
    cumsum_low = precomputed['cumsum_low']

    # Test indices
    test_indices = np.random.randint(50, len(close_prices) - 1, n_calculations)

    # Benchmark traditional SMA calculation
    print("📊 Traditional SMA calculation...")
    start_time = time.time()
    for idx in test_indices:
        # Simulate traditional pandas rolling mean
        sma_20 = close_prices[max(0, idx-19):idx+1].mean()
    traditional_time = time.time() - start_time
    print(f"   Traditional SMA time: {traditional_time:.3f}s")

    # Benchmark precomputed SMA calculation
    print("⚡ Precomputed SMA calculation...")
    start_time = time.time()
    for idx in test_indices:
        sma_20 = sma_instant(cumsum_close, 20, idx)
    precomputed_time = time.time() - start_time
    print(f"   Precomputed SMA time: {precomputed_time:.3f}s")

    # Speedup calculation
    speedup = traditional_time / precomputed_time if precomputed_time > 0 else float('inf')

    print(f"\n✨ Indicator Speed Results:")
    print(f"   Speedup: {speedup:.1f}x faster")
    print(f"   Calculations per second: {n_calculations / precomputed_time:,.0f}")

    return {
        'traditional_time': traditional_time,
        'precomputed_time': precomputed_time,
        'speedup': speedup,
        'calculations_per_second': n_calculations / precomputed_time
    }


if __name__ == "__main__":
    # Example usage
    from .data_loader import UltraFastDataLoader

    parquet_path = "data_parquet/nq/1min"
    cache_key = "nq_1min_example"

    if Path(parquet_path).exists():
        # Load data arrays
        loader = UltraFastDataLoader()
        memmap_files = loader.parquet_to_memmap(parquet_path)
        arrays = loader.load_memmap_arrays(memmap_files)

        # Create precomputed data
        engine = PrecomputeEngine()
        precomputed_files = engine.create_precomputed_data(arrays, cache_key)
        precomputed = engine.load_precomputed_data(precomputed_files)

        # Benchmark
        results = benchmark_indicator_speed(arrays, precomputed)
        print(f"\n📊 Task 2 Complete: {results['speedup']:.1f}x faster indicators achieved!")
    else:
        print(f"❌ Parquet path not found: {parquet_path}")