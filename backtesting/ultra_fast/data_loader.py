"""
Ultra-Fast Data Loading with Memory-Mapped Arrays
================================================
Task 1: Convert Parquet to Memory-Mapped Integer Arrays for Zero-Copy Access
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import os
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from numba import njit


class UltraFastDataLoader:
    """Convert parquet data to memory-mapped arrays for ultra-fast access."""

    def __init__(self, cache_dir: str = "backtesting/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memmap_dir = self.cache_dir / "memmap"
        self.metadata_dir = self.cache_dir / "metadata"
        self.memmap_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

        # Price multiplier for integer conversion (5 decimal precision)
        self.PRICE_MULTIPLIER = 100000
        self.VOLUME_MULTIPLIER = 1  # Keep volume as-is

    def parquet_to_memmap(self, parquet_path: str, force_rebuild: bool = False) -> Dict[str, str]:
        """
        Convert parquet file to memory-mapped integer arrays.

        Args:
            parquet_path: Path to parquet file
            force_rebuild: Force rebuild even if cache exists

        Returns:
            Dict with paths to memory-mapped files
        """
        parquet_path = Path(parquet_path)
        cache_key = self._get_cache_key(parquet_path)

        memmap_files = {
            'timestamps': self.memmap_dir / f"{cache_key}_timestamps.dat",
            'open': self.memmap_dir / f"{cache_key}_open.dat",
            'high': self.memmap_dir / f"{cache_key}_high.dat",
            'low': self.memmap_dir / f"{cache_key}_low.dat",
            'close': self.memmap_dir / f"{cache_key}_close.dat",
            'volume': self.memmap_dir / f"{cache_key}_volume.dat",
            'metadata': self.metadata_dir / f"{cache_key}_metadata.json"
        }

        # Check if cache exists and is valid
        if not force_rebuild and self._cache_is_valid(parquet_path, memmap_files):
            print(f"✅ Using cached memmap for {parquet_path.name}")
            return {k: str(v) for k, v in memmap_files.items()}

        print(f"🔄 Converting {parquet_path.name} to memory-mapped arrays...")
        start_time = time.time()

        # Read parquet file
        try:
            if parquet_path.is_dir():
                # Partitioned parquet dataset
                df = pq.read_table(parquet_path).to_pandas()
            else:
                # Single parquet file
                df = pq.read_parquet(parquet_path)
        except Exception as e:
            raise ValueError(f"Failed to read parquet file {parquet_path}: {e}")

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Handle timestamp column
        timestamp_col = None
        for col in ['timestamp', 'datetime', 'time', 'date']:
            if col in df.columns:
                timestamp_col = col
                break

        if timestamp_col is None:
            raise ValueError("No timestamp column found")

        # Convert timestamps to Unix seconds (int32)
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        timestamps_unix = df[timestamp_col].astype('int64') // 10**9  # Convert to seconds

        # Sort by timestamp
        sort_idx = timestamps_unix.argsort()
        df = df.iloc[sort_idx]
        timestamps_unix = timestamps_unix.iloc[sort_idx]

        n_bars = len(df)
        print(f"📊 Processing {n_bars:,} bars")

        # Create memory-mapped arrays
        timestamps_mm = np.memmap(memmap_files['timestamps'], dtype='uint32', mode='w+', shape=(n_bars,))
        open_mm = np.memmap(memmap_files['open'], dtype='int32', mode='w+', shape=(n_bars,))
        high_mm = np.memmap(memmap_files['high'], dtype='int32', mode='w+', shape=(n_bars,))
        low_mm = np.memmap(memmap_files['low'], dtype='int32', mode='w+', shape=(n_bars,))
        close_mm = np.memmap(memmap_files['close'], dtype='int32', mode='w+', shape=(n_bars,))

        # Handle volume (optional)
        volume_mm = None
        if 'volume' in df.columns:
            volume_mm = np.memmap(memmap_files['volume'], dtype='uint32', mode='w+', shape=(n_bars,))

        # Convert and store data
        timestamps_mm[:] = timestamps_unix.values.astype('uint32')
        open_mm[:] = (df['open'].values * self.PRICE_MULTIPLIER).astype('int32')
        high_mm[:] = (df['high'].values * self.PRICE_MULTIPLIER).astype('int32')
        low_mm[:] = (df['low'].values * self.PRICE_MULTIPLIER).astype('int32')
        close_mm[:] = (df['close'].values * self.PRICE_MULTIPLIER).astype('int32')

        if volume_mm is not None:
            volume_mm[:] = df['volume'].values.astype('uint32')

        # Force write to disk
        del timestamps_mm, open_mm, high_mm, low_mm, close_mm, volume_mm

        # Save metadata
        metadata = {
            'source_file': str(parquet_path),
            'n_bars': n_bars,
            'date_range': [
                str(df[timestamp_col].min().date()),
                str(df[timestamp_col].max().date())
            ],
            'price_multiplier': self.PRICE_MULTIPLIER,
            'volume_multiplier': self.VOLUME_MULTIPLIER,
            'created_at': pd.Timestamp.now().isoformat(),
            'file_size_mb': sum(f.stat().st_size for f in memmap_files.values() if f.exists()) / 1024 / 1024
        }

        with open(memmap_files['metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)

        conversion_time = time.time() - start_time
        print(f"✅ Conversion completed in {conversion_time:.2f}s")
        print(f"📦 Cache size: {metadata['file_size_mb']:.1f} MB")

        return {k: str(v) for k, v in memmap_files.items()}

    def load_memmap_arrays(self, memmap_files: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Load memory-mapped arrays for fast access."""

        # Load metadata
        with open(memmap_files['metadata'], 'r') as f:
            metadata = json.load(f)

        n_bars = metadata['n_bars']

        # Load memory-mapped arrays
        arrays = {
            'timestamps': np.memmap(memmap_files['timestamps'], dtype='uint32', mode='r', shape=(n_bars,)),
            'open': np.memmap(memmap_files['open'], dtype='int32', mode='r', shape=(n_bars,)),
            'high': np.memmap(memmap_files['high'], dtype='int32', mode='r', shape=(n_bars,)),
            'low': np.memmap(memmap_files['low'], dtype='int32', mode='r', shape=(n_bars,)),
            'close': np.memmap(memmap_files['close'], dtype='int32', mode='r', shape=(n_bars,)),
        }

        # Load volume if exists
        volume_path = Path(memmap_files['volume'])
        if volume_path.exists():
            arrays['volume'] = np.memmap(volume_path, dtype='uint32', mode='r', shape=(n_bars,))

        arrays['metadata'] = metadata
        return arrays

    def _get_cache_key(self, parquet_path: Path) -> str:
        """Generate cache key from parquet path."""
        # Use path components to create unique key
        parts = parquet_path.parts
        if len(parts) >= 3:
            # e.g., data_parquet/nq/1min -> nq_1min
            return f"{parts[-3]}_{parts[-2]}_{parts[-1]}".replace('=', '_').replace('.parquet', '')
        else:
            return parquet_path.stem

    def _cache_is_valid(self, parquet_path: Path, memmap_files: Dict[str, Path]) -> bool:
        """Check if cache is valid and up-to-date."""
        try:
            # Check if all files exist
            for file_path in memmap_files.values():
                if not file_path.exists():
                    return False

            # Check if parquet file is newer than cache
            parquet_mtime = parquet_path.stat().st_mtime
            cache_mtime = memmap_files['metadata'].stat().st_mtime

            return cache_mtime > parquet_mtime
        except:
            return False


@njit(cache=True, fastmath=True)
def prices_to_float(prices_int: np.ndarray, multiplier: int = 100000) -> np.ndarray:
    """Convert integer prices back to float (Numba-compiled)."""
    return prices_int.astype(np.float64) / multiplier


@njit(cache=True, fastmath=True)
def get_price_range(open_int: np.ndarray, high_int: np.ndarray,
                   low_int: np.ndarray, close_int: np.ndarray,
                   start_idx: int, end_idx: int) -> tuple:
    """Get price range for a given period (Numba-compiled)."""
    period_high = high_int[start_idx:end_idx].max()
    period_low = low_int[start_idx:end_idx].min()
    period_open = open_int[start_idx]
    period_close = close_int[end_idx-1]

    return period_open, period_high, period_low, period_close


def benchmark_load_speed(parquet_path: str, cache_dir: str = "backtesting/cache") -> Dict[str, float]:
    """Benchmark loading speed: Parquet vs Memory-mapped arrays."""

    loader = UltraFastDataLoader(cache_dir)

    print("🏁 Benchmarking Load Speed")
    print("=" * 40)

    # Benchmark parquet loading
    print("📖 Loading from Parquet...")
    start_time = time.time()
    if Path(parquet_path).is_dir():
        df_parquet = pq.read_table(parquet_path).to_pandas()
    else:
        df_parquet = pq.read_table(parquet_path).to_pandas()
    parquet_time = time.time() - start_time
    print(f"   Parquet load time: {parquet_time:.3f}s")

    # Convert to memmap (one-time cost)
    memmap_files = loader.parquet_to_memmap(parquet_path)

    # Benchmark memmap loading
    print("⚡ Loading from Memory-mapped arrays...")
    start_time = time.time()
    arrays = loader.load_memmap_arrays(memmap_files)
    memmap_time = time.time() - start_time
    print(f"   Memmap load time: {memmap_time:.3f}s")

    # Speedup calculation
    speedup = parquet_time / memmap_time if memmap_time > 0 else float('inf')

    print(f"\n✨ Load Speed Results:")
    print(f"   Speedup: {speedup:.1f}x faster")
    print(f"   Memory usage reduction: ~{df_parquet.memory_usage(deep=True).sum() / arrays['metadata']['file_size_mb'] / 1024 / 1024:.1f}x")

    return {
        'parquet_time': parquet_time,
        'memmap_time': memmap_time,
        'speedup': speedup,
        'n_bars': len(df_parquet)
    }


if __name__ == "__main__":
    # Example usage and benchmark
    parquet_path = "data_parquet/nq/1min"

    if Path(parquet_path).exists():
        results = benchmark_load_speed(parquet_path)
        print(f"\n📊 Task 1 Complete: {results['speedup']:.1f}x faster loading achieved!")
    else:
        print(f"❌ Parquet path not found: {parquet_path}")