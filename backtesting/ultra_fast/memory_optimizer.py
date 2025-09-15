"""
Memory Layout Optimization for Cache Efficiency
==============================================
Task 10: Optimize memory layout and alignment for maximum cache performance
"""

import numpy as np
import time
from typing import Dict, List, Tuple
from numba import njit
from pathlib import Path
import sys


class MemoryOptimizer:
    """Optimize memory layout for cache efficiency and SIMD operations."""

    def __init__(self):
        self.CACHE_LINE_SIZE = 64  # bytes
        self.SIMD_ALIGNMENT = 32   # bytes for AVX2

    def optimize_array_layout(self, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Optimize memory layout of all arrays for cache efficiency.

        Args:
            arrays: Dictionary of numpy arrays

        Returns:
            Dictionary of optimized arrays
        """
        print("🔧 Optimizing memory layout for cache efficiency...")

        optimized_arrays = {}
        total_original_size = 0
        total_optimized_size = 0

        for array_name, array in arrays.items():
            if array_name in ['session_bounds', 'metadata']:
                # Skip special arrays
                optimized_arrays[array_name] = array
                continue

            print(f"   Optimizing {array_name}...")

            # Calculate original size
            original_size = array.nbytes
            total_original_size += original_size

            # Create cache-aligned, contiguous array
            optimized_array = self._create_aligned_array(array)

            # Verify optimization
            if not optimized_array.flags.c_contiguous:
                print(f"   ⚠️  Warning: {array_name} not contiguous after optimization")

            # Calculate optimized size
            optimized_size = optimized_array.nbytes
            total_optimized_size += optimized_size

            optimized_arrays[array_name] = optimized_array

            print(f"     Original: {original_size / 1024:.1f} KB")
            print(f"     Optimized: {optimized_size / 1024:.1f} KB")
            print(f"     Alignment: {self._check_alignment(optimized_array)} bytes")

        print(f"✅ Memory optimization complete")
        print(f"   Total original size: {total_original_size / 1024 / 1024:.1f} MB")
        print(f"   Total optimized size: {total_optimized_size / 1024 / 1024:.1f} MB")
        print(f"   Memory efficiency: {100 * total_optimized_size / total_original_size:.1f}%")

        return optimized_arrays

    def _create_aligned_array(self, array: np.ndarray) -> np.ndarray:
        """Create memory-aligned, contiguous array."""

        # Ensure C-contiguous
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)

        # Check if already aligned
        alignment = self._check_alignment(array)
        if alignment >= self.SIMD_ALIGNMENT:
            return array

        # Create aligned array
        dtype = array.dtype
        shape = array.shape

        # Calculate padding needed for alignment
        element_size = dtype.itemsize
        total_elements = array.size
        total_bytes = total_elements * element_size

        # Add padding for alignment
        padding_bytes = self.SIMD_ALIGNMENT - (total_bytes % self.SIMD_ALIGNMENT)
        if padding_bytes == self.SIMD_ALIGNMENT:
            padding_bytes = 0

        if padding_bytes > 0:
            padding_elements = padding_bytes // element_size
            total_padded_elements = total_elements + padding_elements

            # Create aligned array with padding
            aligned_array = np.empty(total_padded_elements, dtype=dtype)

            # Copy original data
            aligned_array[:total_elements] = array.ravel()

            # Reshape to original shape
            aligned_array = aligned_array[:total_elements].reshape(shape)
        else:
            aligned_array = array

        # Ensure contiguous
        return np.ascontiguousarray(aligned_array)

    def _check_alignment(self, array: np.ndarray) -> int:
        """Check memory alignment of array."""
        return array.ctypes.data % self.CACHE_LINE_SIZE

    def create_structure_of_arrays(self, ohlcv_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert to Structure of Arrays (SoA) for better cache performance.

        Instead of OHLCV interleaved, store each component separately.
        """
        print("📊 Creating Structure of Arrays layout...")

        # Determine the size
        n_bars = len(ohlcv_data['close'])

        # Create SoA layout - all prices together for locality
        soa_arrays = {}

        # Price arrays (most frequently accessed together)
        price_block = np.empty((4, n_bars), dtype=np.int32)  # OHLC
        price_block[0] = ohlcv_data['open']
        price_block[1] = ohlcv_data['high']
        price_block[2] = ohlcv_data['low']
        price_block[3] = ohlcv_data['close']

        # Make price block contiguous and aligned
        price_block = np.ascontiguousarray(price_block)

        soa_arrays['price_block'] = price_block
        soa_arrays['open'] = price_block[0]
        soa_arrays['high'] = price_block[1]
        soa_arrays['low'] = price_block[2]
        soa_arrays['close'] = price_block[3]

        # Other arrays
        for key in ['timestamps', 'volume']:
            if key in ohlcv_data:
                soa_arrays[key] = np.ascontiguousarray(ohlcv_data[key])

        # Copy over other arrays
        for key, array in ohlcv_data.items():
            if key not in ['open', 'high', 'low', 'close', 'timestamps', 'volume']:
                soa_arrays[key] = array

        print(f"✅ SoA layout created with {len(soa_arrays)} arrays")
        return soa_arrays

    def prefetch_data(self, arrays: Dict[str, np.ndarray], indices: np.ndarray) -> None:
        """
        Prefetch data into cache for upcoming operations.

        Args:
            arrays: Dictionary of arrays to prefetch
            indices: Array indices that will be accessed
        """
        # This is a hint to the system for data prefetching
        # In practice, modern CPUs do this automatically, but we can help
        for array_name, array in arrays.items():
            if len(array.shape) == 1 and array_name not in ['session_bounds', 'metadata']:
                # Touch the memory locations we'll need
                for idx in indices[:min(100, len(indices))]:  # Prefetch first 100
                    if idx < len(array):
                        _ = array[idx]  # Memory access to trigger prefetch


def benchmark_memory_layout(original_arrays: Dict[str, np.ndarray],
                          optimized_arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Benchmark memory layout optimization."""

    print("🏁 Benchmarking Memory Layout Optimization")
    print("=" * 50)

    n_operations = 100000
    n_bars = min(50000, len(original_arrays['close']))

    # Create random indices for testing
    test_indices = np.random.randint(0, n_bars - 1000, n_operations)

    # Benchmark original layout
    print("📊 Testing original memory layout...")
    start_time = time.time()

    original_sum = benchmark_memory_access_numba(
        original_arrays['open'], original_arrays['high'],
        original_arrays['low'], original_arrays['close'],
        test_indices
    )

    original_time = time.time() - start_time
    print(f"   Original time: {original_time:.3f}s")

    # Benchmark optimized layout
    print("⚡ Testing optimized memory layout...")
    start_time = time.time()

    optimized_sum = benchmark_memory_access_numba(
        optimized_arrays['open'], optimized_arrays['high'],
        optimized_arrays['low'], optimized_arrays['close'],
        test_indices
    )

    optimized_time = time.time() - start_time
    print(f"   Optimized time: {optimized_time:.3f}s")

    # Verify results match
    results_match = abs(original_sum - optimized_sum) < 1e-6

    speedup = original_time / optimized_time if optimized_time > 0 else 1.0

    print(f"\n✨ Memory Layout Results:")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   Results match: {'✅' if results_match else '❌'}")
    print(f"   Memory access rate: {n_operations / optimized_time:,.0f} ops/sec")

    return {
        'original_time': original_time,
        'optimized_time': optimized_time,
        'speedup': speedup,
        'results_match': results_match,
        'access_rate': n_operations / optimized_time
    }


@njit(cache=True, fastmath=True)
def benchmark_memory_access_numba(open_arr: np.ndarray, high_arr: np.ndarray,
                                 low_arr: np.ndarray, close_arr: np.ndarray,
                                 indices: np.ndarray) -> float:
    """Benchmark memory access patterns with Numba."""
    total = 0.0

    for i in range(len(indices)):
        idx = indices[i]
        # Simulate OHLC processing
        ohlc_avg = (open_arr[idx] + high_arr[idx] + low_arr[idx] + close_arr[idx]) / 4
        total += ohlc_avg

        # Simulate some computation with neighboring values
        if idx > 0:
            prev_close = close_arr[idx - 1]
            change = close_arr[idx] - prev_close
            total += change * 0.1

    return total


if __name__ == "__main__":
    # Test memory optimization
    from .data_loader import UltraFastDataLoader
    from .precompute import PrecomputeEngine
    from .numba_engine import NumbaBacktestEngine

    test_path = "data_parquet/nq/1min/year=2024"

    if Path(test_path).exists():
        print("🚀 Testing Memory Layout Optimization")

        # Load data
        loader = UltraFastDataLoader()
        memmap_files = loader.parquet_to_memmap(test_path)
        arrays = loader.load_memmap_arrays(memmap_files)

        precompute = PrecomputeEngine()
        precomputed_files = precompute.create_precomputed_data(arrays, "memory_test")
        precomputed = precompute.load_precomputed_data(precomputed_files)

        engine = NumbaBacktestEngine()
        numba_arrays = engine.prepare_numba_arrays(arrays, precomputed)

        # Optimize memory layout
        optimizer = MemoryOptimizer()
        optimized_arrays = optimizer.optimize_array_layout(numba_arrays)

        # Benchmark improvement
        results = benchmark_memory_layout(numba_arrays, optimized_arrays)
        print(f"\n📊 Task 10 Complete: {results['speedup']:.2f}x memory access improvement!")
    else:
        print(f"❌ Test data not found: {test_path}")