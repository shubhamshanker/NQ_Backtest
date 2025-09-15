"""
Parallel Strategy Execution and Batch Processing
===============================================
Task 5: Batch processing for large datasets
Task 6: Parallel strategy execution using Numba prange
"""

import numpy as np
import time
from typing import Dict, List, Tuple
from numba import njit, prange
from pathlib import Path
import json


class ParallelExecutor:
    """Execute multiple strategies in parallel with batch processing."""

    def __init__(self):
        self.CHUNK_SIZE = 50000  # Default chunk size for batch processing

    def batch_process_strategies(self, numba_arrays: Dict[str, np.ndarray],
                               strategies: List[Dict],
                               chunk_size: int = None) -> Dict[str, Dict]:
        """
        Process multiple strategies in batches for memory efficiency.

        Args:
            numba_arrays: Prepared Numba-ready arrays
            strategies: List of strategy configurations
            chunk_size: Size of data chunks to process

        Returns:
            Dictionary with results for each strategy
        """
        if chunk_size is None:
            chunk_size = self.CHUNK_SIZE

        print(f"🔄 Batch processing {len(strategies)} strategies with chunks of {chunk_size:,} bars")

        n_bars = len(numba_arrays['close'])
        n_chunks = (n_bars + chunk_size - 1) // chunk_size

        all_results = {}

        for i, strategy in enumerate(strategies):
            strategy_name = strategy.get('name', f'Strategy_{i+1}')
            print(f"   Processing {strategy_name}...")

            # Process strategy in chunks
            chunk_results = []
            total_trades = 0

            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, n_bars)

                # Run strategy on chunk
                from .numba_engine import NumbaBacktestEngine
                engine = NumbaBacktestEngine()
                chunk_result = engine.run_strategy(numba_arrays, strategy, start_idx, end_idx)

                chunk_results.append(chunk_result)
                total_trades += chunk_result['total_trades']

            # Aggregate chunk results
            aggregated_result = self._aggregate_chunk_results(chunk_results, strategy_name)
            all_results[strategy_name] = aggregated_result

            print(f"     ✅ {strategy_name}: {total_trades} trades across {n_chunks} chunks")

        return all_results

    def parallel_strategy_execution(self, numba_arrays: Dict[str, np.ndarray],
                                  strategies: List[Dict]) -> Dict[str, Dict]:
        """
        Execute multiple strategies in parallel using Numba prange.

        Args:
            numba_arrays: Prepared Numba-ready arrays
            strategies: List of strategy configurations

        Returns:
            Results for all strategies
        """
        print(f"⚡ Executing {len(strategies)} strategies in parallel")

        # Prepare strategy parameters for Numba
        strategy_params = self._prepare_strategy_params(strategies)

        # Extract arrays for Numba
        open_prices = numba_arrays['open']
        high_prices = numba_arrays['high']
        low_prices = numba_arrays['low']
        close_prices = numba_arrays['close']
        cumsum_close = numba_arrays['cumsum_close']
        session_bounds = numba_arrays['session_bounds']

        # Execute strategies in parallel
        results = parallel_strategy_runner_numba(
            open_prices, high_prices, low_prices, close_prices,
            cumsum_close, session_bounds, strategy_params
        )

        # Process and format results
        formatted_results = {}
        for i, strategy in enumerate(strategies):
            strategy_name = strategy.get('name', f'Strategy_{i+1}')
            raw_result = {
                'trades': results[0][i],  # trades array
                'equity_curve': results[1][i],  # equity curves
                'trade_count': results[2][i]  # trade counts
            }

            from .numba_engine import NumbaBacktestEngine
            engine = NumbaBacktestEngine()
            formatted_result = engine._process_results(raw_result, strategy)
            formatted_results[strategy_name] = formatted_result

            print(f"   ✅ {strategy_name}: {formatted_result['total_trades']} trades")

        return formatted_results

    def _prepare_strategy_params(self, strategies: List[Dict]) -> np.ndarray:
        """
        Convert strategy list to Numba-compatible parameter arrays.

        Returns:
            Structured array with strategy parameters
        """
        n_strategies = len(strategies)

        # Create structured array for strategy parameters
        # Format: [strategy_type, ma_fast, ma_slow, orb_minutes, stop_loss, profit_target]
        params = np.zeros((n_strategies, 6), dtype=np.int32)

        for i, strategy in enumerate(strategies):
            strategy_type = strategy.get('type', 'simple_ma')

            if strategy_type == 'simple_ma':
                params[i, 0] = 0  # Type: Simple MA
                params[i, 1] = strategy.get('ma_fast', 10)
                params[i, 2] = strategy.get('ma_slow', 20)
            elif strategy_type == 'orb_breakout':
                params[i, 0] = 1  # Type: ORB
                params[i, 3] = strategy.get('orb_minutes', 30)
                params[i, 4] = strategy.get('stop_loss', 10)
                params[i, 5] = strategy.get('profit_target', 20)

        return params

    def _aggregate_chunk_results(self, chunk_results: List[Dict], strategy_name: str) -> Dict:
        """Aggregate results from multiple chunks."""

        if not chunk_results:
            return self._empty_result()

        # Combine trades from all chunks
        all_trades = []
        for result in chunk_results:
            if result['total_trades'] > 0:
                all_trades.extend(result['trades'])

        # Calculate aggregated statistics
        if not all_trades:
            return self._empty_result()

        # Convert to arrays for calculations
        trades_array = np.array(all_trades)
        pnl_points = trades_array[:, 3] / 100000  # Assuming PRICE_MULTIPLIER = 100000

        total_pnl_points = pnl_points.sum()
        winning_trades = (pnl_points > 0).sum()
        losing_trades = (pnl_points < 0).sum()
        total_trades = len(pnl_points)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        return {
            'strategy_name': strategy_name,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_percent': win_rate * 100,
            'total_pnl_points': total_pnl_points,
            'total_pnl_dollars': total_pnl_points * 20,  # NQ multiplier
            'trades': trades_array
        }

    def _empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate_percent': 0.0,
            'total_pnl_points': 0.0,
            'total_pnl_dollars': 0.0,
            'trades': np.array([])
        }


@njit(cache=True, fastmath=True)
def sma_instant_parallel(cumsum: np.ndarray, period: int, index: int) -> int:
    """Fast SMA calculation for parallel execution."""
    if index < period - 1:
        return cumsum[index] // (index + 1)
    return (cumsum[index] - cumsum[index - period]) // period


@njit(cache=True, fastmath=True, parallel=True)
def parallel_strategy_runner_numba(open_prices: np.ndarray, high_prices: np.ndarray,
                                  low_prices: np.ndarray, close_prices: np.ndarray,
                                  cumsum_close: np.ndarray, session_bounds: np.ndarray,
                                  strategy_params: np.ndarray):
    """
    Execute multiple strategies in parallel using Numba prange.

    Args:
        strategy_params: Array of shape (n_strategies, 6) with strategy parameters

    Returns:
        Tuple of (all_trades, all_equity_curves, all_trade_counts)
    """
    n_strategies = strategy_params.shape[0]
    n_bars = len(close_prices)
    max_trades_per_strategy = n_bars // 5  # Conservative estimate

    # Pre-allocate output arrays
    all_trades = np.zeros((n_strategies, max_trades_per_strategy, 6), dtype=np.int32)
    all_equity_curves = np.zeros((n_strategies, n_bars), dtype=np.int64)
    all_trade_counts = np.zeros(n_strategies, dtype=np.int32)

    # Execute strategies in parallel
    for strategy_idx in prange(n_strategies):
        params = strategy_params[strategy_idx]
        strategy_type = params[0]

        if strategy_type == 0:  # Simple MA
            ma_fast = params[1]
            ma_slow = params[2]

            trades, equity_curve, trade_count = run_simple_ma_parallel(
                open_prices, close_prices, cumsum_close, ma_fast, ma_slow
            )

        elif strategy_type == 1:  # ORB
            orb_minutes = params[3]
            stop_loss_points = params[4]
            profit_target_points = params[5]

            trades, equity_curve, trade_count = run_orb_parallel(
                open_prices, high_prices, low_prices, close_prices,
                session_bounds, orb_minutes, stop_loss_points, profit_target_points
            )

        else:
            # Default to empty results
            trades = np.zeros((0, 6), dtype=np.int32)
            equity_curve = np.zeros(n_bars, dtype=np.int64)
            trade_count = 0

        # Store results
        n_trades = min(trade_count, max_trades_per_strategy)
        all_trades[strategy_idx, :n_trades] = trades[:n_trades]
        all_equity_curves[strategy_idx] = equity_curve
        all_trade_counts[strategy_idx] = trade_count

    return all_trades, all_equity_curves, all_trade_counts


@njit(cache=True, fastmath=True)
def run_simple_ma_parallel(open_prices: np.ndarray, close_prices: np.ndarray,
                          cumsum_close: np.ndarray, ma_fast: int, ma_slow: int):
    """Simple MA strategy optimized for parallel execution."""
    n_bars = len(close_prices)
    max_trades = n_bars // 10

    trades = np.zeros((max_trades, 6), dtype=np.int32)
    equity_curve = np.zeros(n_bars, dtype=np.int64)

    trade_count = 0
    position = 0
    entry_price = 0
    running_pnl = 0

    start_idx = max(ma_fast, ma_slow)

    for i in range(start_idx, n_bars - 1):
        # Calculate MAs
        ma_fast_val = sma_instant_parallel(cumsum_close, ma_fast, i)
        ma_slow_val = sma_instant_parallel(cumsum_close, ma_slow, i)

        # Previous MAs
        ma_fast_prev = sma_instant_parallel(cumsum_close, ma_fast, i - 1)
        ma_slow_prev = sma_instant_parallel(cumsum_close, ma_slow, i - 1)

        # Entry logic
        if position == 0:
            if ma_fast_prev <= ma_slow_prev and ma_fast_val > ma_slow_val:
                position = 1
                entry_price = open_prices[i + 1]
            elif ma_fast_prev >= ma_slow_prev and ma_fast_val < ma_slow_val:
                position = -1
                entry_price = open_prices[i + 1]

        # Exit logic
        elif position != 0:
            exit_signal = False
            if position == 1 and ma_fast_val < ma_slow_val:
                exit_signal = True
            elif position == -1 and ma_fast_val > ma_slow_val:
                exit_signal = True

            if exit_signal and trade_count < max_trades:
                exit_price = open_prices[i + 1]
                pnl = (exit_price - entry_price) * position

                trades[trade_count, 0] = i - 1  # entry_idx
                trades[trade_count, 1] = i + 1  # exit_idx
                trades[trade_count, 2] = position
                trades[trade_count, 3] = pnl
                trades[trade_count, 4] = entry_price
                trades[trade_count, 5] = exit_price

                running_pnl += pnl
                trade_count += 1
                position = 0

        equity_curve[i] = running_pnl

    return trades[:trade_count].copy(), equity_curve, trade_count


@njit(cache=True, fastmath=True)
def run_orb_parallel(open_prices: np.ndarray, high_prices: np.ndarray,
                    low_prices: np.ndarray, close_prices: np.ndarray,
                    session_bounds: np.ndarray, orb_minutes: int,
                    stop_loss_points: int, profit_target_points: int):
    """ORB strategy optimized for parallel execution."""
    PRICE_MULTIPLIER = 100000
    stop_loss_int = stop_loss_points * PRICE_MULTIPLIER
    profit_target_int = profit_target_points * PRICE_MULTIPLIER

    n_bars = len(close_prices)
    max_trades = len(session_bounds)

    trades = np.zeros((max_trades, 6), dtype=np.int32)
    equity_curve = np.zeros(n_bars, dtype=np.int64)

    trade_count = 0
    running_pnl = 0

    # Process each session
    for session_idx in range(len(session_bounds)):
        session_start = session_bounds[session_idx, 0]
        session_end = session_bounds[session_idx, 1]

        # Calculate ORB range
        orb_end_idx = min(session_start + orb_minutes, session_end)
        if orb_end_idx <= session_start:
            continue

        # Find ORB high/low
        orb_high = high_prices[session_start]
        orb_low = low_prices[session_start]

        for j in range(session_start, orb_end_idx):
            if high_prices[j] > orb_high:
                orb_high = high_prices[j]
            if low_prices[j] < orb_low:
                orb_low = low_prices[j]

        # Look for breakouts
        for i in range(orb_end_idx, session_end - 1):
            if trade_count >= max_trades:
                break

            # Long breakout
            if high_prices[i] > orb_high:
                entry_price = orb_high
                stop_price = entry_price - stop_loss_int
                target_price = entry_price + profit_target_int

                exit_idx, exit_price = find_exit_parallel(
                    i + 1, session_end, open_prices, high_prices, low_prices,
                    stop_price, target_price, 1
                )

                if exit_idx > 0:
                    pnl = exit_price - entry_price
                    trades[trade_count, 0] = i
                    trades[trade_count, 1] = exit_idx
                    trades[trade_count, 2] = 1
                    trades[trade_count, 3] = pnl
                    trades[trade_count, 4] = entry_price
                    trades[trade_count, 5] = exit_price

                    running_pnl += pnl
                    trade_count += 1
                    break

            # Short breakout
            elif low_prices[i] < orb_low:
                entry_price = orb_low
                stop_price = entry_price + stop_loss_int
                target_price = entry_price - profit_target_int

                exit_idx, exit_price = find_exit_parallel(
                    i + 1, session_end, open_prices, high_prices, low_prices,
                    stop_price, target_price, -1
                )

                if exit_idx > 0:
                    pnl = entry_price - exit_price
                    trades[trade_count, 0] = i
                    trades[trade_count, 1] = exit_idx
                    trades[trade_count, 2] = -1
                    trades[trade_count, 3] = pnl
                    trades[trade_count, 4] = entry_price
                    trades[trade_count, 5] = exit_price

                    running_pnl += pnl
                    trade_count += 1
                    break

    # Update equity curve
    for i in range(n_bars):
        equity_curve[i] = running_pnl

    return trades[:trade_count].copy(), equity_curve, trade_count


@njit(cache=True, fastmath=True)
def find_exit_parallel(start_idx: int, end_idx: int, open_prices: np.ndarray,
                      high_prices: np.ndarray, low_prices: np.ndarray,
                      stop_price: int, target_price: int, direction: int) -> tuple:
    """Find exit point optimized for parallel execution."""
    for i in range(start_idx, end_idx):
        if direction == 1:  # Long
            if low_prices[i] <= stop_price:
                return i, stop_price
            elif high_prices[i] >= target_price:
                return i, target_price
        else:  # Short
            if high_prices[i] >= stop_price:
                return i, stop_price
            elif low_prices[i] <= target_price:
                return i, target_price

    return end_idx - 1, open_prices[end_idx - 1]


def benchmark_parallel_performance(numba_arrays: Dict[str, np.ndarray],
                                  n_strategies: int = 10) -> Dict[str, float]:
    """Benchmark parallel vs sequential strategy execution."""

    print(f"🏁 Benchmarking Parallel Execution ({n_strategies} strategies)")
    print("=" * 55)

    # Create test strategies
    strategies = []
    for i in range(n_strategies):
        if i % 2 == 0:
            strategies.append({
                'name': f'MA_Strategy_{i+1}',
                'type': 'simple_ma',
                'ma_fast': 5 + i,
                'ma_slow': 20 + i * 2
            })
        else:
            strategies.append({
                'name': f'ORB_Strategy_{i+1}',
                'type': 'orb_breakout',
                'orb_minutes': 20 + i * 5,
                'stop_loss': 8 + i,
                'profit_target': 15 + i * 2
            })

    executor = ParallelExecutor()

    # Benchmark sequential execution
    print(f"📊 Sequential execution...")
    start_time = time.time()
    from .numba_engine import NumbaBacktestEngine
    engine = NumbaBacktestEngine()

    sequential_results = {}
    for strategy in strategies:
        result = engine.run_strategy(numba_arrays, strategy, 0, 10000)  # Test subset
        sequential_results[strategy['name']] = result

    sequential_time = time.time() - start_time
    print(f"   Sequential time: {sequential_time:.3f}s")

    # Benchmark parallel execution
    print(f"⚡ Parallel execution...")
    start_time = time.time()

    # Use subset of data for fair comparison
    subset_arrays = {}
    for key, array in numba_arrays.items():
        if key in ['session_bounds', 'metadata']:
            subset_arrays[key] = array
        else:
            subset_arrays[key] = array[:10000] if len(array.shape) == 1 else array

    parallel_results = executor.parallel_strategy_execution(subset_arrays, strategies)

    parallel_time = time.time() - start_time
    print(f"   Parallel time: {parallel_time:.3f}s")

    speedup = sequential_time / parallel_time if parallel_time > 0 else float('inf')

    print(f"\n✨ Parallel Execution Results:")
    print(f"   Sequential: {sequential_time:.3f}s")
    print(f"   Parallel: {parallel_time:.3f}s")
    print(f"   Speedup: {speedup:.1f}x faster")
    print(f"   Strategies processed: {len(strategies)}")

    return {
        'sequential_time': sequential_time,
        'parallel_time': parallel_time,
        'speedup': speedup,
        'n_strategies': n_strategies
    }


if __name__ == "__main__":
    # Test parallel executor
    from .data_loader import UltraFastDataLoader
    from .precompute import PrecomputeEngine
    from .numba_engine import NumbaBacktestEngine

    test_path = "data_parquet/nq/1min/year=2024"

    if Path(test_path).exists():
        print("🚀 Testing Parallel Executor")

        # Load data
        loader = UltraFastDataLoader()
        memmap_files = loader.parquet_to_memmap(test_path)
        arrays = loader.load_memmap_arrays(memmap_files)

        precompute = PrecomputeEngine()
        precomputed_files = precompute.create_precomputed_data(arrays, "parallel_test")
        precomputed = precompute.load_precomputed_data(precomputed_files)

        engine = NumbaBacktestEngine()
        numba_arrays = engine.prepare_numba_arrays(arrays, precomputed)

        # Run benchmark
        results = benchmark_parallel_performance(numba_arrays, 8)
        print(f"\n📊 Tasks 5 & 6 Complete: {results['speedup']:.1f}x speedup with parallel execution!")
    else:
        print(f"❌ Test data not found: {test_path}")