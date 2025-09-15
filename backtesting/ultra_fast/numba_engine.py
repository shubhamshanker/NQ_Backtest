"""
Numba-Compatible Data Loader and Core Backtest Engine
====================================================
Task 3: Create Numba-ready data structures and backtest functions
Task 4: Implement basic Numba JIT-compiled backtest engine
"""

import numpy as np
import time
from typing import Dict, Tuple, List
from numba import njit, prange
from numba.types import int32, int64, float64, boolean, uint32


class NumbaBacktestEngine:
    """Ultra-fast backtesting engine using Numba JIT compilation."""

    def __init__(self):
        self.PRICE_MULTIPLIER = 100000
        self.POINT_VALUE = 20  # NQ futures: $20 per point

    def prepare_numba_arrays(self, arrays: Dict[str, np.ndarray],
                            precomputed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Prepare all arrays for Numba compatibility.

        Args:
            arrays: Memory-mapped arrays from data_loader
            precomputed: Precomputed data from precompute engine

        Returns:
            Dictionary of Numba-ready arrays
        """
        # Ensure all arrays are C-contiguous and correct dtype
        numba_arrays = {
            # Price data (int32)
            'timestamps': np.ascontiguousarray(arrays['timestamps'], dtype=np.uint32),
            'open': np.ascontiguousarray(arrays['open'], dtype=np.int32),
            'high': np.ascontiguousarray(arrays['high'], dtype=np.int32),
            'low': np.ascontiguousarray(arrays['low'], dtype=np.int32),
            'close': np.ascontiguousarray(arrays['close'], dtype=np.int32),

            # Precomputed data (int64)
            'cumsum_close': np.ascontiguousarray(precomputed['cumsum_close'], dtype=np.int64),
            'cumsum_high': np.ascontiguousarray(precomputed['cumsum_high'], dtype=np.int64),
            'cumsum_low': np.ascontiguousarray(precomputed['cumsum_low'], dtype=np.int64),

            # Session boundaries
            'session_bounds': np.ascontiguousarray(precomputed['session_bounds'], dtype=np.int32),
        }

        # Add volume if available
        if 'volume' in arrays:
            numba_arrays['volume'] = np.ascontiguousarray(arrays['volume'], dtype=np.uint32)

        if 'cumsum_volume' in precomputed:
            numba_arrays['cumsum_volume'] = np.ascontiguousarray(precomputed['cumsum_volume'], dtype=np.int64)

        return numba_arrays

    def run_strategy(self, numba_arrays: Dict[str, np.ndarray],
                    strategy_params: Dict,
                    start_idx: int = 0,
                    end_idx: int = -1) -> Dict:
        """
        Run a strategy using Numba-optimized functions.

        Args:
            numba_arrays: Prepared Numba arrays
            strategy_params: Strategy configuration
            start_idx: Start index for backtest
            end_idx: End index for backtest (-1 for all data)

        Returns:
            Backtest results
        """
        if end_idx == -1:
            end_idx = len(numba_arrays['close'])

        # Extract arrays for Numba functions
        timestamps = numba_arrays['timestamps']
        open_prices = numba_arrays['open']
        high_prices = numba_arrays['high']
        low_prices = numba_arrays['low']
        close_prices = numba_arrays['close']
        cumsum_close = numba_arrays['cumsum_close']
        session_bounds = numba_arrays['session_bounds']

        # Run the strategy based on type
        strategy_type = strategy_params.get('type', 'simple_ma')

        if strategy_type == 'simple_ma':
            results = self._run_simple_ma_strategy(
                open_prices, high_prices, low_prices, close_prices,
                cumsum_close, session_bounds, timestamps,
                strategy_params, start_idx, end_idx
            )
        elif strategy_type == 'orb_breakout':
            results = self._run_orb_strategy(
                open_prices, high_prices, low_prices, close_prices,
                cumsum_close, session_bounds, timestamps,
                strategy_params, start_idx, end_idx
            )
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        # Convert results back to float and calculate statistics
        return self._process_results(results, strategy_params)

    def _run_simple_ma_strategy(self, open_prices, high_prices, low_prices, close_prices,
                               cumsum_close, session_bounds, timestamps,
                               strategy_params, start_idx, end_idx):
        """Run simple moving average crossover strategy."""

        ma_fast = strategy_params.get('ma_fast', 10)
        ma_slow = strategy_params.get('ma_slow', 20)

        trades, equity_curve, trade_count = run_simple_ma_numba(
            open_prices, close_prices, cumsum_close,
            ma_fast, ma_slow, start_idx, end_idx
        )

        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'trade_count': trade_count
        }

    def _run_orb_strategy(self, open_prices, high_prices, low_prices, close_prices,
                         cumsum_close, session_bounds, timestamps,
                         strategy_params, start_idx, end_idx):
        """Run Opening Range Breakout strategy."""

        orb_minutes = strategy_params.get('orb_minutes', 30)
        stop_loss_points = strategy_params.get('stop_loss', 10)
        profit_target_points = strategy_params.get('profit_target', 20)

        trades, equity_curve, trade_count = run_orb_strategy_numba(
            open_prices, high_prices, low_prices, close_prices,
            session_bounds, timestamps, orb_minutes,
            stop_loss_points, profit_target_points,
            start_idx, end_idx
        )

        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'trade_count': trade_count
        }

    def _process_results(self, raw_results: Dict, strategy_params: Dict) -> Dict:
        """Process raw Numba results into final statistics."""

        trades = raw_results['trades']
        equity_curve = raw_results['equity_curve']

        if len(trades) == 0:
            return self._empty_results()

        # Convert to float values
        pnl_points = trades[:, 3] / self.PRICE_MULTIPLIER  # Column 3 is PnL
        pnl_dollars = pnl_points * self.POINT_VALUE

        # Calculate statistics
        total_pnl_dollars = pnl_dollars.sum()
        total_pnl_points = pnl_points.sum()

        winning_trades = pnl_dollars > 0
        losing_trades = pnl_dollars < 0

        win_count = winning_trades.sum()
        loss_count = losing_trades.sum()
        total_trades = len(trades)

        win_rate = win_count / total_trades if total_trades > 0 else 0

        avg_win = pnl_dollars[winning_trades].mean() if win_count > 0 else 0
        avg_loss = pnl_dollars[losing_trades].mean() if loss_count > 0 else 0

        profit_factor = abs(pnl_dollars[winning_trades].sum() / pnl_dollars[losing_trades].sum()) if loss_count > 0 else float('inf')

        # Calculate drawdown
        equity_float = equity_curve / self.PRICE_MULTIPLIER * self.POINT_VALUE
        running_max = np.maximum.accumulate(equity_float)
        drawdown = running_max - equity_float
        max_drawdown = drawdown.max()

        return {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate_percent': win_rate * 100,
            'total_pnl_points': total_pnl_points,
            'total_pnl_dollars': total_pnl_dollars,
            'avg_win_points': avg_win / self.POINT_VALUE if avg_win != 0 else 0,
            'avg_loss_points': avg_loss / self.POINT_VALUE if avg_loss != 0 else 0,
            'avg_win_dollars': avg_win,
            'avg_loss_dollars': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_dollars': max_drawdown,
            'max_drawdown_points': max_drawdown / self.POINT_VALUE,
            'equity_curve': equity_float,
            'trades': trades
        }

    def _empty_results(self) -> Dict:
        """Return empty results when no trades are generated."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate_percent': 0.0,
            'total_pnl_points': 0.0,
            'total_pnl_dollars': 0.0,
            'avg_win_points': 0.0,
            'avg_loss_points': 0.0,
            'avg_win_dollars': 0.0,
            'avg_loss_dollars': 0.0,
            'profit_factor': 0.0,
            'max_drawdown_dollars': 0.0,
            'max_drawdown_points': 0.0,
            'equity_curve': np.array([]),
            'trades': np.array([])
        }


@njit(cache=True, fastmath=True)
def sma_instant_numba(cumsum: np.ndarray, period: int, index: int) -> int:
    """Fast SMA calculation using precomputed cumsum."""
    if index < period - 1:
        return cumsum[index] // (index + 1)
    return (cumsum[index] - cumsum[index - period]) // period


@njit(cache=True, fastmath=True)
def run_simple_ma_numba(open_prices: np.ndarray, close_prices: np.ndarray,
                       cumsum_close: np.ndarray, ma_fast: int, ma_slow: int,
                       start_idx: int, end_idx: int):
    """
    Simple moving average crossover strategy (Numba-compiled).

    Returns:
        Dict with trades array and equity curve
    """
    n_bars = end_idx - start_idx
    max_trades = n_bars // 10  # Estimate max possible trades

    # Pre-allocate arrays
    trades = np.zeros((max_trades, 6), dtype=np.int32)  # entry_idx, exit_idx, direction, pnl, entry_price, exit_price
    equity_curve = np.zeros(n_bars, dtype=np.int64)

    trade_count = 0
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0
    entry_idx = 0
    running_pnl = 0

    for i in range(start_idx + max(ma_fast, ma_slow), end_idx - 1):
        # Calculate moving averages
        ma_fast_val = sma_instant_numba(cumsum_close, ma_fast, i)
        ma_slow_val = sma_instant_numba(cumsum_close, ma_slow, i)

        # Previous MAs for crossover detection
        ma_fast_prev = sma_instant_numba(cumsum_close, ma_fast, i - 1)
        ma_slow_prev = sma_instant_numba(cumsum_close, ma_slow, i - 1)

        # Entry signals
        if position == 0:  # No position
            if ma_fast_prev <= ma_slow_prev and ma_fast_val > ma_slow_val:  # Bullish crossover
                position = 1
                entry_price = open_prices[i + 1]  # Enter at next bar open
                entry_idx = i + 1
            elif ma_fast_prev >= ma_slow_prev and ma_fast_val < ma_slow_val:  # Bearish crossover
                position = -1
                entry_price = open_prices[i + 1]
                entry_idx = i + 1

        # Exit signals
        elif position != 0:
            exit_signal = False

            if position == 1 and ma_fast_val < ma_slow_val:  # Exit long
                exit_signal = True
            elif position == -1 and ma_fast_val > ma_slow_val:  # Exit short
                exit_signal = True

            if exit_signal and trade_count < max_trades:
                exit_price = open_prices[i + 1]  # Exit at next bar open
                pnl = (exit_price - entry_price) * position

                # Store trade
                trades[trade_count, 0] = entry_idx
                trades[trade_count, 1] = i + 1
                trades[trade_count, 2] = position
                trades[trade_count, 3] = pnl
                trades[trade_count, 4] = entry_price
                trades[trade_count, 5] = exit_price

                running_pnl += pnl
                trade_count += 1
                position = 0

        # Update equity curve
        if i - start_idx < n_bars:
            equity_curve[i - start_idx] = running_pnl

    # Return only the filled portion of trades array
    return trades[:trade_count].copy(), equity_curve, trade_count


@njit(cache=True, fastmath=True)
def run_orb_strategy_numba(open_prices: np.ndarray, high_prices: np.ndarray,
                          low_prices: np.ndarray, close_prices: np.ndarray,
                          session_bounds: np.ndarray, timestamps: np.ndarray,
                          orb_minutes: int, stop_loss_points: int,
                          profit_target_points: int, start_idx: int, end_idx: int):
    """
    Opening Range Breakout strategy (Numba-compiled).

    Args:
        orb_minutes: Opening range period in minutes
        stop_loss_points: Stop loss in points (multiply by PRICE_MULTIPLIER)
        profit_target_points: Profit target in points
    """
    PRICE_MULTIPLIER = 100000
    stop_loss_int = stop_loss_points * PRICE_MULTIPLIER
    profit_target_int = profit_target_points * PRICE_MULTIPLIER

    n_bars = end_idx - start_idx
    max_trades = len(session_bounds) * 2  # Max 2 trades per session

    # Pre-allocate arrays
    trades = np.zeros((max_trades, 6), dtype=np.int32)
    equity_curve = np.zeros(n_bars, dtype=np.int64)

    trade_count = 0
    running_pnl = 0

    # Process each trading session
    for session_idx in range(len(session_bounds)):
        session_start = session_bounds[session_idx, 0]
        session_end = session_bounds[session_idx, 1]

        if session_start < start_idx or session_end > end_idx:
            continue

        # Calculate ORB range (first orb_minutes bars of session)
        orb_end_idx = min(session_start + orb_minutes, session_end)

        if orb_end_idx <= session_start:
            continue

        # Find ORB high and low
        orb_high = high_prices[session_start]
        orb_low = low_prices[session_start]

        for j in range(session_start, orb_end_idx):
            if high_prices[j] > orb_high:
                orb_high = high_prices[j]
            if low_prices[j] < orb_low:
                orb_low = low_prices[j]

        # Look for breakouts after ORB period
        for i in range(orb_end_idx, session_end - 1):
            if trade_count >= max_trades:
                break

            # Long breakout
            if high_prices[i] > orb_high:
                entry_price = orb_high
                stop_price = entry_price - stop_loss_int
                target_price = entry_price + profit_target_int

                # Find exit
                exit_idx, exit_price = find_exit_numba(
                    i + 1, session_end, open_prices, high_prices, low_prices,
                    stop_price, target_price, 1
                )

                if exit_idx > 0:
                    pnl = exit_price - entry_price

                    trades[trade_count, 0] = i
                    trades[trade_count, 1] = exit_idx
                    trades[trade_count, 2] = 1  # Long
                    trades[trade_count, 3] = pnl
                    trades[trade_count, 4] = entry_price
                    trades[trade_count, 5] = exit_price

                    running_pnl += pnl
                    trade_count += 1
                    break  # One trade per session

            # Short breakout
            elif low_prices[i] < orb_low:
                entry_price = orb_low
                stop_price = entry_price + stop_loss_int
                target_price = entry_price - profit_target_int

                # Find exit
                exit_idx, exit_price = find_exit_numba(
                    i + 1, session_end, open_prices, high_prices, low_prices,
                    stop_price, target_price, -1
                )

                if exit_idx > 0:
                    pnl = entry_price - exit_price

                    trades[trade_count, 0] = i
                    trades[trade_count, 1] = exit_idx
                    trades[trade_count, 2] = -1  # Short
                    trades[trade_count, 3] = pnl
                    trades[trade_count, 4] = entry_price
                    trades[trade_count, 5] = exit_price

                    running_pnl += pnl
                    trade_count += 1
                    break  # One trade per session

    # Update equity curve
    for i in range(n_bars):
        equity_curve[i] = running_pnl

    return trades[:trade_count].copy(), equity_curve, trade_count


@njit(cache=True, fastmath=True)
def find_exit_numba(start_idx: int, end_idx: int, open_prices: np.ndarray,
                   high_prices: np.ndarray, low_prices: np.ndarray,
                   stop_price: int, target_price: int, direction: int) -> tuple:
    """
    Find exit point for a trade (stop loss or profit target).

    Returns:
        (exit_idx, exit_price)
    """
    for i in range(start_idx, end_idx):
        if direction == 1:  # Long position
            if low_prices[i] <= stop_price:
                return i, stop_price
            elif high_prices[i] >= target_price:
                return i, target_price
        else:  # Short position
            if high_prices[i] >= stop_price:
                return i, stop_price
            elif low_prices[i] <= target_price:
                return i, target_price

    # Exit at session end if no stop/target hit
    return end_idx - 1, open_prices[end_idx - 1]


def benchmark_numba_speed(numba_arrays: Dict[str, np.ndarray],
                         n_runs: int = 100) -> Dict[str, float]:
    """Benchmark Numba backtest engine speed."""

    print("🏁 Benchmarking Numba Engine Speed")
    print("=" * 40)

    engine = NumbaBacktestEngine()

    # Simple strategy params
    strategy_params = {
        'type': 'simple_ma',
        'ma_fast': 10,
        'ma_slow': 20
    }

    n_bars = len(numba_arrays['close'])
    test_size = min(10000, n_bars)  # Test on subset for speed

    print(f"📊 Testing on {test_size:,} bars with {n_runs} runs")

    # Warm up Numba compilation
    print("🔥 Warming up Numba compilation...")
    engine.run_strategy(numba_arrays, strategy_params, 0, test_size)

    # Benchmark
    print("⚡ Benchmarking compiled code...")
    start_time = time.time()

    for _ in range(n_runs):
        results = engine.run_strategy(numba_arrays, strategy_params, 0, test_size)

    benchmark_time = time.time() - start_time
    avg_time_per_run = benchmark_time / n_runs
    bars_per_second = test_size / avg_time_per_run

    print(f"✨ Numba Engine Results:")
    print(f"   Average time per backtest: {avg_time_per_run:.4f}s")
    print(f"   Bars processed per second: {bars_per_second:,.0f}")
    print(f"   Total benchmark time: {benchmark_time:.2f}s")
    print(f"   Sample trades generated: {results['total_trades']}")

    return {
        'avg_time_per_run': avg_time_per_run,
        'bars_per_second': bars_per_second,
        'total_time': benchmark_time,
        'sample_trades': results['total_trades']
    }


if __name__ == "__main__":
    # Example usage
    from .data_loader import UltraFastDataLoader
    from .precompute import PrecomputeEngine

    parquet_path = "data_parquet/nq/1min/year=2024"

    if Path(parquet_path).exists():
        print("🚀 Testing Numba Backtest Engine")

        # Load data
        loader = UltraFastDataLoader()
        memmap_files = loader.parquet_to_memmap(parquet_path)
        arrays = loader.load_memmap_arrays(memmap_files)

        # Create precomputed data
        precompute = PrecomputeEngine()
        precomputed_files = precompute.create_precomputed_data(arrays, "numba_test")
        precomputed = precompute.load_precomputed_data(precomputed_files)

        # Prepare for Numba
        engine = NumbaBacktestEngine()
        numba_arrays = engine.prepare_numba_arrays(arrays, precomputed)

        # Run benchmark
        results = benchmark_numba_speed(numba_arrays)
        print(f"\n📊 Task 3 & 4 Complete: {results['bars_per_second']:,.0f} bars/sec achieved!")
    else:
        print(f"❌ Test data not found: {parquet_path}")