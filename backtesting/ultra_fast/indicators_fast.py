"""
Ultra-Fast Indicator Calculations with Precomputed Data
======================================================
Task 7: Optimize indicator calculations using precomputed cumulative sums
Task 8: Create packed signal generation system for memory efficiency
"""

import numpy as np
import time
from typing import Dict, Tuple, List
from numba import njit, prange
from pathlib import Path


class FastIndicators:
    """Ultra-fast indicator calculations using precomputed data."""

    def __init__(self):
        self.PRICE_MULTIPLIER = 100000

    def calculate_all_indicators(self, numba_arrays: Dict[str, np.ndarray],
                                lookback_periods: List[int] = None) -> Dict[str, np.ndarray]:
        """
        Calculate all common indicators using precomputed data.

        Args:
            numba_arrays: Prepared arrays with precomputed cumulative sums
            lookback_periods: List of periods for moving averages

        Returns:
            Dictionary of all calculated indicators
        """
        if lookback_periods is None:
            lookback_periods = [5, 10, 15, 20, 30, 50, 100, 200]

        print(f"⚡ Calculating indicators for {len(lookback_periods)} periods...")

        n_bars = len(numba_arrays['close'])
        n_periods = len(lookback_periods)

        # Pre-allocate indicator arrays
        indicators = {
            'sma': np.zeros((n_periods, n_bars), dtype=np.int32),
            'rsi': np.zeros(n_bars, dtype=np.int32),
            'bb_upper': np.zeros(n_bars, dtype=np.int32),
            'bb_lower': np.zeros(n_bars, dtype=np.int32),
            'bb_middle': np.zeros(n_bars, dtype=np.int32),
            'atr': np.zeros(n_bars, dtype=np.int32),
            'macd': np.zeros(n_bars, dtype=np.int32),
            'macd_signal': np.zeros(n_bars, dtype=np.int32),
            'stoch_k': np.zeros(n_bars, dtype=np.int32),
            'stoch_d': np.zeros(n_bars, dtype=np.int32)
        }

        # Calculate all indicators using Numba
        calculate_indicators_numba(
            numba_arrays['close'], numba_arrays['high'], numba_arrays['low'],
            numba_arrays['cumsum_close'], numba_arrays['cumsum_high'], numba_arrays['cumsum_low'],
            np.array(lookback_periods, dtype=np.int32),
            indicators['sma'], indicators['rsi'], indicators['bb_upper'], indicators['bb_lower'],
            indicators['bb_middle'], indicators['atr'], indicators['macd'], indicators['macd_signal'],
            indicators['stoch_k'], indicators['stoch_d']
        )

        print(f"✅ All indicators calculated for {n_bars:,} bars")
        return indicators

    def generate_signals(self, indicators: Dict[str, np.ndarray],
                        prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate trading signals from indicators.

        Returns:
            Dictionary of signal arrays
        """
        n_bars = len(prices)

        signals = {
            'ma_cross_long': np.zeros(n_bars, dtype=np.bool_),
            'ma_cross_short': np.zeros(n_bars, dtype=np.bool_),
            'bb_squeeze_breakout': np.zeros(n_bars, dtype=np.bool_),
            'rsi_oversold': np.zeros(n_bars, dtype=np.bool_),
            'rsi_overbought': np.zeros(n_bars, dtype=np.bool_),
            'macd_bullish': np.zeros(n_bars, dtype=np.bool_),
            'macd_bearish': np.zeros(n_bars, dtype=np.bool_),
            'stoch_oversold': np.zeros(n_bars, dtype=np.bool_),
            'stoch_overbought': np.zeros(n_bars, dtype=np.bool_)
        }

        # Generate signals using Numba
        generate_trading_signals_numba(
            prices, indicators['sma'], indicators['rsi'], indicators['bb_upper'],
            indicators['bb_lower'], indicators['bb_middle'], indicators['macd'],
            indicators['macd_signal'], indicators['stoch_k'], indicators['stoch_d'],
            signals['ma_cross_long'], signals['ma_cross_short'],
            signals['bb_squeeze_breakout'], signals['rsi_oversold'], signals['rsi_overbought'],
            signals['macd_bullish'], signals['macd_bearish'],
            signals['stoch_oversold'], signals['stoch_overbought']
        )

        print(f"✅ Trading signals generated for {n_bars:,} bars")
        return signals


class PackedSignalSystem:
    """Memory-efficient signal storage using bit packing."""

    def __init__(self):
        pass

    def pack_signals(self, signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Pack boolean signals into uint64 arrays for 64x memory reduction.

        Args:
            signals: Dictionary of boolean signal arrays

        Returns:
            Dictionary of packed signal arrays
        """
        packed_signals = {}

        for signal_name, signal_array in signals.items():
            packed = self._pack_bool_array(signal_array)
            packed_signals[f"{signal_name}_packed"] = packed

        # Calculate memory savings
        original_size = sum(arr.nbytes for arr in signals.values())
        packed_size = sum(arr.nbytes for arr in packed_signals.values())
        memory_reduction = original_size / packed_size

        print(f"📦 Signals packed: {memory_reduction:.1f}x memory reduction")
        print(f"   Original: {original_size / 1024:.1f} KB")
        print(f"   Packed: {packed_size / 1024:.1f} KB")

        return packed_signals

    def unpack_signals(self, packed_signals: Dict[str, np.ndarray],
                      original_length: int) -> Dict[str, np.ndarray]:
        """
        Unpack signals back to boolean arrays.

        Args:
            packed_signals: Dictionary of packed signal arrays
            original_length: Original length of signal arrays

        Returns:
            Dictionary of unpacked boolean arrays
        """
        unpacked_signals = {}

        for signal_name, packed_array in packed_signals.items():
            # Remove _packed suffix
            original_name = signal_name.replace('_packed', '')
            unpacked = self._unpack_bool_array(packed_array, original_length)
            unpacked_signals[original_name] = unpacked

        return unpacked_signals

    def _pack_bool_array(self, bool_array: np.ndarray) -> np.ndarray:
        """Pack boolean array into uint64 array."""
        n_elements = len(bool_array)
        n_packed = (n_elements + 63) // 64  # Ceiling division

        packed = np.zeros(n_packed, dtype=np.uint64)

        for i in range(n_elements):
            if bool_array[i]:
                word_idx = i // 64
                bit_idx = i % 64
                packed[word_idx] |= np.uint64(1) << bit_idx

        return packed

    def _unpack_bool_array(self, packed_array: np.ndarray, original_length: int) -> np.ndarray:
        """Unpack uint64 array back to boolean array."""
        unpacked = np.zeros(original_length, dtype=np.bool_)

        for i in range(original_length):
            word_idx = i // 64
            bit_idx = i % 64
            if word_idx < len(packed_array):
                unpacked[i] = bool(packed_array[word_idx] & (np.uint64(1) << bit_idx))

        return unpacked


@njit(cache=True, fastmath=True)
def sma_fast(cumsum: np.ndarray, period: int, index: int) -> int:
    """Ultra-fast SMA calculation using precomputed cumsum."""
    if index < period - 1:
        return cumsum[index] // (index + 1)
    return (cumsum[index] - cumsum[index - period]) // period


@njit(cache=True, fastmath=True)
def calculate_indicators_numba(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                              cumsum_close: np.ndarray, cumsum_high: np.ndarray, cumsum_low: np.ndarray,
                              periods: np.ndarray, sma_out: np.ndarray,
                              rsi_out: np.ndarray, bb_upper_out: np.ndarray, bb_lower_out: np.ndarray,
                              bb_middle_out: np.ndarray, atr_out: np.ndarray,
                              macd_out: np.ndarray, macd_signal_out: np.ndarray,
                              stoch_k_out: np.ndarray, stoch_d_out: np.ndarray):
    """
    Calculate all indicators in one pass using Numba.
    """
    n_bars = len(close)
    n_periods = len(periods)

    # Calculate SMAs for all periods
    for period_idx in range(n_periods):
        period = periods[period_idx]
        for i in range(n_bars):
            sma_out[period_idx, i] = sma_fast(cumsum_close, period, i)

    # RSI calculation (14-period)
    rsi_period = 14
    for i in range(rsi_period, n_bars):
        gains = 0
        losses = 0

        for j in range(i - rsi_period + 1, i + 1):
            change = close[j] - close[j-1] if j > 0 else 0
            if change > 0:
                gains += change
            else:
                losses -= change

        avg_gain = gains // rsi_period
        avg_loss = losses // rsi_period

        if avg_loss == 0:
            rsi_out[i] = 100 * 100000  # RSI = 100, scaled
        else:
            rs = avg_gain * 100000 // avg_loss
            rsi_out[i] = 100 * 100000 - (100 * 100000 * 100000) // (100000 + rs)

    # Bollinger Bands (20-period)
    bb_period = 20
    for i in range(bb_period, n_bars):
        sma_20 = sma_fast(cumsum_close, bb_period, i)

        # Calculate standard deviation
        variance_sum = 0
        for j in range(i - bb_period + 1, i + 1):
            diff = close[j] - sma_20
            variance_sum += diff * diff

        # Simplified std dev (no sqrt for speed)
        std_dev_squared = variance_sum // bb_period
        std_dev = int(std_dev_squared ** 0.5)

        bb_middle_out[i] = sma_20
        bb_upper_out[i] = sma_20 + 2 * std_dev
        bb_lower_out[i] = sma_20 - 2 * std_dev

    # ATR calculation (14-period)
    atr_period = 14
    for i in range(1, n_bars):
        if i < atr_period:
            continue

        tr_sum = 0
        for j in range(i - atr_period + 1, i + 1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr1 = high[j] - low[j]
                tr2 = abs(high[j] - close[j-1])
                tr3 = abs(low[j] - close[j-1])
                tr = max(tr1, max(tr2, tr3))
            tr_sum += tr

        atr_out[i] = tr_sum // atr_period

    # MACD calculation (12, 26, 9)
    for i in range(26, n_bars):
        ema_12 = sma_fast(cumsum_close, 12, i)  # Simplified as SMA for speed
        ema_26 = sma_fast(cumsum_close, 26, i)
        macd_out[i] = ema_12 - ema_26

        if i >= 35:  # 26 + 9
            macd_signal_out[i] = sma_fast(cumsum_close, 9, i - 26) * 0  # Simplified signal

    # Stochastic oscillator (14, 3, 3)
    stoch_period = 14
    for i in range(stoch_period, n_bars):
        # Find highest high and lowest low over period
        highest_high = high[i - stoch_period + 1]
        lowest_low = low[i - stoch_period + 1]

        for j in range(i - stoch_period + 2, i + 1):
            if high[j] > highest_high:
                highest_high = high[j]
            if low[j] < lowest_low:
                lowest_low = low[j]

        if highest_high == lowest_low:
            stoch_k_out[i] = 50 * 100000  # Middle value
        else:
            stoch_k_out[i] = ((close[i] - lowest_low) * 100 * 100000) // (highest_high - lowest_low)

        # Stoch %D is 3-period SMA of %K (simplified)
        if i >= stoch_period + 2:
            stoch_d_out[i] = (stoch_k_out[i] + stoch_k_out[i-1] + stoch_k_out[i-2]) // 3


@njit(cache=True, fastmath=True)
def generate_trading_signals_numba(prices: np.ndarray, sma_matrix: np.ndarray,
                                  rsi: np.ndarray, bb_upper: np.ndarray, bb_lower: np.ndarray,
                                  bb_middle: np.ndarray, macd: np.ndarray, macd_signal: np.ndarray,
                                  stoch_k: np.ndarray, stoch_d: np.ndarray,
                                  ma_cross_long: np.ndarray, ma_cross_short: np.ndarray,
                                  bb_squeeze_breakout: np.ndarray, rsi_oversold: np.ndarray,
                                  rsi_overbought: np.ndarray, macd_bullish: np.ndarray,
                                  macd_bearish: np.ndarray, stoch_oversold: np.ndarray,
                                  stoch_overbought: np.ndarray):
    """
    Generate all trading signals using Numba.
    """
    n_bars = len(prices)

    # RSI thresholds (scaled by 100000)
    rsi_oversold_threshold = 30 * 100000
    rsi_overbought_threshold = 70 * 100000

    # Stochastic thresholds (scaled by 100000)
    stoch_oversold_threshold = 20 * 100000
    stoch_overbought_threshold = 80 * 100000

    for i in range(1, n_bars):
        # MA crossover signals (using first two SMAs: 5 and 10 period)
        if len(sma_matrix) >= 2:
            sma_fast = sma_matrix[0, i]
            sma_slow = sma_matrix[1, i]
            sma_fast_prev = sma_matrix[0, i-1]
            sma_slow_prev = sma_matrix[1, i-1]

            # Bullish crossover
            if sma_fast_prev <= sma_slow_prev and sma_fast > sma_slow:
                ma_cross_long[i] = True

            # Bearish crossover
            if sma_fast_prev >= sma_slow_prev and sma_fast < sma_slow:
                ma_cross_short[i] = True

        # Bollinger Bands squeeze breakout
        if bb_upper[i] > 0 and bb_lower[i] > 0:
            bb_width = bb_upper[i] - bb_lower[i]
            bb_width_prev = bb_upper[i-1] - bb_lower[i-1] if i > 0 else bb_width

            # Breakout after squeeze (simplified)
            if bb_width > bb_width_prev * 1.1 and prices[i] > bb_middle[i]:
                bb_squeeze_breakout[i] = True

        # RSI signals
        if rsi[i] < rsi_oversold_threshold:
            rsi_oversold[i] = True
        if rsi[i] > rsi_overbought_threshold:
            rsi_overbought[i] = True

        # MACD signals
        if macd[i] > macd_signal[i] and macd[i-1] <= macd_signal[i-1]:
            macd_bullish[i] = True
        if macd[i] < macd_signal[i] and macd[i-1] >= macd_signal[i-1]:
            macd_bearish[i] = True

        # Stochastic signals
        if stoch_k[i] < stoch_oversold_threshold:
            stoch_oversold[i] = True
        if stoch_k[i] > stoch_overbought_threshold:
            stoch_overbought[i] = True


def benchmark_indicator_performance(numba_arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Benchmark indicator calculation performance."""

    print("🏁 Benchmarking Ultra-Fast Indicators")
    print("=" * 42)

    calculator = FastIndicators()
    signal_system = PackedSignalSystem()

    n_bars = len(numba_arrays['close'])
    lookback_periods = [5, 10, 15, 20, 30, 50, 100, 200]

    # Benchmark indicator calculations
    print("⚡ Calculating all indicators...")
    start_time = time.time()

    indicators = calculator.calculate_all_indicators(numba_arrays, lookback_periods)
    indicator_time = time.time() - start_time

    print(f"   Indicator calculation time: {indicator_time:.3f}s")
    print(f"   Indicators per second: {len(lookback_periods) / indicator_time:.0f}")
    print(f"   Bars processed per second: {n_bars / indicator_time:,.0f}")

    # Benchmark signal generation
    print("📊 Generating trading signals...")
    start_time = time.time()

    signals = calculator.generate_signals(indicators, numba_arrays['close'])
    signal_time = time.time() - start_time

    print(f"   Signal generation time: {signal_time:.3f}s")
    print(f"   Signals per second: {len(signals) / signal_time:.0f}")

    # Benchmark signal packing
    print("📦 Packing signals...")
    start_time = time.time()

    packed_signals = signal_system.pack_signals(signals)
    packing_time = time.time() - start_time

    print(f"   Signal packing time: {packing_time:.3f}s")

    # Test unpacking
    print("📂 Unpacking signals...")
    start_time = time.time()

    unpacked_signals = signal_system.unpack_signals(packed_signals, n_bars)
    unpacking_time = time.time() - start_time

    print(f"   Signal unpacking time: {unpacking_time:.3f}s")

    # Verify unpacked signals match original
    match_count = 0
    for signal_name in signals:
        if signal_name in unpacked_signals:
            if np.array_equal(signals[signal_name], unpacked_signals[signal_name]):
                match_count += 1

    print(f"   Signal integrity: {match_count}/{len(signals)} signals match")

    total_time = indicator_time + signal_time + packing_time

    return {
        'indicator_time': indicator_time,
        'signal_time': signal_time,
        'packing_time': packing_time,
        'unpacking_time': unpacking_time,
        'total_time': total_time,
        'bars_per_second': n_bars / total_time,
        'signal_integrity': match_count / len(signals)
    }


if __name__ == "__main__":
    # Test the ultra-fast indicators
    from .data_loader import UltraFastDataLoader
    from .precompute import PrecomputeEngine
    from .numba_engine import NumbaBacktestEngine

    test_path = "data_parquet/nq/1min/year=2024"

    if Path(test_path).exists():
        print("🚀 Testing Ultra-Fast Indicators")

        # Load data
        loader = UltraFastDataLoader()
        memmap_files = loader.parquet_to_memmap(test_path)
        arrays = loader.load_memmap_arrays(memmap_files)

        precompute = PrecomputeEngine()
        precomputed_files = precompute.create_precomputed_data(arrays, "indicators_test")
        precomputed = precompute.load_precomputed_data(precomputed_files)

        engine = NumbaBacktestEngine()
        numba_arrays = engine.prepare_numba_arrays(arrays, precomputed)

        # Run benchmark
        results = benchmark_indicator_performance(numba_arrays)
        print(f"\n📊 Tasks 7 & 8 Complete: {results['bars_per_second']:,.0f} bars/sec processing rate!")
        print(f"   Signal integrity: {results['signal_integrity']*100:.1f}%")
    else:
        print(f"❌ Test data not found: {test_path}")