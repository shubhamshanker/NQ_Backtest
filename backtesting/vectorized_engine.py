"""
Ultra-Fast Vectorized Backtesting Engine
========================================
- Vectorized calculations using NumPy/Pandas
- Support for 1min, 3min, 5min, 15min timeframes
- Lightning fast execution
- Optimized for single contract strategies
- Correct calculations and strong logic
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import time, datetime
import warnings
warnings.filterwarnings('ignore')

class VectorizedBacktestEngine:
    """Ultra-fast vectorized backtesting engine."""

    def __init__(self, data_path: str, initial_capital: float = 100000.0,
                 point_value: float = 20.0, commission_per_trade: float = 2.50):
        """Initialize vectorized backtest engine."""
        self.data_path = data_path
        self.initial_capital = initial_capital
        self.point_value = point_value
        self.commission_per_trade = commission_per_trade

        # Load and prepare data
        self.df = self._load_and_prepare_data()
        self.results = {}

    def _load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare data with timezone handling."""
        print(f"⚡ Loading data from {self.data_path}")

        # Read CSV
        df = pd.read_csv(self.data_path)

        # Parse timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        else:
            # Handle different timestamp column names
            timestamp_cols = ['datetime', 'time', 'date']
            for col in timestamp_cols:
                if col in df.columns:
                    df['timestamp'] = pd.to_datetime(df[col], utc=True)
                    break

        # Ensure OHLCV columns exist
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                # Try capitalized versions
                cap_col = col.capitalize()
                if cap_col in df.columns:
                    df[col] = df[cap_col]

        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # Add time-based columns
        df['time'] = df.index.time
        df['date'] = df.index.date
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute

        print(f"⚡ Data prepared: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
        return df

    def resample_data(self, timeframe: str) -> pd.DataFrame:
        """Resample data to different timeframes."""
        if timeframe == 'original':
            return self.df.copy()

        # Mapping for pandas resample
        tf_map = {
            '1min': '1T',
            '3min': '3T',
            '5min': '5T',
            '15min': '15T',
            '30min': '30T',
            '1H': '1H'
        }

        if timeframe not in tf_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        print(f"⚡ Resampling to {timeframe}")

        # Resample OHLCV data
        ohlc_data = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
        }

        if 'volume' in self.df.columns:
            ohlc_data['volume'] = 'sum'

        resampled = self.df.resample(tf_map[timeframe]).agg(ohlc_data).dropna()

        # Add time columns
        resampled['time'] = resampled.index.time
        resampled['date'] = resampled.index.date
        resampled['hour'] = resampled.index.hour
        resampled['minute'] = resampled.index.minute

        print(f"⚡ Resampled to {len(resampled)} {timeframe} bars")
        return resampled

    def run_vectorized_orb_strategy(self, timeframe: str = '15min',
                                  stop_points: float = 60.0, rr_ratio: float = 2.0,
                                  max_trades_per_day: int = 5, risk_per_trade: float = 0.025,
                                  or_minutes: int = 15) -> Dict[str, Any]:
        """Run vectorized ORB strategy - ultra fast."""

        # Get data for timeframe
        data = self.resample_data(timeframe)

        print(f"🚀 Running Vectorized ORB - {timeframe} - SL:{stop_points} RR:{rr_ratio} MT:{max_trades_per_day}")

        # Trading hours filter - vectorized
        market_open = time(9, 30)
        market_close = time(16, 0)
        trade_start = time(9, 45)

        # Create boolean masks - handle different time formats
        if 'time' not in data.columns:
            data['time'] = data.index.time

        trading_hours = (data['time'] >= trade_start) & (data['time'] <= market_close)
        opening_range_time = data['time'] == market_open

        # Calculate opening ranges per day - vectorized
        or_data = data[opening_range_time].copy()
        or_data['or_high'] = or_data['high']
        or_data['or_low'] = or_data['low']
        or_data['or_date'] = or_data['date']

        # Forward fill OR levels throughout each day
        data_with_or = data.copy()
        data_with_or = data_with_or.merge(
            or_data[['or_date', 'or_high', 'or_low']].rename(columns={'or_date': 'date'}),
            on='date', how='left'
        )
        data_with_or['or_high'] = data_with_or.groupby('date')['or_high'].ffill()
        data_with_or['or_low'] = data_with_or.groupby('date')['or_low'].ffill()

        # Filter to trading hours only
        trade_data = data_with_or[trading_hours & data_with_or['or_high'].notna()].copy()

        if len(trade_data) == 0:
            return {'error': 'No trading data available'}

        # Generate signals - vectorized
        trade_data['long_signal'] = (trade_data['close'] > trade_data['or_high']).astype(int)
        trade_data['short_signal'] = (trade_data['close'] < trade_data['or_low']).astype(int)

        # Find signal changes (entries)
        trade_data['long_entry'] = (trade_data['long_signal'].diff() == 1)
        trade_data['short_entry'] = (trade_data['short_signal'].diff() == 1)

        # Generate trades
        trades = self._generate_vectorized_trades(trade_data, stop_points, rr_ratio, max_trades_per_day)

        if len(trades) == 0:
            return {'error': 'No trades generated'}

        # Calculate performance
        return self._calculate_vectorized_performance(trades, timeframe)

    def _generate_vectorized_trades(self, data: pd.DataFrame, stop_points: float,
                                  rr_ratio: float, max_trades_per_day: int) -> pd.DataFrame:
        """Generate trades using vectorized operations."""

        all_trades = []
        target_points = stop_points * rr_ratio

        # Process each day separately for trade limits
        for date, day_data in data.groupby('date'):
            day_trades = 0
            i = 0
            day_data_list = day_data.reset_index()

            while i < len(day_data_list) and day_trades < max_trades_per_day:
                row = day_data_list.iloc[i]

                # Check for long entry
                if row['long_entry'] and day_trades < max_trades_per_day:
                    trade = self._process_trade(day_data_list, i, 'LONG', row['close'],
                                             row['close'] - stop_points,
                                             row['close'] + target_points)
                    if trade:
                        all_trades.append(trade)
                        day_trades += 1

                # Check for short entry
                elif row['short_entry'] and day_trades < max_trades_per_day:
                    trade = self._process_trade(day_data_list, i, 'SHORT', row['close'],
                                             row['close'] + stop_points,
                                             row['close'] - target_points)
                    if trade:
                        all_trades.append(trade)
                        day_trades += 1

                i += 1

        return pd.DataFrame(all_trades)

    def _process_trade(self, data: pd.DataFrame, entry_idx: int, direction: str,
                      entry_price: float, stop_loss: float, take_profit: float) -> Optional[Dict]:
        """Process individual trade with vectorized exit detection."""

        if entry_idx >= len(data) - 1:
            return None

        entry_row = data.iloc[entry_idx]
        remaining_data = data.iloc[entry_idx + 1:]

        if len(remaining_data) == 0:
            return None

        # Vectorized exit conditions
        if direction == 'LONG':
            stop_hit = remaining_data['low'] <= stop_loss
            target_hit = remaining_data['high'] >= take_profit
        else:  # SHORT
            stop_hit = remaining_data['high'] >= stop_loss
            target_hit = remaining_data['low'] <= take_profit

        # Find first exit
        stop_exits = remaining_data[stop_hit]
        target_exits = remaining_data[target_hit]

        exit_price = entry_price
        exit_time = entry_row['timestamp']
        exit_reason = 'EOD'

        # Determine which exit happened first
        if len(stop_exits) > 0 and len(target_exits) > 0:
            if stop_exits.index[0] <= target_exits.index[0]:
                exit_price = stop_loss
                exit_time = stop_exits.iloc[0]['timestamp']
                exit_reason = 'STOP'
            else:
                exit_price = take_profit
                exit_time = target_exits.iloc[0]['timestamp']
                exit_reason = 'TARGET'
        elif len(stop_exits) > 0:
            exit_price = stop_loss
            exit_time = stop_exits.iloc[0]['timestamp']
            exit_reason = 'STOP'
        elif len(target_exits) > 0:
            exit_price = take_profit
            exit_time = target_exits.iloc[0]['timestamp']
            exit_reason = 'TARGET'

        # Calculate P&L
        if direction == 'LONG':
            points_pnl = exit_price - entry_price
        else:
            points_pnl = entry_price - exit_price

        dollar_pnl = points_pnl * self.point_value - self.commission_per_trade

        return {
            'entry_time': entry_row['timestamp'],
            'exit_time': exit_time,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'points_pnl': points_pnl,
            'dollar_pnl': dollar_pnl,
            'exit_reason': exit_reason,
            'contracts': 1,
            'duration_minutes': (exit_time - entry_row['timestamp']).total_seconds() / 60
        }

    def _calculate_vectorized_performance(self, trades: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Calculate performance metrics using vectorized operations."""

        if len(trades) == 0:
            return {'error': 'No trades to analyze'}

        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(trades[trades['points_pnl'] > 0])
        losing_trades = len(trades[trades['points_pnl'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        # P&L calculations
        total_points = trades['points_pnl'].sum()
        total_dollar_pnl = trades['dollar_pnl'].sum()

        # Drawdown calculation - vectorized
        equity_curve = trades['dollar_pnl'].cumsum() + self.initial_capital
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max * 100
        max_drawdown = drawdowns.min()

        # Profit factor
        gross_profit = trades[trades['points_pnl'] > 0]['points_pnl'].sum()
        gross_loss = abs(trades[trades['points_pnl'] <= 0]['points_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Time-based metrics
        trades['entry_date'] = pd.to_datetime(trades['entry_time']).dt.date
        trading_days = trades['entry_date'].nunique()
        points_per_day = total_points / trading_days if trading_days > 0 else 0
        trades_per_day = total_trades / trading_days if trading_days > 0 else 0

        # Duration analysis
        avg_duration = trades['duration_minutes'].mean()

        return {
            'timeframe': timeframe,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_points': total_points,
            'total_pnl': total_dollar_pnl,
            'points_per_day': points_per_day,
            'trades_per_day': trades_per_day,
            'max_drawdown_percent': max_drawdown,
            'profit_factor': profit_factor,
            'avg_duration_minutes': avg_duration,
            'trading_days': trading_days,
            'final_equity': self.initial_capital + total_dollar_pnl,
            'return_percent': (total_dollar_pnl / self.initial_capital) * 100
        }

class UltraFastORBStrategy:
    """Ultra-fast ORB strategy configurations for different timeframes."""

    @staticmethod
    def get_optimized_configs_by_timeframe(timeframe: str) -> List[Dict[str, Any]]:
        """Get optimized configurations for specific timeframes."""

        if timeframe == '1min':
            return [
                # High frequency 1-minute configs - MORE PARAMS, LOW DRAWDOWN FOCUS
                {'stop': 8, 'rr': 4.0, 'max_trades': 30, 'risk': 0.008, 'or_min': 5},
                {'stop': 10, 'rr': 3.5, 'max_trades': 25, 'risk': 0.01, 'or_min': 5},
                {'stop': 12, 'rr': 3.0, 'max_trades': 22, 'risk': 0.012, 'or_min': 5},
                {'stop': 15, 'rr': 2.8, 'max_trades': 20, 'risk': 0.015, 'or_min': 5},
                {'stop': 18, 'rr': 2.5, 'max_trades': 18, 'risk': 0.018, 'or_min': 5},
                {'stop': 20, 'rr': 2.2, 'max_trades': 15, 'risk': 0.02, 'or_min': 5},
                {'stop': 25, 'rr': 2.0, 'max_trades': 12, 'risk': 0.025, 'or_min': 10},
                {'stop': 30, 'rr': 1.8, 'max_trades': 10, 'risk': 0.03, 'or_min': 10},
                # Conservative low-drawdown configs
                {'stop': 5, 'rr': 5.0, 'max_trades': 40, 'risk': 0.005, 'or_min': 5},
                {'stop': 6, 'rr': 4.5, 'max_trades': 35, 'risk': 0.006, 'or_min': 5},
            ]

        elif timeframe == '3min':
            return [
                # Medium frequency 3-minute configs - EXPANDED WITH LOW DD
                {'stop': 15, 'rr': 3.0, 'max_trades': 18, 'risk': 0.01, 'or_min': 6},
                {'stop': 18, 'rr': 2.8, 'max_trades': 16, 'risk': 0.012, 'or_min': 6},
                {'stop': 20, 'rr': 2.5, 'max_trades': 15, 'risk': 0.015, 'or_min': 9},
                {'stop': 25, 'rr': 2.2, 'max_trades': 12, 'risk': 0.018, 'or_min': 9},
                {'stop': 30, 'rr': 2.0, 'max_trades': 10, 'risk': 0.02, 'or_min': 9},
                {'stop': 35, 'rr': 1.8, 'max_trades': 8, 'risk': 0.025, 'or_min': 12},
                {'stop': 40, 'rr': 1.6, 'max_trades': 7, 'risk': 0.03, 'or_min': 15},
                # Ultra-conservative
                {'stop': 12, 'rr': 3.5, 'max_trades': 20, 'risk': 0.008, 'or_min': 6},
                {'stop': 10, 'rr': 4.0, 'max_trades': 25, 'risk': 0.006, 'or_min': 6},
            ]

        elif timeframe == '5min':
            return [
                # 5-minute optimized configs - COMPREHENSIVE WITH LOW DD
                {'stop': 20, 'rr': 3.0, 'max_trades': 12, 'risk': 0.012, 'or_min': 10},
                {'stop': 25, 'rr': 2.8, 'max_trades': 10, 'risk': 0.015, 'or_min': 10},
                {'stop': 30, 'rr': 2.5, 'max_trades': 9, 'risk': 0.018, 'or_min': 10},
                {'stop': 35, 'rr': 2.2, 'max_trades': 8, 'risk': 0.02, 'or_min': 15},
                {'stop': 40, 'rr': 2.0, 'max_trades': 7, 'risk': 0.025, 'or_min': 15},
                {'stop': 45, 'rr': 1.8, 'max_trades': 6, 'risk': 0.03, 'or_min': 15},
                {'stop': 50, 'rr': 1.6, 'max_trades': 5, 'risk': 0.035, 'or_min': 20},
                # Conservative low-risk
                {'stop': 15, 'rr': 3.5, 'max_trades': 15, 'risk': 0.01, 'or_min': 10},
                {'stop': 18, 'rr': 3.2, 'max_trades': 14, 'risk': 0.011, 'or_min': 10},
                {'stop': 55, 'rr': 1.4, 'max_trades': 4, 'risk': 0.04, 'or_min': 20},
            ]

        elif timeframe == '15min':
            return [
                # 15-minute configs (starting at 15+ points baseline) - EXPANDED
                {'stop': 40, 'rr': 3.5, 'max_trades': 4, 'risk': 0.035, 'or_min': 15},
                {'stop': 45, 'rr': 3.0, 'max_trades': 5, 'risk': 0.03, 'or_min': 15},
                {'stop': 50, 'rr': 2.8, 'max_trades': 5, 'risk': 0.028, 'or_min': 15},
                {'stop': 55, 'rr': 2.5, 'max_trades': 6, 'risk': 0.025, 'or_min': 15},
                {'stop': 60, 'rr': 2.2, 'max_trades': 6, 'risk': 0.025, 'or_min': 15},
                {'stop': 65, 'rr': 2.0, 'max_trades': 7, 'risk': 0.022, 'or_min': 15},
                {'stop': 70, 'rr': 1.8, 'max_trades': 7, 'risk': 0.02, 'or_min': 15},
                # Ultra-aggressive for 30+ target
                {'stop': 35, 'rr': 4.0, 'max_trades': 3, 'risk': 0.04, 'or_min': 15},
                {'stop': 30, 'rr': 4.5, 'max_trades': 3, 'risk': 0.045, 'or_min': 15},
                {'stop': 75, 'rr': 1.6, 'max_trades': 8, 'risk': 0.018, 'or_min': 15},
            ]

        else:
            # Default configs
            return [
                {'stop': 50, 'rr': 2.0, 'max_trades': 5, 'risk': 0.025, 'or_min': 15},
                {'stop': 60, 'rr': 1.8, 'max_trades': 6, 'risk': 0.025, 'or_min': 15},
            ]

    @staticmethod
    def get_all_timeframe_configs() -> Dict[str, List[Dict[str, Any]]]:
        """Get all optimized configs for all timeframes."""
        return {
            '1min': UltraFastORBStrategy.get_optimized_configs_by_timeframe('1min'),
            '3min': UltraFastORBStrategy.get_optimized_configs_by_timeframe('3min'),
            '5min': UltraFastORBStrategy.get_optimized_configs_by_timeframe('5min'),
            '15min': UltraFastORBStrategy.get_optimized_configs_by_timeframe('15min'),
        }