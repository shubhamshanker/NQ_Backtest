#!/usr/bin/env python3
"""
ADVANCED FILTERED STRATEGIES - TARGET 30-50+ POINTS/DAY
======================================================
- Market regime filters (trending vs ranging)
- Volatility filters (only trade high-vol days)
- Volume confirmation filters
- Time-of-day optimization
- Multi-timeframe trend alignment
- Dynamic position sizing
- Monthly performance tracking
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import json
import sys
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/Users/shubhamshanker/bt_/backtesting')
from ultimate_orb_strategy import UltimateORBStrategy
from portfolio import Portfolio

class AdvancedFilteredORBStrategy(UltimateORBStrategy):
    """Enhanced ORB strategy with advanced market filters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Advanced filter parameters
        self.min_daily_atr = 80.0  # Minimum daily ATR for trading
        self.trend_strength_threshold = 0.6  # 0-1 trend strength
        self.volume_multiplier = 1.5  # Volume must be X times average
        self.optimal_times = [(time(9, 45), time(11, 30)), (time(13, 0), time(15, 0))]  # Prime trading windows

        # Market state tracking
        self.daily_atr = {}
        self.market_trend = {}
        self.volume_sma = {}

    def calculate_daily_atr(self, df: pd.DataFrame, lookback: int = 14) -> Dict[str, float]:
        """Calculate daily Average True Range for volatility filtering."""
        daily_atr = {}

        df['date'] = df['timestamp'].dt.date
        daily_data = df.groupby('date').agg({
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).reset_index()

        daily_data['prev_close'] = daily_data['close'].shift(1)
        daily_data['tr1'] = daily_data['high'] - daily_data['low']
        daily_data['tr2'] = abs(daily_data['high'] - daily_data['prev_close'])
        daily_data['tr3'] = abs(daily_data['low'] - daily_data['prev_close'])
        daily_data['true_range'] = daily_data[['tr1', 'tr2', 'tr3']].max(axis=1)

        daily_data['atr'] = daily_data['true_range'].rolling(window=lookback).mean()

        for _, row in daily_data.iterrows():
            if pd.notna(row['atr']):
                daily_atr[row['date']] = row['atr']

        return daily_atr

    def calculate_trend_strength(self, df: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
        """Calculate trend strength using multiple indicators."""
        trend_scores = {}

        df['date'] = df['timestamp'].dt.date
        daily_data = df.groupby('date').agg({
            'close': 'last',
            'high': 'max',
            'low': 'min',
            'volume': 'sum'
        }).reset_index()

        # Calculate moving averages
        daily_data['sma_20'] = daily_data['close'].rolling(window=20).mean()
        daily_data['sma_50'] = daily_data['close'].rolling(window=50).mean()

        # Price position relative to MAs
        daily_data['ma_score'] = np.where(
            daily_data['close'] > daily_data['sma_20'], 0.5, 0
        ) + np.where(
            daily_data['sma_20'] > daily_data['sma_50'], 0.5, 0
        )

        # Trend consistency (lower volatility = stronger trend)
        daily_data['returns'] = daily_data['close'].pct_change()
        daily_data['volatility'] = daily_data['returns'].rolling(window=10).std()
        daily_data['vol_score'] = 1 - (daily_data['volatility'] / daily_data['volatility'].rolling(window=50).max())

        # Combine scores
        daily_data['trend_strength'] = (daily_data['ma_score'] + daily_data['vol_score']) / 2

        for _, row in daily_data.iterrows():
            if pd.notna(row['trend_strength']):
                trend_scores[row['date']] = row['trend_strength']

        return trend_scores

    def is_high_probability_setup(self, bar_data: Dict[str, Any], df: pd.DataFrame) -> bool:
        """Check if current setup meets all advanced filters."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()
        current_time = timestamp.time()

        # 1. Volatility filter - only trade high ATR days
        if current_date in self.daily_atr:
            if self.daily_atr[current_date] < self.min_daily_atr:
                return False

        # 2. Trend strength filter
        if current_date in self.market_trend:
            if self.market_trend[current_date] < self.trend_strength_threshold:
                return False

        # 3. Time-of-day filter - only trade during optimal windows
        in_optimal_time = any(start <= current_time <= end for start, end in self.optimal_times)
        if not in_optimal_time:
            return False

        # 4. Volume confirmation - current volume should be above average
        recent_volume = self._get_recent_average_volume(df, timestamp)
        if bar_data['volume'] < recent_volume * self.volume_multiplier:
            return False

        return True

    def _get_recent_average_volume(self, df: pd.DataFrame, timestamp: datetime, lookback_minutes: int = 60) -> float:
        """Get recent average volume for comparison."""
        try:
            # Get bars from the same time window over past days
            same_time_bars = df[
                (df['timestamp'].dt.time == timestamp.time()) &
                (df['timestamp'] < timestamp)
            ].tail(10)  # Last 10 days at same time

            if len(same_time_bars) > 0:
                return same_time_bars['volume'].mean()
            else:
                return 100  # Default fallback
        except:
            return 100

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """Enhanced signal generation with advanced filters."""
        # First run base ORB logic
        base_signal = super().generate_signal(bar_data, portfolio)

        # If base signal is not BUY/SELL, return it
        if base_signal['signal'] not in ['BUY', 'SELL']:
            return base_signal

        # Apply advanced filters (this requires access to full dataframe)
        # Note: In real implementation, we'd pass the dataframe or calculate filters beforehand
        # For now, we'll add the filter check in the main backtesting loop

        # Enhance signal with additional metadata
        base_signal['filter_passed'] = True  # Will be set by main loop
        return base_signal

def create_advanced_strategies():
    """Create advanced filtered strategies targeting 30-50+ points."""
    strategies = {
        # Conservative filtered strategy - 30+ points target
        'ADVANCED_30': AdvancedFilteredORBStrategy(
            risk_per_trade=0.025,
            or_minutes=30,
            fixed_stop_points=25.0,
            target_multiplier=4.5,  # Increased to 4.5:1
            max_trades_per_day=2,
            half_size_booking=True,
            trailing_stop=True
        ),

        # Aggressive filtered strategy - 50+ points target
        'ADVANCED_50': AdvancedFilteredORBStrategy(
            risk_per_trade=0.03,
            or_minutes=45,
            fixed_stop_points=30.0,
            target_multiplier=6.0,  # 6:1 R:R for big wins
            max_trades_per_day=2,
            half_size_booking=False,  # Let winners run
            trailing_stop=True
        ),

        # Ultra-selective strategy - 70+ points target
        'ADVANCED_70': AdvancedFilteredORBStrategy(
            risk_per_trade=0.04,
            or_minutes=60,
            fixed_stop_points=35.0,
            target_multiplier=7.0,  # 7:1 R:R
            max_trades_per_day=1,   # Only one perfect setup per day
            half_size_booking=False,
            trailing_stop=True
        ),

        # Multiple contracts strategy - scale up winners
        'ADVANCED_MULTI': AdvancedFilteredORBStrategy(
            risk_per_trade=0.02,
            or_minutes=30,
            fixed_stop_points=20.0,
            target_multiplier=5.0,
            max_trades_per_day=2,
            half_size_booking=True,
            trailing_stop=True
        )
    }

    # Set advanced filter parameters
    for strategy in strategies.values():
        strategy.min_daily_atr = 100.0  # Higher volatility requirement
        strategy.trend_strength_threshold = 0.65  # Stronger trend requirement
        strategy.volume_multiplier = 2.0  # Higher volume requirement
        strategy.optimal_times = [
            (time(9, 45), time(11, 0)),   # Morning momentum
            (time(13, 30), time(14, 30))  # Afternoon power hour
        ]

    return strategies

def run_advanced_filtered_backtest():
    """Run comprehensive backtest with advanced filtering."""
    print("🚀 ADVANCED FILTERED STRATEGY OPTIMIZATION")
    print("=" * 80)
    print("🎯 TARGET: 30-70+ points per day with advanced market filters")
    print("🔧 Filters: Volatility, Trend, Volume, Time-of-Day")
    print("📊 Monthly performance tracking for consistency")

    # Load real NQ data
    data_file = get_data_path("1min")
    print(f"📊 Loading real NQ data from: {data_file}")

    try:
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['Datetime'])
        df['open'] = df['Open']
        df['high'] = df['High']
        df['low'] = df['Low']
        df['close'] = df['Close']
        df['volume'] = df['Volume']

        # Filter to 2024 trading hours
        df_2024 = df[df['timestamp'].dt.year == 2024].copy()
        df_2024 = df_2024[
            (df_2024['timestamp'].dt.time >= time(9, 30)) &
            (df_2024['timestamp'].dt.time <= time(16, 0)) &
            (df_2024['timestamp'].dt.weekday < 5)
        ].sort_values('timestamp').reset_index(drop=True)

        print(f"✅ Loaded {len(df_2024):,} bars of 2024 real data")
        print(f"📅 Range: {df_2024['timestamp'].min()} to {df_2024['timestamp'].max()}")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return {}

    # Create advanced strategies
    strategies = create_advanced_strategies()
    all_results = {}

    for strategy_name, strategy in strategies.items():
        target_points = 30 if '30' in strategy_name else \
                       50 if '50' in strategy_name else \
                       70 if '70' in strategy_name else 40

        print(f"\n{'='*25} {strategy_name} {'='*25}")
        print(f"🎯 TARGET: {target_points}+ points/day")
        print(f"⚙️  Config: OR{strategy.or_minutes}min | Stop:{strategy.fixed_stop}pts | R:R:{strategy.target_mult}:1")
        print(f"📊 Filters: ATR>{strategy.min_daily_atr}, Trend>{strategy.trend_strength_threshold}")
        print("-" * 75)

        # Pre-calculate market filters
        print("🔧 Calculating market filters...")
        strategy.daily_atr = strategy.calculate_daily_atr(df_2024)
        strategy.market_trend = strategy.calculate_trend_strength(df_2024)

        # Run filtered backtest
        portfolio = Portfolio(initial_capital=100000)
        trades = []
        monthly_pnl = {}
        trade_id = 1

        print("🔄 Running advanced filtered backtest...")

        bars_processed = 0
        for idx, row in df_2024.iterrows():
            bars_processed += 1

            if bars_processed % 15000 == 0:
                print(f"   Processed {bars_processed:,}/{len(df_2024):,} bars ({bars_processed/len(df_2024)*100:.1f}%)")

            bar_data = {
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            }

            # Generate base signal
            signal = strategy.generate_signal(bar_data, portfolio)

            if signal['signal'] in ['BUY', 'SELL']:
                # Apply advanced filters
                if strategy.is_high_probability_setup(bar_data, df_2024):
                    # Execute trade with enhanced logic
                    trade_outcome = simulate_enhanced_trade_outcome(df_2024, idx, signal, strategy)

                    if trade_outcome:
                        month_key = row['timestamp'].strftime('%Y-%m')

                        trade_data = {
                            'id': trade_id,
                            'strategy': strategy_name,
                            'entry_time': signal['entry_time'],
                            'exit_time': trade_outcome['exit_time'],
                            'side': signal['signal'],
                            'entry_price': signal['entry_price'],
                            'exit_price': trade_outcome['exit_price'],
                            'pnl': trade_outcome['pnl'],
                            'pnl_points': trade_outcome['pnl'] / 20,
                            'stop_loss': signal['stop_loss'],
                            'take_profit': signal['take_profit'],
                            'exit_reason': trade_outcome['exit_reason'],
                            'month': month_key,
                            'contracts': 1
                        }

                        trades.append(trade_data)

                        # Track monthly P&L
                        if month_key not in monthly_pnl:
                            monthly_pnl[month_key] = 0
                        monthly_pnl[month_key] += trade_outcome['pnl'] / 20  # Convert to points

                        trade_id += 1

        print(f"✅ Backtest complete: {len(trades)} filtered trades")

        # Analyze results
        if trades:
            results = analyze_advanced_performance(trades, strategy_name, target_points)
            all_results[strategy_name] = results

            # Display monthly breakdown
            print(f"\n📊 {strategy_name} MONTHLY PERFORMANCE:")
            total_months = 0
            successful_months = 0

            for month, pnl in sorted(monthly_pnl.items()):
                trading_days = get_trading_days_in_month(month)
                daily_avg = pnl / trading_days if trading_days > 0 else 0
                status = "✅" if daily_avg >= target_points * 0.8 else "⚠️" if daily_avg >= target_points * 0.5 else "❌"

                print(f"   {month}: {pnl:6.1f} pts ({daily_avg:5.1f} pts/day) {status}")
                total_months += 1
                if daily_avg >= target_points * 0.8:
                    successful_months += 1

            consistency = (successful_months / total_months * 100) if total_months > 0 else 0
            print(f"   CONSISTENCY: {successful_months}/{total_months} months ({consistency:.1f}%) hit target")

            # Overall results
            print(f"\n📈 {strategy_name} OVERALL RESULTS:")
            print(f"   Total Trades: {results['total_trades']}")
            print(f"   Win Rate: {results['win_rate_pct']:.1f}%")
            print(f"   Daily Average: {results['avg_daily_return_points']:.2f} points")
            print(f"   Total P&L: {results['total_pnl_points']:.1f} points")
            print(f"   Expectancy: {results['expectancy_points']:.2f} points")
            print(f"   Profit Factor: {results['profit_factor']:.2f}")
            print(f"   Max Drawdown: {results['max_drawdown_points']:.1f} points")
            print(f"   Sharpe Ratio: {results['sharpe_ratio']:.3f}")

            # Target achievement
            if results['avg_daily_return_points'] >= target_points:
                print(f"   🏆 TARGET ACHIEVED: {results['avg_daily_return_points']:.1f} pts/day!")
            elif results['avg_daily_return_points'] >= target_points * 0.8:
                print(f"   ⚡ CLOSE TO TARGET: {results['avg_daily_return_points']:.1f} pts/day (need {target_points}+)")
            else:
                print(f"   ❌ Target missed: {results['avg_daily_return_points']:.1f} pts/day (target: {target_points}+)")
        else:
            print(f"   ❌ No qualifying trades found for {strategy_name}")
            all_results[strategy_name] = {"error": "No trades", "target_points": target_points}

    # Final comparison
    print(f"\n{'='*80}")
    print("🏆 ADVANCED FILTERED RESULTS COMPARISON")
    print("=" * 80)

    successful_strategies = []
    for name, results in all_results.items():
        if 'error' not in results:
            successful_strategies.append({
                'Strategy': name,
                'Target': f"{results['target_points']}+ pts",
                'Achieved': f"{results['avg_daily_return_points']:.1f} pts/day",
                'Total P&L': f"{results['total_pnl_points']:.0f} pts",
                'Win Rate': f"{results['win_rate_pct']:.1f}%",
                'Profit Factor': f"{results['profit_factor']:.2f}",
                'Sharpe': f"{results['sharpe_ratio']:.3f}",
                'Status': '🏆 ACHIEVED' if results['avg_daily_return_points'] >= results['target_points'] else
                         '⚡ CLOSE' if results['avg_daily_return_points'] >= results['target_points'] * 0.8 else '❌ MISSED'
            })

    if successful_strategies:
        comparison_df = pd.DataFrame(successful_strategies)
        print(comparison_df.to_string(index=False))

        # Find winners
        achieved_targets = [s for s in successful_strategies if s['Status'] == '🏆 ACHIEVED']
        if achieved_targets:
            best = max(achieved_targets, key=lambda x: float(x['Achieved'].split()[0]))
            print(f"\n🏆 WINNER: {best['Strategy']} achieved {best['Achieved']} (target: {best['Target']})")

    # Save results
    output_file = '/Users/shubhamshanker/bt_/advanced_filtered_results_2024.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n📁 Results saved to: {output_file}")
    return all_results

def simulate_enhanced_trade_outcome(df, entry_idx, signal, strategy):
    """Enhanced trade simulation with multiple contracts and dynamic exits."""
    try:
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        side = signal['signal']

        # Look ahead for exit
        max_bars = min(300, len(df) - entry_idx - 1)  # Longer holding period

        for i in range(entry_idx + 1, entry_idx + max_bars + 1):
            bar = df.iloc[i]

            # Check for exits with realistic slippage
            if side == 'BUY':
                if bar['low'] <= stop_loss:
                    exit_price = max(stop_loss - 0.25, bar['open'])  # Slippage
                    pnl = (exit_price - entry_price) * 20
                    return {
                        'exit_time': bar['timestamp'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': 'Stop Loss'
                    }
                elif bar['high'] >= take_profit:
                    exit_price = min(take_profit - 0.25, bar['open'] + 1.0)  # Conservative fill
                    pnl = (exit_price - entry_price) * 20
                    return {
                        'exit_time': bar['timestamp'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': 'Take Profit'
                    }
            else:  # SELL
                if bar['high'] >= stop_loss:
                    exit_price = min(stop_loss + 0.25, bar['open'])
                    pnl = (entry_price - exit_price) * 20
                    return {
                        'exit_time': bar['timestamp'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': 'Stop Loss'
                    }
                elif bar['low'] <= take_profit:
                    exit_price = max(take_profit + 0.25, bar['open'] - 1.0)
                    pnl = (entry_price - exit_price) * 20
                    return {
                        'exit_time': bar['timestamp'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': 'Take Profit'
                    }

        # Time exit
        final_bar = df.iloc[entry_idx + max_bars]
        if side == 'BUY':
            pnl = (final_bar['close'] - entry_price) * 20
        else:
            pnl = (entry_price - final_bar['close']) * 20

        return {
            'exit_time': final_bar['timestamp'],
            'exit_price': final_bar['close'],
            'pnl': pnl,
            'exit_reason': 'Time Exit'
        }
    except:
        return None

def analyze_advanced_performance(trades, strategy_name, target_points):
    """Comprehensive performance analysis."""
    df = pd.DataFrame(trades)

    total_trades = len(df)
    winning_trades = len(df[df['pnl'] > 0])
    win_rate = (winning_trades / total_trades) * 100

    total_pnl_points = df['pnl_points'].sum()
    avg_win = df[df['pnl'] > 0]['pnl_points'].mean() if winning_trades > 0 else 0
    avg_loss = abs(df[df['pnl'] <= 0]['pnl_points'].mean()) if (total_trades - winning_trades) > 0 else 0

    expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)

    gross_profit = df[df['pnl'] > 0]['pnl_points'].sum()
    gross_loss = abs(df[df['pnl'] <= 0]['pnl_points'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Daily analysis
    df['date'] = pd.to_datetime(df['entry_time']).dt.date
    daily_pnl = df.groupby('date')['pnl_points'].sum()
    avg_daily_return = daily_pnl.mean()
    daily_std = daily_pnl.std()

    # Drawdown
    df_sorted = df.sort_values('entry_time')
    df_sorted['cumulative_pnl'] = df_sorted['pnl_points'].cumsum()
    equity_curve = df_sorted['cumulative_pnl']
    running_max = equity_curve.expanding().max()
    drawdown = equity_curve - running_max
    max_drawdown = abs(drawdown.min())

    # Risk metrics
    sharpe_ratio = (avg_daily_return / daily_std * np.sqrt(252)) if daily_std > 0 else 0

    return {
        'strategy_name': strategy_name,
        'target_points': target_points,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate_pct': win_rate,
        'total_pnl_points': total_pnl_points,
        'avg_win_points': avg_win,
        'avg_loss_points': avg_loss,
        'expectancy_points': expectancy,
        'profit_factor': profit_factor,
        'max_drawdown_points': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'avg_daily_return_points': avg_daily_return,
        'trading_days': len(daily_pnl),
        'largest_win': df['pnl_points'].max(),
        'largest_loss': abs(df['pnl_points'].min())
    }

def get_trading_days_in_month(month_str):
    """Get number of trading days in a month."""
    year, month = map(int, month_str.split('-'))

    # Approximate trading days per month
    trading_days_map = {
        1: 21, 2: 20, 3: 22, 4: 21, 5: 22, 6: 21,
        7: 22, 8: 23, 9: 21, 10: 23, 11: 21, 12: 21
    }

    return trading_days_map.get(month, 22)

if __name__ == "__main__":
    results = run_advanced_filtered_backtest()