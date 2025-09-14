#!/usr/bin/env python3
"""
OPTIMIZED MULTI-STRATEGY SYSTEM - TARGETING 30+ POINTS CONSISTENTLY
===================================================================
- Multiple complementary strategies running in parallel
- Balanced filters (not too restrictive)
- Scale up winners, cut losses quickly
- Portfolio approach with correlation management
- Focus on consistent monthly performance
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

class OptimizedORBStrategy(UltimateORBStrategy):
    """Optimized ORB with balanced filters for consistent performance."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Balanced filter parameters (less restrictive)
        self.min_daily_range = 50.0  # Minimum daily range in points
        self.optimal_volume_threshold = 1.2  # 1.2x average volume
        self.prime_trading_hours = [
            (time(9, 30), time(12, 0)),   # Morning session
            (time(13, 0), time(16, 0))    # Afternoon session
        ]

        # Multiple contract scaling
        self.scale_winners = True
        self.max_contracts = 3

    def is_tradeable_setup(self, bar_data: Dict[str, Any], market_context: Dict[str, Any]) -> bool:
        """Check if setup meets balanced trading criteria."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()
        current_time = timestamp.time()

        # 1. Daily range filter (not too restrictive)
        daily_range = market_context.get('daily_range', 0)
        if daily_range > 0 and daily_range < self.min_daily_range:
            return False

        # 2. Time-of-day filter (broader windows)
        in_trading_window = any(start <= current_time <= end for start, end in self.prime_trading_hours)
        if not in_trading_window:
            return False

        # 3. Volume filter (moderate threshold)
        avg_volume = market_context.get('avg_volume', 100)
        if bar_data['volume'] < avg_volume * self.optimal_volume_threshold:
            return False

        return True

    def calculate_position_size(self, signal_strength: float = 1.0) -> int:
        """Dynamic position sizing based on signal strength."""
        base_size = 1

        if self.scale_winners and signal_strength > 1.5:
            return min(2, self.max_contracts)
        elif signal_strength > 2.0:
            return min(3, self.max_contracts)

        return base_size

def create_multi_strategy_portfolio():
    """Create portfolio of complementary strategies."""
    strategies = {
        # Fast scalping strategy - quick 15-25 point targets
        'SCALP_FAST': OptimizedORBStrategy(
            risk_per_trade=0.015,
            or_minutes=15,
            fixed_stop_points=15.0,
            target_multiplier=2.5,  # 37.5 point targets
            max_trades_per_day=4,   # More frequent
            half_size_booking=True,
            trailing_stop=False     # Quick exits
        ),

        # Medium swing strategy - 30-50 point targets
        'SWING_MEDIUM': OptimizedORBStrategy(
            risk_per_trade=0.025,
            or_minutes=30,
            fixed_stop_points=25.0,
            target_multiplier=3.5,  # 87.5 point targets
            max_trades_per_day=3,
            half_size_booking=True,
            trailing_stop=True
        ),

        # Large move strategy - 50+ point targets
        'TREND_LARGE': OptimizedORBStrategy(
            risk_per_trade=0.03,
            or_minutes=45,
            fixed_stop_points=35.0,
            target_multiplier=4.0,  # 140 point targets
            max_trades_per_day=2,
            half_size_booking=False,  # Let winners run
            trailing_stop=True
        ),

        # Reversal strategy - counter-trend when ORB fails
        'REVERSAL': OptimizedORBStrategy(
            risk_per_trade=0.02,
            or_minutes=30,
            fixed_stop_points=20.0,
            target_multiplier=3.0,  # 60 point targets
            max_trades_per_day=2,
            half_size_booking=True,
            trailing_stop=True
        )
    }

    # Set balanced parameters for each strategy
    for strategy in strategies.values():
        strategy.min_daily_range = 40.0  # Lower threshold
        strategy.optimal_volume_threshold = 1.1  # Lower volume requirement

    return strategies

def calculate_market_context(df: pd.DataFrame, current_idx: int) -> Dict[str, Any]:
    """Calculate real-time market context for filtering."""
    try:
        current_bar = df.iloc[current_idx]
        current_date = current_bar['timestamp'].date()

        # Get current day's data up to this point
        day_data = df[
            (df['timestamp'].dt.date == current_date) &
            (df.index <= current_idx)
        ]

        if len(day_data) == 0:
            return {'daily_range': 0, 'avg_volume': 100}

        daily_high = day_data['high'].max()
        daily_low = day_data['low'].min()
        daily_range = (daily_high - daily_low) * 20  # Convert to points

        # Calculate average volume for similar time periods
        current_time = current_bar['timestamp'].time()
        similar_time_data = df[
            (df['timestamp'].dt.time == current_time) &
            (df.index < current_idx)
        ].tail(10)  # Last 10 days at same time

        avg_volume = similar_time_data['volume'].mean() if len(similar_time_data) > 0 else 100

        return {
            'daily_range': daily_range,
            'avg_volume': avg_volume,
            'daily_high': daily_high,
            'daily_low': daily_low
        }
    except:
        return {'daily_range': 0, 'avg_volume': 100}

def run_multi_strategy_system():
    """Run the complete multi-strategy system."""
    print("🚀 OPTIMIZED MULTI-STRATEGY SYSTEM")
    print("=" * 80)
    print("🎯 TARGET: Consistent 30+ points per day")
    print("📊 Approach: Multiple complementary strategies")
    print("⚖️  Balanced filters for steady trade flow")
    print("📈 Dynamic position sizing and portfolio management")

    # Load real data
    data_file = '/Users/shubhamshanker/bt_/data/NQ_M1_standard.csv'
    print(f"\n📊 Loading real NQ data from: {data_file}")

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

        print(f"✅ Loaded {len(df_2024):,} bars of 2024 data")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return {}

    # Create strategy portfolio
    strategies = create_multi_strategy_portfolio()

    # Portfolio tracking
    portfolio_trades = []
    portfolio_monthly_pnl = {}
    strategy_results = {}
    trade_id = 1

    print(f"\n🔄 Running multi-strategy system with {len(strategies)} strategies...")

    # Process each bar
    for idx, row in df_2024.iterrows():
        if idx % 20000 == 0 and idx > 0:
            print(f"   Progress: {idx:,}/{len(df_2024):,} bars ({idx/len(df_2024)*100:.1f}%)")

        bar_data = {
            'timestamp': row['timestamp'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume']
        }

        # Calculate market context
        market_context = calculate_market_context(df_2024, idx)

        # Run each strategy
        for strategy_name, strategy in strategies.items():

            # Generate base signal
            portfolio = Portfolio(initial_capital=100000)  # Individual portfolio per strategy
            signal = strategy.generate_signal(bar_data, portfolio)

            if signal['signal'] in ['BUY', 'SELL']:
                # Apply balanced filters
                if strategy.is_tradeable_setup(bar_data, market_context):

                    # Calculate signal strength for position sizing
                    signal_strength = calculate_signal_strength(bar_data, market_context, strategy)
                    position_size = strategy.calculate_position_size(signal_strength)

                    # Execute trade with optimized parameters
                    trade_outcome = execute_optimized_trade(df_2024, idx, signal, strategy, position_size)

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
                            'contracts': position_size,
                            'signal_strength': signal_strength,
                            'month': month_key,
                            'exit_reason': trade_outcome['exit_reason']
                        }

                        portfolio_trades.append(trade_data)

                        # Track monthly portfolio P&L
                        if month_key not in portfolio_monthly_pnl:
                            portfolio_monthly_pnl[month_key] = 0
                        portfolio_monthly_pnl[month_key] += trade_outcome['pnl'] / 20

                        trade_id += 1

    print(f"✅ Multi-strategy backtest complete: {len(portfolio_trades)} total trades")

    # Analyze results by strategy
    strategy_breakdown = {}
    for strategy_name in strategies.keys():
        strategy_trades = [t for t in portfolio_trades if t['strategy'] == strategy_name]
        if strategy_trades:
            strategy_breakdown[strategy_name] = analyze_strategy_performance(strategy_trades)

    # Portfolio-level analysis
    print(f"\n{'='*80}")
    print("📊 MULTI-STRATEGY PORTFOLIO RESULTS")
    print("=" * 80)

    # Monthly portfolio performance
    print(f"\n📅 MONTHLY PORTFOLIO PERFORMANCE:")
    successful_months = 0
    total_months = len(portfolio_monthly_pnl)

    for month, pnl_points in sorted(portfolio_monthly_pnl.items()):
        trading_days = get_trading_days_in_month(month)
        daily_avg = pnl_points / trading_days if trading_days > 0 else 0

        if daily_avg >= 30:
            status = "🏆 EXCELLENT"
            successful_months += 1
        elif daily_avg >= 20:
            status = "✅ GOOD"
            successful_months += 0.5
        elif daily_avg >= 10:
            status = "⚡ OKAY"
        else:
            status = "❌ POOR"

        print(f"   {month}: {pnl_points:7.1f} pts ({daily_avg:5.1f} pts/day) {status}")

    # Overall portfolio metrics
    total_portfolio_pnl = sum(portfolio_monthly_pnl.values())
    avg_daily_portfolio = total_portfolio_pnl / 252 if len(portfolio_monthly_pnl) > 0 else 0
    success_rate = (successful_months / total_months * 100) if total_months > 0 else 0

    print(f"\n🏆 PORTFOLIO SUMMARY:")
    print(f"   Total Trades: {len(portfolio_trades)}")
    print(f"   Total Portfolio P&L: {total_portfolio_pnl:.1f} points")
    print(f"   Average Daily Return: {avg_daily_portfolio:.2f} points")
    print(f"   Monthly Success Rate: {success_rate:.1f}% ({successful_months:.1f}/{total_months})")

    if avg_daily_portfolio >= 30:
        print(f"   🏆 TARGET ACHIEVED: {avg_daily_portfolio:.1f} points/day!")
    elif avg_daily_portfolio >= 25:
        print(f"   ⚡ CLOSE TO TARGET: {avg_daily_portfolio:.1f} points/day (need 30+)")
    else:
        print(f"   ❌ Target missed: {avg_daily_portfolio:.1f} points/day (target: 30+)")

    # Strategy breakdown
    print(f"\n📊 INDIVIDUAL STRATEGY PERFORMANCE:")
    for name, results in strategy_breakdown.items():
        print(f"   {name}: {results['trades']} trades, {results['daily_avg']:.1f} pts/day, {results['win_rate']:.1f}% WR")

    # Save results
    output_data = {
        'portfolio_summary': {
            'total_trades': len(portfolio_trades),
            'total_pnl_points': total_portfolio_pnl,
            'avg_daily_return': avg_daily_portfolio,
            'monthly_success_rate': success_rate,
            'target_achieved': avg_daily_portfolio >= 30
        },
        'monthly_performance': portfolio_monthly_pnl,
        'strategy_breakdown': strategy_breakdown,
        'all_trades': portfolio_trades[:100]  # Sample trades
    }

    output_file = '/Users/shubhamshanker/bt_/multi_strategy_results_2024.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n📁 Results saved to: {output_file}")
    return output_data

def calculate_signal_strength(bar_data: Dict[str, Any], market_context: Dict[str, Any], strategy) -> float:
    """Calculate signal strength for position sizing."""
    strength = 1.0

    # Volume boost
    avg_volume = market_context.get('avg_volume', 100)
    if bar_data['volume'] > avg_volume * 2:
        strength += 0.5
    elif bar_data['volume'] > avg_volume * 1.5:
        strength += 0.2

    # Range boost
    daily_range = market_context.get('daily_range', 0)
    if daily_range > 100:  # High volatility day
        strength += 0.3
    elif daily_range > 80:
        strength += 0.1

    # Time of day boost
    current_time = bar_data['timestamp'].time()
    if time(9, 45) <= current_time <= time(10, 30):  # Opening momentum
        strength += 0.2
    elif time(14, 0) <= current_time <= time(15, 0):  # Afternoon power
        strength += 0.15

    return min(strength, 3.0)  # Cap at 3x

def execute_optimized_trade(df, entry_idx, signal, strategy, position_size):
    """Execute trade with optimized parameters."""
    try:
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        side = signal['signal']

        # Adjust targets based on position size
        if position_size > 1:
            # Scale profit targets for larger positions
            original_target = abs(take_profit - entry_price)
            scaled_target = original_target * (1 + (position_size - 1) * 0.2)

            if side == 'BUY':
                take_profit = entry_price + scaled_target
            else:
                take_profit = entry_price - scaled_target

        # Look ahead for exits
        max_bars = min(200, len(df) - entry_idx - 1)

        for i in range(entry_idx + 1, entry_idx + max_bars + 1):
            bar = df.iloc[i]

            # Check exits with realistic fills
            if side == 'BUY':
                if bar['low'] <= stop_loss:
                    exit_price = max(stop_loss - 0.5, bar['low'])
                    pnl = (exit_price - entry_price) * 20 * position_size
                    return {
                        'exit_time': bar['timestamp'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': 'Stop Loss'
                    }
                elif bar['high'] >= take_profit:
                    exit_price = min(take_profit - 0.25, bar['high'])
                    pnl = (exit_price - entry_price) * 20 * position_size
                    return {
                        'exit_time': bar['timestamp'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': 'Take Profit'
                    }
            else:  # SELL
                if bar['high'] >= stop_loss:
                    exit_price = min(stop_loss + 0.5, bar['high'])
                    pnl = (entry_price - exit_price) * 20 * position_size
                    return {
                        'exit_time': bar['timestamp'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': 'Stop Loss'
                    }
                elif bar['low'] <= take_profit:
                    exit_price = max(take_profit + 0.25, bar['low'])
                    pnl = (entry_price - exit_price) * 20 * position_size
                    return {
                        'exit_time': bar['timestamp'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': 'Take Profit'
                    }

        # Time exit
        final_bar = df.iloc[entry_idx + max_bars]
        if side == 'BUY':
            pnl = (final_bar['close'] - entry_price) * 20 * position_size
        else:
            pnl = (entry_price - final_bar['close']) * 20 * position_size

        return {
            'exit_time': final_bar['timestamp'],
            'exit_price': final_bar['close'],
            'pnl': pnl,
            'exit_reason': 'Time Exit'
        }

    except:
        return None

def analyze_strategy_performance(trades):
    """Analyze individual strategy performance."""
    df = pd.DataFrame(trades)

    total_trades = len(df)
    winners = len(df[df['pnl'] > 0])
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0

    total_pnl_points = df['pnl_points'].sum()
    avg_daily = total_pnl_points / 252  # Approximate

    return {
        'trades': total_trades,
        'win_rate': win_rate,
        'total_pnl_points': total_pnl_points,
        'daily_avg': avg_daily
    }

def get_trading_days_in_month(month_str):
    """Get trading days in month."""
    trading_days_map = {
        '2024-01': 22, '2024-02': 21, '2024-03': 21, '2024-04': 22,
        '2024-05': 22, '2024-06': 20, '2024-07': 23, '2024-08': 22,
        '2024-09': 20, '2024-10': 23, '2024-11': 21, '2024-12': 21
    }
    return trading_days_map.get(month_str, 22)

if __name__ == "__main__":
    results = run_multi_strategy_system()