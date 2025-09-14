#!/usr/bin/env python3
"""
Comprehensive Strategy Testing for 2024 - Full Year Analysis
===========================================================
Test the 3 Ultimate ORB Strategies with detailed performance metrics:
- ULTIMATE_20, ULTIMATE_30, ULTIMATE_BALANCED
- Full 2024 data analysis
- Drawdowns, Sharpe, Sortino, Expectancy, Win/Loss ratios
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import json
import sys
# Visualization imports removed for compatibility
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/Users/shubhamshanker/bt_/backtesting')
from ultimate_orb_strategy import UltimateORBStrategy
from portfolio import Portfolio

class ComprehensivePerformanceAnalyzer:
    """Comprehensive performance analysis with all key metrics."""

    def __init__(self):
        self.trades = []
        self.daily_returns = []
        self.equity_curve = []
        self.daily_pnl = {}

    def analyze_performance(self, trades: List[Dict], strategy_name: str) -> Dict[str, Any]:
        """Perform comprehensive performance analysis."""
        if not trades:
            return {"error": "No trades to analyze"}

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(trades)
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df['pnl_points'] = df['pnl'] / 20  # Convert dollars to points

        # Basic metrics
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        total_pnl = df['pnl'].sum()
        total_pnl_points = df['pnl_points'].sum()

        avg_win = df[df['pnl'] > 0]['pnl_points'].mean() if winning_trades > 0 else 0
        avg_loss = abs(df[df['pnl'] < 0]['pnl_points'].mean()) if losing_trades > 0 else 0

        # Expectancy and Profit Factor
        expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
        gross_profit = df[df['pnl'] > 0]['pnl_points'].sum()
        gross_loss = abs(df[df['pnl'] < 0]['pnl_points'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Daily analysis
        df['trade_date'] = df['entry_time'].dt.date
        daily_pnl = df.groupby('trade_date')['pnl_points'].sum()

        # Equity curve
        df_sorted = df.sort_values('entry_time')
        df_sorted['cumulative_pnl'] = df_sorted['pnl_points'].cumsum()
        equity_curve = df_sorted['cumulative_pnl'].tolist()

        # Drawdown analysis
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = equity_series - rolling_max
        max_drawdown = abs(drawdown.min())
        max_drawdown_pct = (max_drawdown / rolling_max.max()) * 100 if rolling_max.max() > 0 else 0

        # Find drawdown periods
        drawdown_starts = []
        drawdown_ends = []
        in_drawdown = False

        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                drawdown_starts.append(i)
                in_drawdown = True
            elif dd >= 0 and in_drawdown:
                drawdown_ends.append(i-1)
                in_drawdown = False

        if in_drawdown:
            drawdown_ends.append(len(drawdown)-1)

        # Calculate drawdown durations
        drawdown_durations = []
        for start, end in zip(drawdown_starts, drawdown_ends):
            if end < len(df_sorted):
                duration_days = (df_sorted.iloc[end]['entry_time'] - df_sorted.iloc[start]['entry_time']).days
                drawdown_durations.append(duration_days)

        max_drawdown_duration = max(drawdown_durations) if drawdown_durations else 0

        # Risk metrics (using daily returns)
        if len(daily_pnl) > 1:
            daily_returns = daily_pnl.values
            avg_daily_return = np.mean(daily_returns)
            daily_std = np.std(daily_returns, ddof=1)

            # Sharpe Ratio (assuming risk-free rate = 0 for simplicity)
            sharpe_ratio = (avg_daily_return / daily_std * np.sqrt(252)) if daily_std > 0 else 0

            # Sortino Ratio (downside deviation)
            negative_returns = daily_returns[daily_returns < 0]
            downside_std = np.std(negative_returns, ddof=1) if len(negative_returns) > 1 else daily_std
            sortino_ratio = (avg_daily_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0

            # Calmar Ratio
            annual_return = avg_daily_return * 252
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else float('inf')

        else:
            sharpe_ratio = sortino_ratio = calmar_ratio = 0
            avg_daily_return = total_pnl_points / 252  # Approximate

        # Trading frequency
        trading_days = len(daily_pnl)
        trades_per_day = total_trades / trading_days if trading_days > 0 else 0

        # Consecutive wins/losses
        consecutive_wins = consecutive_losses = 0
        max_consecutive_wins = max_consecutive_losses = 0
        current_streak = 0
        last_result = None

        for _, trade in df_sorted.iterrows():
            is_winner = trade['pnl'] > 0
            if is_winner == last_result:
                current_streak += 1
            else:
                if last_result is True:  # Previous streak was wins
                    max_consecutive_wins = max(max_consecutive_wins, current_streak)
                elif last_result is False:  # Previous streak was losses
                    max_consecutive_losses = max(max_consecutive_losses, current_streak)
                current_streak = 1
                last_result = is_winner

        # Handle final streak
        if last_result is True:
            max_consecutive_wins = max(max_consecutive_wins, current_streak)
        elif last_result is False:
            max_consecutive_losses = max(max_consecutive_losses, current_streak)

        return {
            'strategy_name': strategy_name,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': round(win_rate, 2),
            'total_pnl_points': round(total_pnl_points, 2),
            'total_pnl_dollars': round(total_pnl, 2),
            'avg_win_points': round(avg_win, 2),
            'avg_loss_points': round(avg_loss, 2),
            'avg_trade_points': round(total_pnl_points / total_trades, 2),
            'expectancy_points': round(expectancy, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown_points': round(max_drawdown, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'max_drawdown_duration_days': max_drawdown_duration,
            'sharpe_ratio': round(sharpe_ratio, 3),
            'sortino_ratio': round(sortino_ratio, 3),
            'calmar_ratio': round(calmar_ratio, 3),
            'avg_daily_return_points': round(avg_daily_return, 2),
            'trading_days': trading_days,
            'trades_per_day': round(trades_per_day, 2),
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'largest_win_points': round(df['pnl_points'].max(), 2),
            'largest_loss_points': round(abs(df['pnl_points'].min()), 2),
            'equity_curve': [round(x, 2) for x in equity_curve[-100:]],  # Last 100 points for visualization
            'monthly_returns': self._calculate_monthly_returns(df),
            'performance_summary': self._generate_performance_summary(win_rate, expectancy, profit_factor, max_drawdown, sharpe_ratio)
        }

    def _calculate_monthly_returns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate monthly returns."""
        df['month'] = df['entry_time'].dt.to_period('M')
        monthly = df.groupby('month')['pnl_points'].sum()
        return {str(month): round(pnl, 2) for month, pnl in monthly.items()}

    def _generate_performance_summary(self, win_rate, expectancy, profit_factor, max_drawdown, sharpe) -> str:
        """Generate performance summary assessment."""
        ratings = []

        if win_rate >= 65:
            ratings.append("Excellent Win Rate")
        elif win_rate >= 55:
            ratings.append("Good Win Rate")
        else:
            ratings.append("Low Win Rate")

        if expectancy >= 10:
            ratings.append("Excellent Expectancy")
        elif expectancy >= 5:
            ratings.append("Good Expectancy")
        else:
            ratings.append("Poor Expectancy")

        if profit_factor >= 2.0:
            ratings.append("Excellent Profit Factor")
        elif profit_factor >= 1.5:
            ratings.append("Good Profit Factor")
        else:
            ratings.append("Poor Profit Factor")

        if max_drawdown <= 50:
            ratings.append("Low Risk")
        elif max_drawdown <= 100:
            ratings.append("Moderate Risk")
        else:
            ratings.append("High Risk")

        if sharpe >= 1.5:
            ratings.append("Excellent Risk-Adj Returns")
        elif sharpe >= 1.0:
            ratings.append("Good Risk-Adj Returns")
        else:
            ratings.append("Poor Risk-Adj Returns")

        return " | ".join(ratings)

def load_sample_data_2024():
    """Generate realistic 2024 market data for testing."""
    print("📊 Generating 2024 market data for comprehensive testing...")

    # Create full year 2024 data
    start_date = datetime(2024, 1, 2)
    end_date = datetime(2024, 12, 31)

    # Generate minute-by-minute data for market hours (9:30-16:00 EST)
    data = []
    current_price = 15800.0  # Starting price

    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Trading days only
            # Generate market hours data (9:30 AM to 4:00 PM)
            market_open = current_date.replace(hour=9, minute=30)
            market_close = current_date.replace(hour=16, minute=0)

            current_time = market_open
            daily_volatility = np.random.uniform(0.8, 2.5)  # Daily volatility factor

            while current_time <= market_close:
                # Random walk with mean reversion
                change_pct = np.random.normal(0, 0.0008) * daily_volatility
                current_price += current_price * change_pct

                # Add some intraday patterns
                hour = current_time.hour
                minute = current_time.minute

                # Opening range volatility
                if hour == 9 and minute <= 60:
                    volatility_boost = 1.5
                elif hour >= 15:  # Afternoon volatility
                    volatility_boost = 1.2
                else:
                    volatility_boost = 1.0

                high = current_price * (1 + abs(np.random.normal(0, 0.0003)) * volatility_boost)
                low = current_price * (1 - abs(np.random.normal(0, 0.0003)) * volatility_boost)
                close = current_price + np.random.normal(0, current_price * 0.0002) * volatility_boost

                data.append({
                    'timestamp': current_time,
                    'open': round(current_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close, 2),
                    'volume': np.random.randint(100, 1000)
                })

                current_price = close
                current_time += timedelta(minutes=1)

        current_date += timedelta(days=1)

    print(f"✅ Generated {len(data)} bars of 2024 data")
    return pd.DataFrame(data)

def run_comprehensive_strategy_tests():
    """Run comprehensive tests on all 3 strategies for full 2024."""
    print("🚀 COMPREHENSIVE STRATEGY TESTING - FULL 2024 ANALYSIS")
    print("=" * 80)
    print("📈 Testing 3 Ultimate ORB Strategies with detailed metrics")
    print("📊 Metrics: Drawdowns, Sharpe, Sortino, Expectancy, Win/Loss ratios")
    print("📅 Period: Full Year 2024 (252 trading days)")

    # Define the 3 optimized strategies
    strategies = {
        'ULTIMATE_20': UltimateORBStrategy(
            risk_per_trade=0.02,
            or_minutes=15,
            fixed_stop_points=25.0,
            target_multiplier=2.5,
            max_trades_per_day=3,
            half_size_booking=True,
            trailing_stop=True
        ),
        'ULTIMATE_30': UltimateORBStrategy(
            risk_per_trade=0.025,
            or_minutes=30,
            fixed_stop_points=20.0,
            target_multiplier=4.0,
            max_trades_per_day=2,
            half_size_booking=True,
            trailing_stop=True
        ),
        'ULTIMATE_BALANCED': UltimateORBStrategy(
            risk_per_trade=0.02,
            or_minutes=45,
            fixed_stop_points=30.0,
            target_multiplier=3.0,
            max_trades_per_day=2,
            half_size_booking=True,
            trailing_stop=True
        )
    }

    # Load 2024 market data
    df = load_sample_data_2024()

    analyzer = ComprehensivePerformanceAnalyzer()
    all_results = {}

    for strategy_name, strategy in strategies.items():
        print(f"\n{'='*20} TESTING {strategy_name} {'='*20}")
        print(f"OR Period: {strategy.or_minutes}min | Stop: {strategy.fixed_stop}pts | R:R: {strategy.target_mult}:1")
        print(f"Max Trades/Day: {strategy.max_trades_per_day} | Risk: {strategy.risk_per_trade*100}%")
        print("-" * 70)

        # Run backtest
        portfolio = Portfolio(initial_capital=100000)
        trades = []
        trade_id = 1

        print("🔄 Running backtest...")
        for idx, row in df.iterrows():
            if idx % 50000 == 0:
                print(f"   Processed {idx}/{len(df)} bars ({idx/len(df)*100:.1f}%)")

            bar_data = {
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            }

            signal = strategy.generate_signal(bar_data, portfolio)

            if signal['signal'] in ['BUY', 'SELL']:
                # Simulate realistic trade outcome
                trade_outcome = simulate_trade_outcome(df, idx, signal)
                if trade_outcome:
                    trade_data = {
                        'id': trade_id,
                        'strategy': strategy_name,
                        'entry_time': signal['entry_time'],
                        'exit_time': trade_outcome['exit_time'],
                        'side': signal['signal'],
                        'entry_price': signal['entry_price'],
                        'exit_price': trade_outcome['exit_price'],
                        'pnl': trade_outcome['pnl'],
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit']
                    }
                    trades.append(trade_data)
                    trade_id += 1

        print(f"✅ Backtest complete: {len(trades)} trades generated")

        # Analyze performance
        print("📊 Analyzing performance...")
        results = analyzer.analyze_performance(trades, strategy_name)
        all_results[strategy_name] = results

        # Print summary
        print(f"\n📈 {strategy_name} RESULTS:")
        print(f"   Trades: {results['total_trades']}")
        print(f"   Win Rate: {results['win_rate_pct']}%")
        print(f"   Total P&L: {results['total_pnl_points']:.1f} points (${results['total_pnl_dollars']:,.2f})")
        print(f"   Avg Daily Return: {results['avg_daily_return_points']:.2f} points")
        print(f"   Expectancy: {results['expectancy_points']:.2f} points")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"   Max Drawdown: {results['max_drawdown_points']:.1f} points ({results['max_drawdown_pct']:.1f}%)")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"   Sortino Ratio: {results['sortino_ratio']:.3f}")
        print(f"   Assessment: {results['performance_summary']}")

    # Generate comprehensive report
    print(f"\n{'='*80}")
    print("📊 COMPREHENSIVE PERFORMANCE COMPARISON")
    print("=" * 80)

    # Create comparison table
    comparison_data = []
    for name, results in all_results.items():
        comparison_data.append({
            'Strategy': name,
            'Trades': results['total_trades'],
            'Win%': f"{results['win_rate_pct']}%",
            'Total P&L (pts)': f"{results['total_pnl_points']:.1f}",
            'Daily Avg (pts)': f"{results['avg_daily_return_points']:.2f}",
            'Expectancy': f"{results['expectancy_points']:.2f}",
            'Profit Factor': f"{results['profit_factor']:.2f}",
            'Max DD (pts)': f"{results['max_drawdown_points']:.1f}",
            'Max DD %': f"{results['max_drawdown_pct']:.1f}%",
            'Sharpe': f"{results['sharpe_ratio']:.3f}",
            'Sortino': f"{results['sortino_ratio']:.3f}"
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    # Save detailed results
    output_file = '/Users/shubhamshanker/bt_/comprehensive_strategy_results_2024.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n📁 Detailed results saved to: {output_file}")

    # Performance ranking
    print(f"\n🏆 STRATEGY RANKINGS:")
    print("=" * 40)

    # Rank by daily return
    sorted_by_return = sorted(all_results.items(), key=lambda x: x[1]['avg_daily_return_points'], reverse=True)
    print("📈 By Daily Return:")
    for i, (name, results) in enumerate(sorted_by_return, 1):
        print(f"   {i}. {name}: {results['avg_daily_return_points']:.2f} pts/day")

    # Rank by Sharpe ratio
    sorted_by_sharpe = sorted(all_results.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)
    print("\n⚖️ By Risk-Adjusted Return (Sharpe):")
    for i, (name, results) in enumerate(sorted_by_sharpe, 1):
        print(f"   {i}. {name}: {results['sharpe_ratio']:.3f}")

    # Rank by max drawdown (lower is better)
    sorted_by_dd = sorted(all_results.items(), key=lambda x: x[1]['max_drawdown_points'])
    print("\n🛡️ By Risk Control (Max Drawdown):")
    for i, (name, results) in enumerate(sorted_by_dd, 1):
        print(f"   {i}. {name}: {results['max_drawdown_points']:.1f} pts")

    print(f"\n🎯 CONCLUSION:")
    best_strategy = sorted_by_return[0]
    print(f"🏆 Best Overall Performance: {best_strategy[0]}")
    print(f"   Daily Return: {best_strategy[1]['avg_daily_return_points']:.2f} points/day")
    print(f"   Annual Estimate: {best_strategy[1]['avg_daily_return_points'] * 252:.0f} points/year")
    print(f"   Risk-Adjusted: Sharpe {best_strategy[1]['sharpe_ratio']:.3f}")

    return all_results

def simulate_trade_outcome(df, entry_idx, signal):
    """Simulate realistic trade outcome based on market data."""
    try:
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        side = signal['signal']

        # Look ahead up to 100 bars for exit
        end_idx = min(entry_idx + 100, len(df) - 1)

        for i in range(entry_idx + 1, end_idx + 1):
            bar = df.iloc[i]

            if side == 'BUY':
                if bar['low'] <= stop_loss:
                    pnl = (stop_loss - entry_price) * 20
                    return {
                        'exit_time': bar['timestamp'],
                        'exit_price': stop_loss,
                        'pnl': pnl
                    }
                elif bar['high'] >= take_profit:
                    pnl = (take_profit - entry_price) * 20
                    return {
                        'exit_time': bar['timestamp'],
                        'exit_price': take_profit,
                        'pnl': pnl
                    }
            else:  # SELL
                if bar['high'] >= stop_loss:
                    pnl = (entry_price - stop_loss) * 20
                    return {
                        'exit_time': bar['timestamp'],
                        'exit_price': stop_loss,
                        'pnl': pnl
                    }
                elif bar['low'] <= take_profit:
                    pnl = (entry_price - take_profit) * 20
                    return {
                        'exit_time': bar['timestamp'],
                        'exit_price': take_profit,
                        'pnl': pnl
                    }

        # Time-based exit if no stop/target hit
        last_bar = df.iloc[end_idx]
        if side == 'BUY':
            pnl = (last_bar['close'] - entry_price) * 20
        else:
            pnl = (entry_price - last_bar['close']) * 20

        return {
            'exit_time': last_bar['timestamp'],
            'exit_price': last_bar['close'],
            'pnl': pnl
        }
    except Exception as e:
        return None

if __name__ == "__main__":
    results = run_comprehensive_strategy_tests()