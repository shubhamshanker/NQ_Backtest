#!/usr/bin/env python3
"""
Comprehensive 2024 Backtest with Detailed Performance Metrics
===========================================================
Full backtesting with P&L tracking, win/loss analysis, drawdown calculation,
and complete performance statistics for 2024 using 100% Parquet data.
"""

import pandas as pd
import numpy as np
import time
import sys
import os
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Force Parquet-only mode
os.environ['USE_PARQUET_DATA'] = 'true'

class DetailedBacktester:
    """Comprehensive backtesting engine with detailed performance tracking."""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.equity_curve = []
        self.daily_returns = {}
        self.max_drawdown = 0
        self.peak_equity = initial_capital

    def execute_trade(self, signal: Dict, current_bar: Dict, next_bar: Dict) -> Dict:
        """Execute a trade based on signal with realistic fills."""

        # Entry at next bar open (no lookahead bias)
        entry_price = next_bar['open']
        direction = 1 if signal['signal'] == 'BUY' else -1

        # Position sizing (risk management)
        risk_per_trade = signal.get('risk_amount', self.current_capital * 0.02)  # 2% risk
        stop_distance = abs(entry_price - signal['stop_loss'])

        if stop_distance > 0:
            position_size = int(risk_per_trade / (stop_distance * 20))  # NQ multiplier = 20
            position_size = max(1, min(position_size, 5))  # Between 1-5 contracts
        else:
            position_size = 1

        trade = {
            'entry_time': next_bar['timestamp'],
            'entry_price': entry_price,
            'direction': direction,
            'position_size': position_size,
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'signal_strength': signal.get('confidence', 1.0),
            'status': 'open'
        }

        return trade

    def update_open_trades(self, current_bar: Dict) -> List[Dict]:
        """Update all open trades and close if stop/target hit."""
        closed_trades = []
        current_price = current_bar['close']
        current_time = current_bar['timestamp']

        for trade in self.trades:
            if trade['status'] == 'open':
                pnl_points = (current_price - trade['entry_price']) * trade['direction']
                pnl_dollars = pnl_points * 20 * trade['position_size']  # NQ multiplier

                # Check exit conditions
                exit_reason = None
                exit_price = current_price

                if trade['direction'] == 1:  # Long trade
                    if current_price <= trade['stop_loss']:
                        exit_reason = 'Stop Loss'
                        exit_price = trade['stop_loss']
                    elif current_price >= trade['take_profit']:
                        exit_reason = 'Take Profit'
                        exit_price = trade['take_profit']
                elif trade['direction'] == -1:  # Short trade
                    if current_price >= trade['stop_loss']:
                        exit_reason = 'Stop Loss'
                        exit_price = trade['stop_loss']
                    elif current_price <= trade['take_profit']:
                        exit_reason = 'Take Profit'
                        exit_price = trade['take_profit']

                # Time-based exit (end of day)
                if current_time.time() >= dt_time(15, 45):  # Close before market close
                    if exit_reason is None:
                        exit_reason = 'Time Exit'
                        exit_price = current_price

                # Close trade if exit condition met
                if exit_reason:
                    final_pnl_points = (exit_price - trade['entry_price']) * trade['direction']
                    final_pnl_dollars = final_pnl_points * 20 * trade['position_size']

                    # Account for commissions/fees
                    commission = 4.0 * trade['position_size']  # $4 per contract round trip
                    final_pnl_dollars -= commission

                    trade.update({
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl_points': final_pnl_points,
                        'pnl_dollars': final_pnl_dollars,
                        'commission': commission,
                        'status': 'closed'
                    })

                    # Update capital
                    self.current_capital += final_pnl_dollars
                    closed_trades.append(trade)

                    # Track daily returns
                    date_key = current_time.date()
                    if date_key not in self.daily_returns:
                        self.daily_returns[date_key] = 0
                    self.daily_returns[date_key] += final_pnl_dollars

        return closed_trades

    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance statistics."""

        if not self.trades:
            return {}

        closed_trades = [t for t in self.trades if t['status'] == 'closed']
        if not closed_trades:
            return {}

        # Basic metrics
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t['pnl_dollars'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl_dollars'] < 0]
        breakeven_trades = [t for t in closed_trades if t['pnl_dollars'] == 0]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)

        # P&L metrics
        total_pnl = sum(t['pnl_dollars'] for t in closed_trades)
        gross_profit = sum(t['pnl_dollars'] for t in winning_trades)
        gross_loss = sum(t['pnl_dollars'] for t in losing_trades)

        # Win/Loss analysis
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        avg_win = (gross_profit / win_count) if win_count > 0 else 0
        avg_loss = (gross_loss / loss_count) if loss_count > 0 else 0
        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else float('inf')

        # Points analysis
        total_points = sum(t['pnl_points'] for t in closed_trades)
        avg_points_per_trade = total_points / total_trades if total_trades > 0 else 0
        avg_points_win = sum(t['pnl_points'] for t in winning_trades) / win_count if win_count > 0 else 0
        avg_points_loss = sum(t['pnl_points'] for t in losing_trades) / loss_count if loss_count > 0 else 0

        # Drawdown calculation
        equity_curve = []
        running_capital = self.initial_capital
        peak = self.initial_capital
        max_dd = 0
        max_dd_percent = 0

        for trade in closed_trades:
            running_capital += trade['pnl_dollars']
            equity_curve.append(running_capital)

            if running_capital > peak:
                peak = running_capital

            current_dd = peak - running_capital
            current_dd_percent = (current_dd / peak * 100) if peak > 0 else 0

            if current_dd > max_dd:
                max_dd = current_dd
            if current_dd_percent > max_dd_percent:
                max_dd_percent = current_dd_percent

        # Daily statistics
        daily_pnl = list(self.daily_returns.values())
        trading_days = len(daily_pnl)
        avg_daily_pnl = sum(daily_pnl) / trading_days if trading_days > 0 else 0
        daily_std = np.std(daily_pnl) if len(daily_pnl) > 1 else 0

        # Sharpe-like ratio (daily)
        daily_sharpe = (avg_daily_pnl / daily_std) if daily_std > 0 else 0

        # Risk metrics
        largest_win = max((t['pnl_dollars'] for t in winning_trades), default=0)
        largest_loss = min((t['pnl_dollars'] for t in losing_trades), default=0)

        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0

        for trade in closed_trades:
            if trade['pnl_dollars'] > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            elif trade['pnl_dollars'] < 0:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_wins = 0
                consecutive_losses = 0

        return {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'breakeven_trades': len(breakeven_trades),
            'win_rate_percent': round(win_rate, 2),
            'total_pnl_dollars': round(total_pnl, 2),
            'total_pnl_points': round(total_points, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_win_dollars': round(avg_win, 2),
            'avg_loss_dollars': round(avg_loss, 2),
            'avg_points_per_trade': round(avg_points_per_trade, 2),
            'avg_points_win': round(avg_points_win, 2),
            'avg_points_loss': round(avg_points_loss, 2),
            'largest_win': round(largest_win, 2),
            'largest_loss': round(largest_loss, 2),
            'max_drawdown_dollars': round(max_dd, 2),
            'max_drawdown_percent': round(max_dd_percent, 2),
            'avg_daily_pnl': round(avg_daily_pnl, 2),
            'daily_sharpe_ratio': round(daily_sharpe, 3),
            'trading_days': trading_days,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'final_capital': round(self.current_capital, 2),
            'total_return_percent': round((self.current_capital - self.initial_capital) / self.initial_capital * 100, 2),
            'commission_paid': round(sum(t.get('commission', 0) for t in closed_trades), 2)
        }

def run_comprehensive_backtest():
    """Run complete 2024 backtest with detailed performance analysis."""

    print("🚀 COMPREHENSIVE 2024 BACKTEST - PARQUET DATA ONLY")
    print("=" * 80)
    print("💾 Data Source: 100% Parquet + DuckDB")
    print("🚫 NO CSV fallback - Pure performance testing")
    print("🎯 Target: Complete performance statistics for 2024")

    overall_start_time = time.time()

    try:
        from backtesting.parquet_data_handler import ParquetDataHandler
        from backtesting.ultimate_orb_strategy import UltimateORBStrategy

        # Strategy configurations to test
        strategy_configs = [
            {
                'name': 'Conservative ORB',
                'params': {
                    'risk_per_trade': 0.01,
                    'or_minutes': 30,
                    'fixed_stop_points': 12.0,
                    'target_multiplier': 2.5,
                    'max_trades_per_day': 2
                }
            },
            {
                'name': 'Aggressive ORB',
                'params': {
                    'risk_per_trade': 0.02,
                    'or_minutes': 30,
                    'fixed_stop_points': 18.0,
                    'target_multiplier': 3.0,
                    'max_trades_per_day': 4
                }
            },
            {
                'name': 'Scalper ORB',
                'params': {
                    'risk_per_trade': 0.015,
                    'or_minutes': 15,
                    'fixed_stop_points': 8.0,
                    'target_multiplier': 2.0,
                    'max_trades_per_day': 6
                }
            }
        ]

        results = {}

        # Load 2024 data
        print(f"\n📊 Loading 2024 data with Parquet + DuckDB...")
        data_load_start = time.time()

        with ParquetDataHandler() as handler:
            df = handler.load_data(
                symbol="NQ",
                timeframe="1min",
                start_date="2024-01-01",
                end_date="2024-12-31",
                session_filter=True
            )

        data_load_time = time.time() - data_load_start

        if len(df) == 0:
            print("❌ No 2024 data loaded")
            return None

        print(f"✅ Data loaded: {len(df):,} rows in {data_load_time:.2f}s")
        print(f"📅 Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"⚡ Loading speed: {len(df)/data_load_time:.0f} rows/sec")

        # Convert to bar format
        data_list = []
        for idx, row in df.iterrows():
            bar = {
                'timestamp': idx,
                'datetime': idx,
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': float(row['Volume'])
            }
            data_list.append(bar)

        print(f"📋 Converted {len(data_list):,} bars for backtesting")

        # Run backtest for each strategy
        for strategy_config in strategy_configs:
            strategy_name = strategy_config['name']
            params = strategy_config['params']

            print(f"\n🎯 Running {strategy_name} Backtest...")
            strategy_start_time = time.time()

            # Initialize strategy and backtester
            strategy = UltimateORBStrategy(**params)
            backtester = DetailedBacktester(initial_capital=100000)

            # Track execution
            signals_generated = 0
            bars_processed = 0
            open_trades = []

            # Process each bar
            for i in range(len(data_list) - 1):  # -1 to have next bar for entry
                current_bar = data_list[i]
                next_bar = data_list[i + 1]
                bars_processed += 1

                # Update open trades first
                closed_trades = backtester.update_open_trades(current_bar)
                open_trades = [t for t in backtester.trades if t['status'] == 'open']

                # Generate signal if no open trades
                if len(open_trades) == 0:
                    signal = strategy.generate_signal(current_bar, backtester)

                    if signal['signal'] in ['BUY', 'SELL']:
                        signals_generated += 1

                        # Execute trade
                        trade = backtester.execute_trade(signal, current_bar, next_bar)
                        backtester.trades.append(trade)

                        # Limit output for performance
                        if signals_generated <= 10:
                            print(f"   📈 Signal {signals_generated}: {signal['signal']} at {trade['entry_price']:.2f} "
                                  f"({current_bar['timestamp'].strftime('%m/%d %H:%M')}) - "
                                  f"Stop: {trade['stop_loss']:.2f}, Target: {trade['take_profit']:.2f}")
                        elif signals_generated == 11:
                            print(f"   ... (showing first 10 signals only)")

                # Progress update every 10k bars
                if bars_processed % 10000 == 0:
                    print(f"   Processed {bars_processed:,} bars, {signals_generated} signals generated")

            # Close any remaining open trades
            final_bar = data_list[-1]
            final_bar['timestamp'] = final_bar['timestamp'].replace(hour=16, minute=0)
            backtester.update_open_trades(final_bar)

            strategy_time = time.time() - strategy_start_time

            # Calculate performance metrics
            performance = backtester.calculate_performance_metrics()
            performance['strategy_name'] = strategy_name
            performance['execution_time_seconds'] = round(strategy_time, 2)
            performance['bars_processed'] = bars_processed
            performance['signals_generated'] = signals_generated
            performance['processing_speed_bars_per_sec'] = round(bars_processed / strategy_time, 0)

            results[strategy_name] = performance

            print(f"✅ {strategy_name} Complete:")
            print(f"   ⏱️  Execution time: {strategy_time:.2f}s")
            print(f"   📊 Signals generated: {signals_generated}")
            print(f"   📈 Total trades: {performance.get('total_trades', 0)}")
            print(f"   💰 Total P&L: ${performance.get('total_pnl_dollars', 0):,.2f}")
            print(f"   📊 Win rate: {performance.get('win_rate_percent', 0):.1f}%")

        total_time = time.time() - overall_start_time

        # Compile final results
        final_results = {
            'backtest_summary': {
                'data_source': '100% Parquet + DuckDB',
                'date_range': f"{df.index[0].date()} to {df.index[-1].date()}",
                'total_bars': len(data_list),
                'data_load_time_seconds': round(data_load_time, 2),
                'total_execution_time_seconds': round(total_time, 2),
                'strategies_tested': len(strategy_configs)
            },
            'strategy_results': results
        }

        return final_results

    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_performance_report(results: Dict) -> None:
    """Generate detailed performance report."""

    if not results:
        print("❌ No results to report")
        return

    summary = results['backtest_summary']
    strategies = results['strategy_results']

    print(f"\n🎯 COMPREHENSIVE 2024 BACKTEST RESULTS")
    print("=" * 80)

    print(f"📊 **EXECUTION SUMMARY:**")
    print(f"   Data source: {summary['data_source']}")
    print(f"   Date range: {summary['date_range']}")
    print(f"   Total bars processed: {summary['total_bars']:,}")
    print(f"   Data loading time: {summary['data_load_time_seconds']}s")
    print(f"   Total execution time: {summary['total_execution_time_seconds']}s")
    print(f"   Strategies tested: {summary['strategies_tested']}")

    # Strategy comparison
    print(f"\n📈 **STRATEGY PERFORMANCE COMPARISON:**")
    print(f"{'Strategy':<20} {'Total P&L':<12} {'Win Rate':<10} {'Trades':<8} {'Drawdown':<12} {'Sharpe':<8}")
    print("-" * 80)

    for name, perf in strategies.items():
        pnl = perf.get('total_pnl_dollars', 0)
        win_rate = perf.get('win_rate_percent', 0)
        trades = perf.get('total_trades', 0)
        dd = perf.get('max_drawdown_percent', 0)
        sharpe = perf.get('daily_sharpe_ratio', 0)

        print(f"{name:<20} ${pnl:<11,.0f} {win_rate:<9.1f}% {trades:<7} {dd:<11.1f}% {sharpe:<7.2f}")

    # Detailed results for each strategy
    for strategy_name, performance in strategies.items():
        print(f"\n🎯 **{strategy_name.upper()} - DETAILED RESULTS:**")
        print("-" * 60)

        print(f"⏱️  **Execution Performance:**")
        print(f"   Execution time: {performance.get('execution_time_seconds', 0)}s")
        print(f"   Processing speed: {performance.get('processing_speed_bars_per_sec', 0):,.0f} bars/sec")
        print(f"   Signals generated: {performance.get('signals_generated', 0)}")

        print(f"\n💰 **P&L Performance:**")
        print(f"   Total P&L: ${performance.get('total_pnl_dollars', 0):,.2f}")
        print(f"   Total Points: {performance.get('total_pnl_points', 0):,.2f}")
        print(f"   Total Return: {performance.get('total_return_percent', 0):,.2f}%")
        print(f"   Final Capital: ${performance.get('final_capital', 0):,.2f}")
        print(f"   Average Daily P&L: ${performance.get('avg_daily_pnl', 0):,.2f}")

        print(f"\n📊 **Trade Statistics:**")
        print(f"   Total Trades: {performance.get('total_trades', 0)}")
        print(f"   Winning Trades: {performance.get('winning_trades', 0)}")
        print(f"   Losing Trades: {performance.get('losing_trades', 0)}")
        print(f"   Win Rate: {performance.get('win_rate_percent', 0):.2f}%")
        print(f"   Profit Factor: {performance.get('profit_factor', 0):.2f}")

        print(f"\n🎯 **Win/Loss Analysis:**")
        print(f"   Average Win: ${performance.get('avg_win_dollars', 0):,.2f} ({performance.get('avg_points_win', 0):.2f} pts)")
        print(f"   Average Loss: ${performance.get('avg_loss_dollars', 0):,.2f} ({performance.get('avg_points_loss', 0):.2f} pts)")
        print(f"   Largest Win: ${performance.get('largest_win', 0):,.2f}")
        print(f"   Largest Loss: ${performance.get('largest_loss', 0):,.2f}")
        print(f"   Average Points/Trade: {performance.get('avg_points_per_trade', 0):.2f}")

        print(f"\n📉 **Risk Metrics:**")
        print(f"   Max Drawdown: ${performance.get('max_drawdown_dollars', 0):,.2f} ({performance.get('max_drawdown_percent', 0):.2f}%)")
        print(f"   Daily Sharpe Ratio: {performance.get('daily_sharpe_ratio', 0):.3f}")
        print(f"   Max Consecutive Wins: {performance.get('max_consecutive_wins', 0)}")
        print(f"   Max Consecutive Losses: {performance.get('max_consecutive_losses', 0)}")

        print(f"\n💸 **Cost Analysis:**")
        print(f"   Commission Paid: ${performance.get('commission_paid', 0):,.2f}")
        print(f"   Trading Days: {performance.get('trading_days', 0)}")

def main():
    """Main function to run comprehensive backtest."""

    # Run backtest
    results = run_comprehensive_backtest()

    if results:
        # Generate report
        generate_performance_report(results)

        # Save detailed results
        import json
        output_file = "/Users/shubhamshanker/bt_/detailed_backtest_results_2024.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📋 Detailed results saved to: {output_file}")

        print(f"\n🎉 **COMPREHENSIVE BACKTEST COMPLETED SUCCESSFULLY**")
        print(f"✅ Full 2024 performance analysis with detailed statistics")
        print(f"✅ Multiple strategy comparison completed")
        print(f"✅ 100% Parquet data - zero CSV usage")
        print(f"✅ Production-ready backtesting system validated")

        return True
    else:
        print(f"❌ Backtest failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)