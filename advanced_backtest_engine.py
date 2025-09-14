#!/usr/bin/env python3
"""
ADVANCED BACKTESTING ENGINE WITH STRICT RULE ENFORCEMENT
========================================================
Enforces all non-negotiable rules while optimizing for >50 pts/day
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import json
import warnings
from typing import Dict, List, Any, Tuple, Optional
warnings.filterwarnings('ignore')


class StrictRulesBacktester:
    """Backtesting engine with strict enforcement of non-negotiable rules."""
    
    def __init__(self):
        # NON-NEGOTIABLE RULES (FIXED)
        self.POSITION_SIZE = 1  # Always 1 contract only
        self.PARTIAL_BOOKING_PCT = 0.5  # Book 50% at fixed target
        self.MAX_TRADES_PER_DAY = 3  # Maximum 3 trades per day only
        self.TRADING_START = time(9, 30)  # NY time
        self.TRADING_END = time(16, 0)  # NY time
        self.NEXT_CANDLE_ENTRY = True  # Must execute on next candle
        
        # Performance tracking
        self.trades = []
        self.daily_results = {}
        self.metrics = {}
        
    def convert_chicago_to_ny(self, chicago_time: datetime) -> datetime:
        """Convert Chicago time to NY time (Chicago is 1 hour behind NY)."""
        return chicago_time + timedelta(hours=1)
    
    def is_ny_trading_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is within NY trading hours."""
        ny_time = self.convert_chicago_to_ny(timestamp)
        return self.TRADING_START <= ny_time.time() <= self.TRADING_END
    
    def execute_trade(self, signal: Dict[str, Any], current_bar: Dict[str, Any], 
                     next_bar: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute trade with strict rule enforcement."""
        
        # RULE: Next candle entry
        if self.NEXT_CANDLE_ENTRY and next_bar:
            entry_price = next_bar['open']
            entry_time = next_bar['timestamp']
        else:
            entry_price = current_bar['close']
            entry_time = current_bar['timestamp']
        
        # RULE: Always 1 contract
        position_size = self.POSITION_SIZE
        
        # Calculate trade parameters
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        
        # RULE: 50% partial booking level
        partial_target = entry_price + ((take_profit - entry_price) * 0.5) if signal['signal'] == 'BUY' else \
                        entry_price - ((entry_price - take_profit) * 0.5)
        
        trade = {
            'entry_time': entry_time,
            'entry_price': entry_price,
            'signal': signal['signal'],
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'partial_target': partial_target,
            'position_size': position_size,
            'partial_booked': False,
            'partial_exit_price': None,
            'partial_exit_time': None,
            'stop_at_breakeven': False,
            'exit_price': None,
            'exit_time': None,
            'pnl': 0,
            'status': 'OPEN',
            'max_favorable_excursion': 0,
            'max_adverse_excursion': 0
        }
        
        return trade
    
    def manage_position(self, trade: Dict[str, Any], bar: Dict[str, Any]) -> Dict[str, Any]:
        """Manage open position with partial booking and stop management."""
        
        if trade['status'] != 'OPEN':
            return trade
        
        current_price = bar['close']
        
        # Track excursions
        if trade['signal'] == 'BUY':
            favorable = current_price - trade['entry_price']
            adverse = trade['entry_price'] - bar['low']
        else:
            favorable = trade['entry_price'] - current_price
            adverse = bar['high'] - trade['entry_price']
        
        trade['max_favorable_excursion'] = max(trade['max_favorable_excursion'], favorable)
        trade['max_adverse_excursion'] = max(trade['max_adverse_excursion'], adverse)
        
        # RULE: 50% partial booking
        if not trade['partial_booked']:
            if trade['signal'] == 'BUY' and current_price >= trade['partial_target']:
                trade['partial_booked'] = True
                trade['partial_exit_price'] = current_price
                trade['partial_exit_time'] = bar['timestamp']
                # Move stop to breakeven after partial booking
                trade['stop_at_breakeven'] = True
                trade['stop_loss'] = trade['entry_price']
                
            elif trade['signal'] == 'SELL' and current_price <= trade['partial_target']:
                trade['partial_booked'] = True
                trade['partial_exit_price'] = current_price
                trade['partial_exit_time'] = bar['timestamp']
                # Move stop to breakeven after partial booking
                trade['stop_at_breakeven'] = True
                trade['stop_loss'] = trade['entry_price']
        
        # Check for stop loss hit
        if trade['signal'] == 'BUY' and bar['low'] <= trade['stop_loss']:
            trade['exit_price'] = trade['stop_loss']
            trade['exit_time'] = bar['timestamp']
            trade['status'] = 'STOPPED'
            
        elif trade['signal'] == 'SELL' and bar['high'] >= trade['stop_loss']:
            trade['exit_price'] = trade['stop_loss']
            trade['exit_time'] = bar['timestamp']
            trade['status'] = 'STOPPED'
        
        # Check for take profit hit
        elif trade['signal'] == 'BUY' and bar['high'] >= trade['take_profit']:
            trade['exit_price'] = trade['take_profit']
            trade['exit_time'] = bar['timestamp']
            trade['status'] = 'TARGET'
            
        elif trade['signal'] == 'SELL' and bar['low'] <= trade['take_profit']:
            trade['exit_price'] = trade['take_profit']
            trade['exit_time'] = bar['timestamp']
            trade['status'] = 'TARGET'
        
        # Calculate PnL if trade closed
        if trade['status'] in ['STOPPED', 'TARGET']:
            trade['pnl'] = self.calculate_pnl(trade)
        
        return trade
    
    def calculate_pnl(self, trade: Dict[str, Any]) -> float:
        """Calculate P&L including partial booking."""
        
        if trade['signal'] == 'BUY':
            # P&L from partial exit (50% of position)
            partial_pnl = 0
            if trade['partial_booked']:
                partial_pnl = (trade['partial_exit_price'] - trade['entry_price']) * 0.5 * 20  # $20 per point
            
            # P&L from final exit (remaining 50%)
            final_pnl = (trade['exit_price'] - trade['entry_price']) * 0.5 * 20
            
        else:  # SELL
            # P&L from partial exit (50% of position)
            partial_pnl = 0
            if trade['partial_booked']:
                partial_pnl = (trade['entry_price'] - trade['partial_exit_price']) * 0.5 * 20
            
            # P&L from final exit (remaining 50%)
            final_pnl = (trade['entry_price'] - trade['exit_price']) * 0.5 * 20
        
        total_pnl = partial_pnl + final_pnl
        return total_pnl
    
    def backtest_strategy(self, data: pd.DataFrame, strategy_func) -> Dict[str, Any]:
        """Run backtest with strict rule enforcement."""
        
        # Initialize tracking
        self.trades = []
        self.daily_results = {}
        daily_trade_count = {}
        open_position = None
        
        # Iterate through data
        for i in range(len(data) - 1):  # -1 to allow next candle entry
            bar = data.iloc[i]
            next_bar = data.iloc[i + 1] if i < len(data) - 1 else None
            
            # Convert to dict for compatibility
            bar_dict = {
                'timestamp': bar['timestamp'],
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'close': bar['close'],
                'volume': bar.get('volume', 1000)
            }
            
            current_date = bar['timestamp'].date()
            
            # Initialize daily tracking
            if current_date not in daily_trade_count:
                daily_trade_count[current_date] = 0
                self.daily_results[current_date] = {
                    'trades': 0,
                    'pnl': 0,
                    'points': 0,
                    'wins': 0,
                    'losses': 0
                }
            
            # Check NY trading hours
            if not self.is_ny_trading_hours(bar['timestamp']):
                continue
            
            # Manage open position
            if open_position:
                open_position = self.manage_position(open_position, bar_dict)
                
                if open_position['status'] in ['STOPPED', 'TARGET']:
                    # Record closed trade
                    self.trades.append(open_position)
                    
                    # Update daily results
                    points = open_position['pnl'] / 20  # Convert dollars to points
                    self.daily_results[current_date]['pnl'] += open_position['pnl']
                    self.daily_results[current_date]['points'] += points
                    
                    if open_position['pnl'] > 0:
                        self.daily_results[current_date]['wins'] += 1
                    else:
                        self.daily_results[current_date]['losses'] += 1
                    
                    open_position = None
            
            # Check for new signals (only if no open position)
            if not open_position:
                # RULE: Max 3 trades per day
                if daily_trade_count[current_date] >= self.MAX_TRADES_PER_DAY:
                    continue
                
                # Get signal from strategy
                signal = strategy_func(bar_dict, self.daily_results.get(current_date, {}))
                
                if signal and signal['signal'] in ['BUY', 'SELL']:
                    # Execute trade on next candle
                    if next_bar is not None:
                        next_bar_dict = {
                            'timestamp': next_bar['timestamp'],
                            'open': next_bar['open'],
                            'high': next_bar['high'],
                            'low': next_bar['low'],
                            'close': next_bar['close'],
                            'volume': next_bar.get('volume', 1000)
                        }
                        
                        open_position = self.execute_trade(signal, bar_dict, next_bar_dict)
                        daily_trade_count[current_date] += 1
                        self.daily_results[current_date]['trades'] += 1
        
        # Calculate metrics
        self.calculate_metrics()
        
        return {
            'trades': self.trades,
            'daily_results': self.daily_results,
            'metrics': self.metrics
        }
    
    def calculate_metrics(self) -> None:
        """Calculate comprehensive performance metrics."""
        
        if not self.trades:
            self.metrics = {
                'total_trades': 0,
                'avg_daily_points': 0,
                'total_points': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0
            }
            return
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100
        
        # Points calculation
        total_points = sum(t['pnl'] / 20 for t in self.trades)  # $20 per point
        
        # Daily metrics
        daily_points = [d['points'] for d in self.daily_results.values()]
        avg_daily_points = np.mean(daily_points) if daily_points else 0
        
        # Drawdown calculation
        cumulative_pnl = []
        running_total = 0
        peak = 0
        max_dd = 0
        
        for trade in self.trades:
            running_total += trade['pnl'] / 20
            cumulative_pnl.append(running_total)
            
            if running_total > peak:
                peak = running_total
            
            drawdown = peak - running_total
            if drawdown > max_dd:
                max_dd = drawdown
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Sharpe ratio (simplified)
        if daily_points:
            daily_returns = np.array(daily_points)
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        else:
            sharpe = 0
        
        # Sortino ratio (downside deviation)
        if daily_points:
            negative_returns = [r for r in daily_points if r < 0]
            if negative_returns:
                downside_dev = np.std(negative_returns)
                sortino = np.mean(daily_points) / downside_dev * np.sqrt(252) if downside_dev > 0 else 0
            else:
                sortino = sharpe  # No negative returns
        else:
            sortino = 0
        
        self.metrics = {
            'total_trades': total_trades,
            'avg_daily_points': avg_daily_points,
            'total_points': total_points,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win': np.mean([t['pnl']/20 for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl']/20 for t in losing_trades]) if losing_trades else 0,
            'largest_win': max([t['pnl']/20 for t in winning_trades]) if winning_trades else 0,
            'largest_loss': min([t['pnl']/20 for t in losing_trades]) if losing_trades else 0,
            'consecutive_wins': self.max_consecutive(winning_trades),
            'consecutive_losses': self.max_consecutive(losing_trades),
            'expectancy': (win_rate/100 * self.metrics.get('avg_win', 0)) - ((1-win_rate/100) * abs(self.metrics.get('avg_loss', 0)))
        }
    
    def max_consecutive(self, trades: List[Dict]) -> int:
        """Calculate maximum consecutive wins or losses."""
        if not trades:
            return 0
        
        max_streak = 1
        current_streak = 1
        
        for i in range(1, len(trades)):
            if abs(trades[i]['entry_time'] - trades[i-1]['entry_time']).days <= 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        
        return max_streak
    
    def generate_report(self) -> str:
        """Generate comprehensive performance report."""
        
        report = []
        report.append("=" * 80)
        report.append("BACKTEST PERFORMANCE REPORT")
        report.append("=" * 80)
        
        # Check target criteria
        target_met = (self.metrics['avg_daily_points'] > 50 and 
                     self.metrics['max_drawdown'] < 200)
        
        if target_met:
            report.append("\n✅ TARGET ACHIEVED! >50 pts/day with <200 max DD")
        else:
            report.append("\n⚠️ Target not met (Need >50 pts/day with <200 DD)")
        
        report.append("\n📊 KEY METRICS:")
        report.append("-" * 40)
        report.append(f"Average Daily Points: {self.metrics['avg_daily_points']:.2f}")
        report.append(f"Total Points: {self.metrics['total_points']:.2f}")
        report.append(f"Max Drawdown: {self.metrics['max_drawdown']:.2f} points")
        report.append(f"Win Rate: {self.metrics['win_rate']:.1f}%")
        report.append(f"Profit Factor: {self.metrics['profit_factor']:.2f}")
        report.append(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        report.append(f"Sortino Ratio: {self.metrics['sortino_ratio']:.2f}")
        
        report.append("\n📈 TRADE STATISTICS:")
        report.append("-" * 40)
        report.append(f"Total Trades: {self.metrics['total_trades']}")
        report.append(f"Winning Trades: {self.metrics['winning_trades']}")
        report.append(f"Losing Trades: {self.metrics['losing_trades']}")
        report.append(f"Average Win: {self.metrics['avg_win']:.2f} points")
        report.append(f"Average Loss: {self.metrics['avg_loss']:.2f} points")
        report.append(f"Largest Win: {self.metrics['largest_win']:.2f} points")
        report.append(f"Largest Loss: {self.metrics['largest_loss']:.2f} points")
        report.append(f"Expectancy: {self.metrics.get('expectancy', 0):.2f} points")
        
        report.append("\n📅 DAILY BREAKDOWN (Top 10 Days):")
        report.append("-" * 40)
        
        # Sort daily results by points
        sorted_days = sorted(self.daily_results.items(), 
                           key=lambda x: x[1]['points'], 
                           reverse=True)[:10]
        
        for date, results in sorted_days:
            report.append(f"{date}: {results['points']:.2f} pts | "
                        f"Trades: {results['trades']} | "
                        f"W/L: {results['wins']}/{results['losses']}")
        
        report.append("\n🔴 WORST DAYS (Bottom 5):")
        report.append("-" * 40)
        
        worst_days = sorted(self.daily_results.items(), 
                          key=lambda x: x[1]['points'])[:5]
        
        for date, results in worst_days:
            report.append(f"{date}: {results['points']:.2f} pts | "
                        f"Trades: {results['trades']}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


def create_sample_strategy():
    """Create a sample strategy function for testing."""
    
    def strategy(bar: Dict[str, Any], daily_stats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Sample strategy with market structure awareness."""
        
        # Simple OR breakout logic for demonstration
        current_time = bar['timestamp'].time()
        
        # Skip if outside prime trading hours
        if current_time < time(10, 0) or current_time > time(15, 30):
            return None
        
        # Simple momentum check
        if bar['close'] > bar['open'] * 1.001:  # 0.1% above open
            return {
                'signal': 'BUY',
                'stop_loss': bar['close'] - 30,
                'take_profit': bar['close'] + 150,  # 5:1 R:R
                'reason': 'Momentum breakout'
            }
        elif bar['close'] < bar['open'] * 0.999:  # 0.1% below open
            return {
                'signal': 'SELL',
                'stop_loss': bar['close'] + 30,
                'take_profit': bar['close'] - 150,
                'reason': 'Momentum breakdown'
            }
        
        return None
    
    return strategy


def main():
    """Main execution function."""
    print("=" * 80)
    print("ADVANCED BACKTESTING ENGINE")
    print("Enforcing All Non-Negotiable Rules")
    print("=" * 80)
    
    # Create backtester
    backtester = StrictRulesBacktester()
    
    # Load sample data (in production, load real NQ data)
    print("\n📊 Loading market data...")
    
    # Create sample data for demonstration
    dates = pd.date_range(start='2024-01-01 09:30:00', 
                         end='2024-01-31 16:00:00', 
                         freq='1min')
    
    # Filter to trading hours only
    dates = [d for d in dates if d.time() >= time(9, 30) and d.time() <= time(16, 0)]
    
    # Create sample OHLCV data
    np.random.seed(42)
    data = pd.DataFrame({
        'timestamp': dates,
        'open': 15000 + np.random.randn(len(dates)) * 50,
        'high': 15050 + np.random.randn(len(dates)) * 50,
        'low': 14950 + np.random.randn(len(dates)) * 50,
        'close': 15000 + np.random.randn(len(dates)) * 50,
        'volume': np.random.randint(1000, 5000, len(dates))
    })
    
    # Ensure high > low
    data['high'] = data[['open', 'high', 'close']].max(axis=1) + abs(np.random.randn(len(dates)) * 5)
    data['low'] = data[['open', 'low', 'close']].min(axis=1) - abs(np.random.randn(len(dates)) * 5)
    
    print(f"✅ Loaded {len(data)} bars of data")
    print(f"📅 Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    
    # Create strategy
    strategy = create_sample_strategy()
    
    # Run backtest
    print("\n🚀 Running backtest with strict rules...")
    results = backtester.backtest_strategy(data, strategy)
    
    # Generate and display report
    report = backtester.generate_report()
    print("\n" + report)
    
    # Save results
    output = {
        'metrics': backtester.metrics,
        'daily_results': {str(k): v for k, v in backtester.daily_results.items()},
        'trade_count': len(backtester.trades),
        'rules_enforced': {
            'position_size': backtester.POSITION_SIZE,
            'partial_booking': f"{backtester.PARTIAL_BOOKING_PCT*100}%",
            'max_trades_per_day': backtester.MAX_TRADES_PER_DAY,
            'trading_hours': f"{backtester.TRADING_START} - {backtester.TRADING_END} NY",
            'next_candle_entry': backtester.NEXT_CANDLE_ENTRY
        }
    }
    
    with open('backtest_results_strict_rules.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\n💾 Results saved to backtest_results_strict_rules.json")
    
    return results


if __name__ == "__main__":
    results = main()