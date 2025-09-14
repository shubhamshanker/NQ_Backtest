"""
Performance Module
==================
Advanced performance metrics calculation for quantitative backtesting.
Provides comprehensive risk-adjusted returns analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import warnings
warnings.filterwarnings('ignore')

if TYPE_CHECKING:
    from pandas import DataFrame

class PerformanceCalculator:
    """
    Advanced performance metrics calculator for trading strategies.
    Provides institutional-grade risk and return analysis.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance calculator.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculation
        """
        self.risk_free_rate = risk_free_rate

    def calculate_comprehensive_metrics(self, portfolio: 'Portfolio') -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics from portfolio.

        Args:
            portfolio: Portfolio object with trade history and equity curve

        Returns:
            Dictionary with all performance metrics
        """
        trades_df = portfolio.get_trades_dataframe()
        equity_df = portfolio.get_equity_series()

        if trades_df.empty:
            return {'error': 'No trades to analyze'}

        metrics = {}

        # Basic trade statistics
        metrics.update(self._calculate_trade_stats(trades_df, portfolio))

        # Points analysis for futures
        metrics.update(self._calculate_points_analysis(trades_df, portfolio.point_value))

        # Return and risk metrics
        if not equity_df.empty:
            metrics.update(self._calculate_return_metrics(equity_df, portfolio.initial_capital))
            metrics.update(self._calculate_risk_metrics(equity_df))

        # Daily performance breakdown
        metrics.update(self._calculate_daily_performance(trades_df))

        # Advanced metrics including consecutive wins/losses
        metrics.update(self._calculate_advanced_metrics(trades_df, equity_df))

        # Additional trade metrics
        if len(trades_df) > 0:
            metrics.update({
                'total_trade_count': len(trades_df),
                'long_trades': len(trades_df[trades_df['side'] == 'LONG']),
                'short_trades': len(trades_df[trades_df['side'] == 'SHORT']),
                'long_win_rate': (len(trades_df[(trades_df['side'] == 'LONG') & (trades_df['pnl'] > 0)]) / len(trades_df[trades_df['side'] == 'LONG']) * 100) if len(trades_df[trades_df['side'] == 'LONG']) > 0 else 0,
                'short_win_rate': (len(trades_df[(trades_df['side'] == 'SHORT') & (trades_df['pnl'] > 0)]) / len(trades_df[trades_df['side'] == 'SHORT']) * 100) if len(trades_df[trades_df['side'] == 'SHORT']) > 0 else 0
            })

        return metrics

    def _calculate_trade_stats(self, trades_df: "DataFrame", portfolio: 'Portfolio') -> Dict[str, Any]:
        """Calculate basic trade statistics."""
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        total_pnl = trades_df['pnl'].sum()
        avg_trade = trades_df['pnl'].mean()

        avg_winner = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loser = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

        profit_factor = abs(avg_winner * winning_trades) / abs(avg_loser * losing_trades) if losing_trades > 0 else float('inf')

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_trade_pnl': avg_trade,
            'avg_winning_trade': avg_winner,
            'avg_losing_trade': avg_loser,
            'profit_factor': profit_factor,
            'initial_capital': portfolio.initial_capital,
            'final_equity': portfolio.get_total_value()
        }

    def _calculate_points_analysis(self, trades_df: "DataFrame", point_value: float) -> Dict[str, Any]:
        """Calculate points-based analysis for futures trading."""
        if trades_df.empty:
            return {}

        # Convert P&L to points
        trades_df = trades_df.copy()
        trades_df['pnl_points'] = trades_df['pnl'] / point_value

        winning_trades = trades_df[trades_df['pnl_points'] > 0]
        losing_trades = trades_df[trades_df['pnl_points'] < 0]

        total_points = trades_df['pnl_points'].sum()
        avg_points_per_trade = trades_df['pnl_points'].mean()

        avg_winning_points = winning_trades['pnl_points'].mean() if len(winning_trades) > 0 else 0
        avg_losing_points = losing_trades['pnl_points'].mean() if len(losing_trades) > 0 else 0

        return {
            'total_points': total_points,
            'avg_points_per_trade': avg_points_per_trade,
            'avg_winning_points': avg_winning_points,
            'avg_losing_points': avg_losing_points,
            'point_value': point_value
        }

    def _calculate_return_metrics(self, equity_df: "DataFrame", initial_capital: float) -> Dict[str, Any]:
        """Calculate return-based performance metrics."""
        if equity_df.empty or len(equity_df) < 2:
            return {}

        final_equity = equity_df['total_equity'].iloc[-1]
        total_return = ((final_equity / initial_capital) - 1) * 100

        # Calculate daily returns
        daily_returns = equity_df['total_equity'].resample('D').last().pct_change().dropna()

        if len(daily_returns) == 0:
            return {'total_return_percent': total_return}

        # Annualized metrics
        trading_days = len(daily_returns)
        mean_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()

        # Annualized return (assuming 252 trading days)
        annualized_return = (1 + mean_daily_return) ** 252 - 1 if mean_daily_return != 0 else 0
        annualized_volatility = std_daily_return * np.sqrt(252) if std_daily_return != 0 else 0

        return {
            'total_return_percent': total_return,
            'annualized_return_percent': annualized_return * 100,
            'annualized_volatility_percent': annualized_volatility * 100,
            'trading_days': trading_days
        }

    def _calculate_risk_metrics(self, equity_df: "DataFrame") -> Dict[str, Any]:
        """Calculate risk-adjusted performance metrics."""
        if equity_df.empty or len(equity_df) < 2:
            return {}

        equity_series = equity_df['total_equity']

        # Maximum Drawdown
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min() * 100

        # Daily returns for Sharpe/Sortino
        daily_returns = equity_series.resample('D').last().pct_change().dropna()

        if len(daily_returns) == 0:
            return {'max_drawdown_percent': max_drawdown}

        excess_returns = daily_returns - (self.risk_free_rate / 252)

        # Sharpe Ratio
        sharpe_ratio = 0
        if daily_returns.std() != 0:
            sharpe_ratio = (daily_returns.mean() - self.risk_free_rate / 252) / daily_returns.std() * np.sqrt(252)

        # Sortino Ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        sortino_ratio = 0
        if len(downside_returns) > 0 and downside_returns.std() != 0:
            sortino_ratio = (daily_returns.mean() - self.risk_free_rate / 252) / downside_returns.std() * np.sqrt(252)

        # Calmar Ratio
        annualized_return = (1 + daily_returns.mean()) ** 252 - 1
        calmar_ratio = abs(annualized_return / (max_drawdown / 100)) if max_drawdown != 0 else 0

        return {
            'max_drawdown_percent': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }

    def _calculate_daily_performance(self, trades_df: "DataFrame") -> Dict[str, Any]:
        """Calculate daily performance breakdown."""
        if trades_df.empty:
            return {}

        # Group by date
        trades_df = trades_df.copy()
        trades_df['trade_date'] = trades_df['entry_time'].dt.date

        daily_stats = trades_df.groupby('trade_date').agg({
            'pnl': ['count', 'sum', 'mean'],
            'pnl_percent': 'mean'
        }).round(2)

        daily_stats.columns = ['trades_per_day', 'daily_pnl', 'avg_pnl_per_trade', 'avg_pnl_percent']

        # Daily performance statistics
        profitable_days = len(daily_stats[daily_stats['daily_pnl'] > 0])
        losing_days = len(daily_stats[daily_stats['daily_pnl'] < 0])
        total_trading_days = len(daily_stats)

        daily_win_rate = (profitable_days / total_trading_days * 100) if total_trading_days > 0 else 0

        best_day = daily_stats['daily_pnl'].max()
        worst_day = daily_stats['daily_pnl'].min()
        avg_daily_pnl = daily_stats['daily_pnl'].mean()

        return {
            'total_trading_days': total_trading_days,
            'profitable_days': profitable_days,
            'losing_days': losing_days,
            'daily_win_rate': daily_win_rate,
            'best_day_pnl': best_day,
            'worst_day_pnl': worst_day,
            'avg_daily_pnl': avg_daily_pnl,
            'avg_trades_per_day': daily_stats['trades_per_day'].mean()
        }

    def _calculate_advanced_metrics(self, trades_df: "DataFrame", equity_df: "DataFrame") -> Dict[str, Any]:
        """Calculate advanced trading metrics."""
        if trades_df.empty:
            return {}

        metrics = {}

        # Trade duration analysis
        if 'duration_minutes' in trades_df.columns:
            avg_duration = trades_df['duration_minutes'].mean()
            max_duration = trades_df['duration_minutes'].max()
            min_duration = trades_df['duration_minutes'].min()

            # Convert to hours for better readability
            avg_duration_hours = avg_duration / 60
            max_duration_hours = max_duration / 60
            min_duration_hours = min_duration / 60

            metrics.update({
                'avg_trade_duration_minutes': avg_duration,
                'avg_trade_duration_hours': avg_duration_hours,
                'max_trade_duration_minutes': max_duration,
                'max_trade_duration_hours': max_duration_hours,
                'min_trade_duration_minutes': min_duration,
                'min_trade_duration_hours': min_duration_hours,
                'median_trade_duration_minutes': trades_df['duration_minutes'].median()
            })

        # Consecutive wins/losses
        pnl_signs = np.sign(trades_df['pnl'])
        consecutive_changes = (pnl_signs != pnl_signs.shift()).cumsum()
        consecutive_groups = trades_df.groupby(consecutive_changes)['pnl'].agg(['count', 'first'])

        max_consecutive_wins = consecutive_groups[consecutive_groups['first'] > 0]['count'].max() if len(consecutive_groups[consecutive_groups['first'] > 0]) > 0 else 0
        max_consecutive_losses = consecutive_groups[consecutive_groups['first'] < 0]['count'].max() if len(consecutive_groups[consecutive_groups['first'] < 0]) > 0 else 0

        metrics.update({
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses
        })

        # MAE/MFE analysis if available
        if 'max_favorable' in trades_df.columns and 'max_adverse' in trades_df.columns:
            avg_mae = trades_df['max_adverse'].mean()
            avg_mfe = trades_df['max_favorable'].mean()

            metrics.update({
                'avg_max_adverse_excursion': avg_mae,
                'avg_max_favorable_excursion': avg_mfe
            })

        return metrics

    def generate_performance_report(self, portfolio: 'Portfolio') -> str:
        """
        Generate formatted performance report.

        Args:
            portfolio: Portfolio object to analyze

        Returns:
            Formatted string report
        """
        metrics = self.calculate_comprehensive_metrics(portfolio)

        if 'error' in metrics:
            return f"❌ {metrics['error']}"

        report = []
        report.append("=" * 60)
        report.append("            TRADING STRATEGY PERFORMANCE REPORT")
        report.append("=" * 60)

        # Overview
        report.append("\n📊 OVERVIEW")
        report.append("-" * 20)
        report.append(f"Total Trades: {metrics.get('total_trades', 0):,}")
        report.append(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
        report.append(f"Total Return: {metrics.get('total_return_percent', 0):.2f}%")
        report.append(f"Initial Capital: ${metrics.get('initial_capital', 0):,.2f}")
        report.append(f"Final Equity: ${metrics.get('final_equity', 0):,.2f}")

        # P&L Analysis
        report.append("\n💰 P&L ANALYSIS")
        report.append("-" * 20)
        report.append(f"Total P&L: ${metrics.get('total_pnl', 0):,.2f}")
        report.append(f"Average Trade: ${metrics.get('avg_trade_pnl', 0):.2f}")
        report.append(f"Average Winner: ${metrics.get('avg_winning_trade', 0):.2f}")
        report.append(f"Average Loser: ${metrics.get('avg_losing_trade', 0):.2f}")
        report.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")

        # Points Analysis (for futures)
        if 'total_points' in metrics:
            report.append("\n📈 POINTS ANALYSIS")
            report.append("-" * 20)
            report.append(f"Total Points: {metrics.get('total_points', 0):.2f}")
            report.append(f"Average Points/Trade: {metrics.get('avg_points_per_trade', 0):.2f}")
            report.append(f"Average Winning Points: {metrics.get('avg_winning_points', 0):.2f}")
            report.append(f"Average Losing Points: {metrics.get('avg_losing_points', 0):.2f}")

        # Trade Duration Analysis
        if 'avg_trade_duration_minutes' in metrics:
            report.append("\n⏱️  TRADE DURATION ANALYSIS")
            report.append("-" * 20)
            report.append(f"Average Trade Duration: {metrics.get('avg_trade_duration_minutes', 0):.1f} min ({metrics.get('avg_trade_duration_hours', 0):.2f} hrs)")
            report.append(f"Median Trade Duration: {metrics.get('median_trade_duration_minutes', 0):.1f} min")
            report.append(f"Longest Trade: {metrics.get('max_trade_duration_minutes', 0):.1f} min ({metrics.get('max_trade_duration_hours', 0):.2f} hrs)")
            report.append(f"Shortest Trade: {metrics.get('min_trade_duration_minutes', 0):.1f} min ({metrics.get('min_trade_duration_hours', 0):.2f} hrs)")

        # Risk Metrics
        report.append("\n⚠️  RISK METRICS")
        report.append("-" * 20)
        report.append(f"Max Drawdown: {metrics.get('max_drawdown_percent', 0):.2f}%")
        report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        report.append(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
        report.append(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")

        # Daily Performance
        if 'total_trading_days' in metrics:
            report.append("\n📅 DAILY PERFORMANCE")
            report.append("-" * 20)
            report.append(f"Trading Days: {metrics.get('total_trading_days', 0)}")
            report.append(f"Daily Win Rate: {metrics.get('daily_win_rate', 0):.2f}%")
            report.append(f"Best Day: ${metrics.get('best_day_pnl', 0):.2f}")
            report.append(f"Worst Day: ${metrics.get('worst_day_pnl', 0):.2f}")
            report.append(f"Average Daily P&L: ${metrics.get('avg_daily_pnl', 0):.2f}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)