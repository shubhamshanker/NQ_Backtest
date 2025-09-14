"""
Quantitative Calculations Module
==============================
Professional-grade trading metrics and statistical calculations.
All calculations use actual trade data for maximum accuracy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date
import logging
from scipy import stats
from models import (
    StatisticsResponse,
    EquityCurveResponse,
    EquityCurvePoint,
    DrawdownAnalysis,
    DrawdownPeriod,
    PerformanceByPeriod,
    MonthlyReturn
)

logger = logging.getLogger(__name__)

class QuantCalculations:
    """Professional quantitative calculations for trading analysis."""

    @staticmethod
    def calculate_expectancy(df: pd.DataFrame) -> float:
        """
        Calculate expectancy: (Win% × Avg Win) - (Loss% × Avg Loss)

        This is the expected value per trade in points.
        """
        if df.empty:
            return 0.0

        total_trades = len(df)
        winning_trades = len(df[df['pnl_points'] > 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        loss_rate = 1 - win_rate

        avg_win = df[df['pnl_points'] > 0]['pnl_points'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl_points'] <= 0]['pnl_points'].mean() if (total_trades - winning_trades) > 0 else 0

        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
        return round(expectancy, 4)

    @staticmethod
    def calculate_profit_factor(df: pd.DataFrame) -> float:
        """
        Calculate profit factor: Gross Profit / Gross Loss

        Values > 1.0 indicate profitable system.
        """
        if df.empty:
            return 0.0

        gross_profit = df[df['pnl_points'] > 0]['pnl_points'].sum()
        gross_loss = abs(df[df['pnl_points'] < 0]['pnl_points'].sum())

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return round(gross_profit / gross_loss, 4)

    @staticmethod
    def calculate_sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe Ratio: (Return - Risk-free Rate) / Standard Deviation

        Uses daily returns for calculation.
        """
        if df.empty or len(df) < 2:
            return 0.0

        # Group by date to get daily returns
        daily_returns = df.groupby('date')['pnl_points'].sum()

        if len(daily_returns) < 2:
            return 0.0

        # Annualize returns (assuming 252 trading days)
        annual_return = daily_returns.mean() * 252
        daily_risk_free = risk_free_rate / 252
        excess_returns = daily_returns - daily_risk_free

        if excess_returns.std() == 0:
            return 0.0

        sharpe = (annual_return - risk_free_rate) / (daily_returns.std() * np.sqrt(252))
        return round(sharpe, 4)

    @staticmethod
    def calculate_sortino_ratio(df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino Ratio: (Return - Risk-free Rate) / Downside Deviation

        Only considers downside volatility in denominator.
        """
        if df.empty or len(df) < 2:
            return 0.0

        daily_returns = df.groupby('date')['pnl_points'].sum()

        if len(daily_returns) < 2:
            return 0.0

        annual_return = daily_returns.mean() * 252
        daily_risk_free = risk_free_rate / 252

        # Calculate downside deviation
        downside_returns = daily_returns[daily_returns < daily_risk_free]

        if len(downside_returns) == 0:
            return float('inf') if annual_return > risk_free_rate else 0.0

        downside_deviation = downside_returns.std() * np.sqrt(252)

        if downside_deviation == 0:
            return 0.0

        sortino = (annual_return - risk_free_rate) / downside_deviation
        return round(sortino, 4)

    @staticmethod
    def calculate_maximum_drawdown(df: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate maximum drawdown in points and percentage.

        Returns: (max_drawdown_points, max_drawdown_percent)
        """
        if df.empty:
            return 0.0, 0.0

        # Sort by date and calculate cumulative PnL
        df_sorted = df.sort_values(['date', 'entry_datetime']).copy()
        cumulative_pnl = df_sorted['pnl_points'].cumsum()

        # Calculate running maximum
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max

        max_drawdown_points = abs(drawdown.min())

        # Calculate percentage drawdown
        if running_max.max() > 0:
            max_drawdown_percent = (max_drawdown_points / running_max.max()) * 100
        else:
            max_drawdown_percent = 0.0

        return round(max_drawdown_points, 2), round(max_drawdown_percent, 2)

    @staticmethod
    def calculate_consecutive_streaks(df: pd.DataFrame) -> Dict[str, int]:
        """
        Calculate maximum consecutive wins and losses.

        Returns: Dict with max_wins, max_losses, current_streak, current_type
        """
        if df.empty:
            return {
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'current_streak': 0,
                'current_streak_type': 'none'
            }

        # Sort by date and time
        df_sorted = df.sort_values(['date', 'entry_datetime']).copy()

        # Create win/loss sequence
        win_loss = (df_sorted['pnl_points'] > 0).astype(int)  # 1 for win, 0 for loss

        # Calculate streaks
        max_wins = 0
        max_losses = 0
        current_win_streak = 0
        current_loss_streak = 0

        for is_win in win_loss:
            if is_win:
                current_win_streak += 1
                current_loss_streak = 0
                max_wins = max(max_wins, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_losses = max(max_losses, current_loss_streak)

        # Determine current streak
        if len(win_loss) > 0:
            if win_loss.iloc[-1]:
                current_streak = current_win_streak
                current_type = 'wins'
            else:
                current_streak = current_loss_streak
                current_type = 'losses'
        else:
            current_streak = 0
            current_type = 'none'

        return {
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses,
            'current_streak': current_streak,
            'current_streak_type': current_type
        }

    @staticmethod
    def calculate_calmar_ratio(df: pd.DataFrame) -> float:
        """
        Calculate Calmar Ratio: Annual Return / Maximum Drawdown

        Measures return per unit of downside risk.
        """
        if df.empty:
            return 0.0

        # Calculate annualized return
        total_days = (df['date'].max() - df['date'].min()).days
        if total_days == 0:
            return 0.0

        total_return = df['pnl_points'].sum()
        annual_return = (total_return / total_days) * 365

        # Calculate maximum drawdown
        max_dd_points, max_dd_percent = QuantCalculations.calculate_maximum_drawdown(df)

        if max_dd_points == 0:
            return float('inf') if annual_return > 0 else 0.0

        calmar = annual_return / max_dd_points
        return round(calmar, 4)

    @staticmethod
    def calculate_recovery_factor(df: pd.DataFrame) -> float:
        """
        Calculate Recovery Factor: Total Net Profit / Maximum Drawdown
        """
        if df.empty:
            return 0.0

        total_profit = df['pnl_points'].sum()
        max_dd_points, _ = QuantCalculations.calculate_maximum_drawdown(df)

        if max_dd_points == 0:
            return float('inf') if total_profit > 0 else 0.0

        recovery = total_profit / max_dd_points
        return round(recovery, 4)

    @staticmethod
    def calculate_ulcer_index(df: pd.DataFrame) -> float:
        """
        Calculate Ulcer Index: Square root of mean squared drawdown percentage.

        Measures the depth and duration of drawdowns.
        """
        if df.empty:
            return 0.0

        df_sorted = df.sort_values(['date', 'entry_datetime']).copy()
        cumulative_pnl = df_sorted['pnl_points'].cumsum()
        running_max = cumulative_pnl.expanding().max()

        # Calculate percentage drawdown from running max
        drawdown_pct = ((cumulative_pnl - running_max) / running_max.abs()) * 100
        drawdown_pct = drawdown_pct.fillna(0)

        # Calculate Ulcer Index
        ulcer = np.sqrt((drawdown_pct ** 2).mean())
        return round(ulcer, 4)

    @staticmethod
    def calculate_var_95(df: pd.DataFrame) -> float:
        """
        Calculate Value at Risk (95%): 5th percentile of trade returns.

        Represents the worst expected loss over a given time period.
        """
        if df.empty:
            return 0.0

        trade_returns = df['pnl_points']
        var_95 = np.percentile(trade_returns, 5)
        return round(var_95, 4)

    @staticmethod
    def calculate_comprehensive_statistics(df: pd.DataFrame) -> StatisticsResponse:
        """
        Calculate comprehensive statistics for filtered trade data.

        Returns complete StatisticsResponse with all metrics.
        """
        if df.empty:
            # Return empty statistics
            return StatisticsResponse(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl_points=0.0,
                total_pnl_dollars=0.0,
                points_per_day=0.0,
                avg_trade_points=0.0,
                avg_winning_trade=0.0,
                avg_losing_trade=0.0,
                expectancy=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown_percent=0.0,
                max_drawdown_points=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                current_streak=0,
                current_streak_type="none",
                best_trade=0.0,
                worst_trade=0.0,
                median_trade=0.0,
                std_deviation=0.0,
                total_trading_days=0,
                avg_trades_per_day=0.0,
                avg_trade_duration_minutes=0.0,
                recovery_factor=0.0,
                ulcer_index=0.0,
                var_95=0.0
            )

        # Basic metrics
        total_trades = len(df)
        winning_trades = len(df[df['pnl_points'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100

        # P&L metrics
        total_pnl_points = df['pnl_points'].sum()
        total_pnl_dollars = df['pnl_dollars'].sum()
        avg_trade_points = df['pnl_points'].mean()
        avg_winning_trade = df[df['pnl_points'] > 0]['pnl_points'].mean() if winning_trades > 0 else 0
        avg_losing_trade = df[df['pnl_points'] <= 0]['pnl_points'].mean() if losing_trades > 0 else 0

        # Time-based metrics
        trading_days = df['date'].nunique()
        points_per_day = total_pnl_points / trading_days if trading_days > 0 else 0
        avg_trades_per_day = total_trades / trading_days if trading_days > 0 else 0
        avg_trade_duration = df['trade_duration_minutes'].mean()

        # Distribution metrics
        best_trade = df['pnl_points'].max()
        worst_trade = df['pnl_points'].min()
        median_trade = df['pnl_points'].median()
        std_deviation = df['pnl_points'].std()

        # Advanced metrics
        expectancy = QuantCalculations.calculate_expectancy(df)
        profit_factor = QuantCalculations.calculate_profit_factor(df)
        sharpe_ratio = QuantCalculations.calculate_sharpe_ratio(df)
        sortino_ratio = QuantCalculations.calculate_sortino_ratio(df)
        calmar_ratio = QuantCalculations.calculate_calmar_ratio(df)

        # Drawdown analysis
        max_dd_points, max_dd_percent = QuantCalculations.calculate_maximum_drawdown(df)

        # Streak analysis
        streaks = QuantCalculations.calculate_consecutive_streaks(df)

        # Risk metrics
        recovery_factor = QuantCalculations.calculate_recovery_factor(df)
        ulcer_index = QuantCalculations.calculate_ulcer_index(df)
        var_95 = QuantCalculations.calculate_var_95(df)

        return StatisticsResponse(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate, 2),
            total_pnl_points=round(total_pnl_points, 2),
            total_pnl_dollars=round(total_pnl_dollars, 2),
            points_per_day=round(points_per_day, 2),
            avg_trade_points=round(avg_trade_points, 2),
            avg_winning_trade=round(avg_winning_trade, 2),
            avg_losing_trade=round(avg_losing_trade, 2),
            expectancy=expectancy,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown_percent=max_dd_percent,
            max_drawdown_points=max_dd_points,
            max_consecutive_wins=streaks['max_consecutive_wins'],
            max_consecutive_losses=streaks['max_consecutive_losses'],
            current_streak=streaks['current_streak'],
            current_streak_type=streaks['current_streak_type'],
            best_trade=round(best_trade, 2),
            worst_trade=round(worst_trade, 2),
            median_trade=round(median_trade, 2),
            std_deviation=round(std_deviation, 2),
            total_trading_days=trading_days,
            avg_trades_per_day=round(avg_trades_per_day, 2),
            avg_trade_duration_minutes=round(avg_trade_duration, 0),
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index,
            var_95=var_95
        )

    @staticmethod
    def calculate_equity_curve(df: pd.DataFrame) -> EquityCurveResponse:
        """
        Calculate detailed equity curve with drawdown analysis.
        """
        if df.empty:
            return EquityCurveResponse(
                strategy=None,
                total_points=0.0,
                max_equity=0.0,
                final_equity=0.0,
                max_drawdown=0.0,
                curve_data=[]
            )

        # Sort by date and time
        df_sorted = df.sort_values(['date', 'entry_datetime']).copy()

        # Calculate cumulative metrics
        df_sorted['cumulative_pnl'] = df_sorted['pnl_points'].cumsum()
        starting_equity = 0
        df_sorted['running_equity'] = starting_equity + df_sorted['cumulative_pnl']

        # Calculate drawdown
        df_sorted['running_max'] = df_sorted['running_equity'].expanding().max()
        df_sorted['drawdown_points'] = df_sorted['running_equity'] - df_sorted['running_max']
        df_sorted['drawdown_percent'] = (df_sorted['drawdown_points'] / df_sorted['running_max'].abs()) * 100
        df_sorted['drawdown_percent'] = df_sorted['drawdown_percent'].fillna(0)

        # Create curve data points
        curve_data = []
        for _, row in df_sorted.iterrows():
            point = EquityCurvePoint(
                date=row['date'].strftime('%Y-%m-%d'),
                cumulative_pnl=round(row['cumulative_pnl'], 2),
                running_equity=round(row['running_equity'], 2),
                drawdown_percent=round(row['drawdown_percent'], 2),
                drawdown_points=round(row['drawdown_points'], 2)
            )
            curve_data.append(point)

        # Summary metrics
        total_points = df_sorted['pnl_points'].sum()
        max_equity = df_sorted['running_equity'].max()
        final_equity = df_sorted['running_equity'].iloc[-1]
        max_drawdown = abs(df_sorted['drawdown_points'].min())

        # Determine strategy (if all trades are from same strategy)
        strategy_name = None
        if 'strategy' in df_sorted.columns:
            unique_strategies = df_sorted['strategy'].unique()
            if len(unique_strategies) == 1:
                strategy_name = unique_strategies[0]

        return EquityCurveResponse(
            strategy=strategy_name,
            total_points=round(total_points, 2),
            max_equity=round(max_equity, 2),
            final_equity=round(final_equity, 2),
            max_drawdown=round(max_drawdown, 2),
            curve_data=curve_data
        )

    @staticmethod
    def calculate_drawdown_analysis(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate detailed drawdown analysis including periods and recovery times.
        """
        if df.empty:
            return {
                'max_drawdown_percent': 0.0,
                'max_drawdown_points': 0.0,
                'current_drawdown_percent': 0.0,
                'current_drawdown_points': 0.0,
                'drawdown_periods': [],
                'average_drawdown_duration': 0.0,
                'average_recovery_time': 0.0
            }

        # Sort and calculate equity curve
        df_sorted = df.sort_values(['date', 'entry_datetime']).copy()
        df_sorted['cumulative_pnl'] = df_sorted['pnl_points'].cumsum()
        df_sorted['running_max'] = df_sorted['cumulative_pnl'].expanding().max()
        df_sorted['drawdown_points'] = df_sorted['cumulative_pnl'] - df_sorted['running_max']
        df_sorted['drawdown_percent'] = (df_sorted['drawdown_points'] / df_sorted['running_max'].abs()) * 100
        df_sorted['drawdown_percent'] = df_sorted['drawdown_percent'].fillna(0)

        # Find drawdown periods (sequences where drawdown < 0)
        df_sorted['in_drawdown'] = df_sorted['drawdown_points'] < -0.01  # Allow for small rounding
        df_sorted['dd_group'] = (df_sorted['in_drawdown'] != df_sorted['in_drawdown'].shift()).cumsum()

        drawdown_periods = []

        for group_id, group_df in df_sorted[df_sorted['in_drawdown']].groupby('dd_group'):
            start_date = group_df['date'].min()
            end_date = group_df['date'].max()
            duration = (end_date - start_date).days + 1

            max_dd_points = abs(group_df['drawdown_points'].min())
            max_dd_percent = abs(group_df['drawdown_percent'].min())

            # Find recovery (if any)
            recovery_date = None
            recovery_duration = None

            # Look for recovery after this drawdown period
            after_dd = df_sorted[df_sorted['date'] > end_date]
            if not after_dd.empty:
                recovery_idx = after_dd[after_dd['drawdown_points'] >= -0.01].index
                if len(recovery_idx) > 0:
                    recovery_row = df_sorted.loc[recovery_idx[0]]
                    recovery_date = recovery_row['date']
                    recovery_duration = (recovery_date - end_date).days

            drawdown_periods.append({
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'duration_days': duration,
                'max_drawdown_percent': round(max_dd_percent, 2),
                'max_drawdown_points': round(max_dd_points, 2),
                'recovery_date': recovery_date.strftime('%Y-%m-%d') if recovery_date else None,
                'recovery_duration_days': recovery_duration
            })

        # Calculate averages
        if drawdown_periods:
            avg_duration = np.mean([p['duration_days'] for p in drawdown_periods])
            recovery_times = [p['recovery_duration_days'] for p in drawdown_periods if p['recovery_duration_days'] is not None]
            avg_recovery = np.mean(recovery_times) if recovery_times else 0
        else:
            avg_duration = 0
            avg_recovery = 0

        # Current drawdown
        current_dd_points = df_sorted['drawdown_points'].iloc[-1]
        current_dd_percent = df_sorted['drawdown_percent'].iloc[-1]

        # Maximum drawdown
        max_dd_points = abs(df_sorted['drawdown_points'].min())
        max_dd_percent = abs(df_sorted['drawdown_percent'].min())

        return {
            'max_drawdown_percent': round(max_dd_percent, 2),
            'max_drawdown_points': round(max_dd_points, 2),
            'current_drawdown_percent': round(current_dd_percent, 2),
            'current_drawdown_points': round(current_dd_points, 2),
            'drawdown_periods': drawdown_periods,
            'average_drawdown_duration': round(avg_duration, 1),
            'average_recovery_time': round(avg_recovery, 1)
        }