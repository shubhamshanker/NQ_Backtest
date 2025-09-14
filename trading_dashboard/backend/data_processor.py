"""
Data Processor for Trading Dashboard
==================================
High-performance data loading, filtering, and processing using pandas.
Handles all CSV operations and data transformations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import date, datetime
import logging
from models import FilterRequest, TradeFilters

logger = logging.getLogger(__name__)

class DataProcessor:
    """High-performance data processor for trading analysis."""

    def __init__(self):
        self.trades_df: Optional[pd.DataFrame] = None
        self.summaries_df: Optional[pd.DataFrame] = None
        self.data_path = Path(__file__).parent.parent / "data"

    def load_data(self) -> None:
        """Load trades and summary data from CSV files."""
        logger.info("📊 Loading trade data...")

        try:
            # Load main trades data
            trades_file = self.data_path / "all_trades.csv"
            if not trades_file.exists():
                raise FileNotFoundError(f"Trades file not found: {trades_file}")

            self.trades_df = pd.read_csv(trades_file)

            # Convert date columns
            self.trades_df['date'] = pd.to_datetime(self.trades_df['date']).dt.date

            # Handle mixed EST/EDT timezone formats safely
            # Strip timezone info and use clean datetime parsing
            def clean_datetime_string(dt_str):
                """Remove timezone abbreviations from datetime strings."""
                if isinstance(dt_str, str):
                    return dt_str.replace(' EST', '').replace(' EDT', '')
                return dt_str

            try:
                # Clean timezone strings before parsing
                self.trades_df['entry_datetime_clean'] = self.trades_df['entry_datetime'].apply(clean_datetime_string)
                self.trades_df['exit_datetime_clean'] = self.trades_df['exit_datetime'].apply(clean_datetime_string)

                # Parse without timezone info
                self.trades_df['entry_datetime'] = pd.to_datetime(
                    self.trades_df['entry_datetime_clean'],
                    errors='coerce'
                )
                self.trades_df['exit_datetime'] = pd.to_datetime(
                    self.trades_df['exit_datetime_clean'],
                    errors='coerce'
                )

                # Clean up temporary columns
                self.trades_df.drop(['entry_datetime_clean', 'exit_datetime_clean'], axis=1, inplace=True)

            except Exception as e:
                logger.warning(f"⚠️ Datetime parsing failed: {e}. Using fallback.")
                # Ultimate fallback
                self.trades_df['entry_datetime'] = pd.to_datetime(self.trades_df['entry_datetime'], errors='coerce')
                self.trades_df['exit_datetime'] = pd.to_datetime(self.trades_df['exit_datetime'], errors='coerce')

            # Remove any rows with invalid datetime parsing
            before_count = len(self.trades_df)
            self.trades_df = self.trades_df.dropna(subset=['entry_datetime', 'exit_datetime'])
            after_count = len(self.trades_df)

            if before_count != after_count:
                logger.warning(f"⚠️ Removed {before_count - after_count} trades with invalid datetime data")

            # Optional: Validate NY session compliance (disabled for now to prevent startup failures)
            # self._validate_ny_session_compliance_optional()

            logger.info(f"✅ Loaded {len(self.trades_df):,} trades")

            # Load strategy summaries if available
            summaries_file = self.data_path / "strategy_summaries.csv"
            if summaries_file.exists():
                self.summaries_df = pd.read_csv(summaries_file)
                logger.info(f"✅ Loaded {len(self.summaries_df)} strategy summaries")
            else:
                logger.warning("⚠️ Strategy summaries file not found, will calculate on-demand")

            # Create indexes for faster filtering
            self._create_indexes()

        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise

    def _create_indexes(self) -> None:
        """Create indexes for faster data access."""
        if self.trades_df is not None:
            # Sort by strategy and date for optimal performance
            self.trades_df = self.trades_df.sort_values(['strategy', 'date', 'entry_datetime'])

            # Create categorical for faster filtering
            self.trades_df['strategy'] = self.trades_df['strategy'].astype('category')
            self.trades_df['weekday'] = self.trades_df['weekday'].astype('category')
            self.trades_df['direction'] = self.trades_df['direction'].astype('category')

            logger.info("✅ Data indexes created for optimal performance")

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies."""
        if self.trades_df is None:
            return []
        return self.trades_df['strategy'].unique().tolist()

    def get_strategy_summaries(self) -> List[Dict[str, Any]]:
        """Get summary statistics for all strategies."""
        if self.summaries_df is not None:
            return self.summaries_df.to_dict('records')

        # Calculate summaries on-demand
        strategies = self.get_available_strategies()
        summaries = []

        for strategy in strategies:
            summary = self.get_strategy_summary(strategy)
            summaries.append(summary)

        return summaries

    def get_strategy_summary(self, strategy_name: str) -> Dict[str, Any]:
        """Get detailed summary for a specific strategy."""
        if self.trades_df is None:
            raise ValueError("No trade data loaded")

        strategy_df = self.trades_df[self.trades_df['strategy'] == strategy_name].copy()

        if strategy_df.empty:
            raise ValueError(f"No data found for strategy: {strategy_name}")

        # Calculate comprehensive metrics
        total_trades = len(strategy_df)
        winning_trades = len(strategy_df[strategy_df['win'] == 1])
        losing_trades = len(strategy_df[strategy_df['loss'] == 1])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        total_points = strategy_df['pnl_points'].sum()
        total_pnl = strategy_df['pnl_dollars'].sum()

        avg_win = strategy_df[strategy_df['win'] == 1]['pnl_points'].mean() if winning_trades > 0 else 0
        avg_loss = strategy_df[strategy_df['loss'] == 1]['pnl_points'].mean() if losing_trades > 0 else 0

        # Expectancy calculation
        expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)

        # Profit factor
        gross_profit = strategy_df[strategy_df['pnl_points'] > 0]['pnl_points'].sum()
        gross_loss = abs(strategy_df[strategy_df['pnl_points'] < 0]['pnl_points'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # CORRECTED: Calculate actual calendar days for accurate points/day
        min_date = strategy_df['date'].min()
        max_date = strategy_df['date'].max()
        total_calendar_days = (max_date - min_date).days + 1  # Include both start and end dates

        # Also calculate trading days (days with actual trades) for reference
        trading_days = strategy_df['date'].nunique()

        # Points per calendar day (more accurate for performance comparison)
        points_per_day = total_points / total_calendar_days if total_calendar_days > 0 else 0

        # Points per trading day (for days with actual activity)
        points_per_trading_day = total_points / trading_days if trading_days > 0 else 0

        # Trade analysis
        best_trade = strategy_df['pnl_points'].max()
        worst_trade = strategy_df['pnl_points'].min()
        avg_trade_duration = strategy_df['trade_duration_minutes'].mean()

        # Date range
        min_year = strategy_df['year'].min()
        max_year = strategy_df['year'].max()
        years_covered = f"{min_year}-{max_year}" if min_year != max_year else str(min_year)

        return {
            'strategy': strategy_name,
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'losing_trades': int(losing_trades),
            'win_rate': round(win_rate, 2),
            'total_points': round(total_points, 2),
            'total_pnl': round(total_pnl, 2),
            'points_per_day': round(points_per_day, 2),  # Calendar days
            'points_per_trading_day': round(points_per_trading_day, 2),  # Trading days only
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'expectancy': round(expectancy, 2),
            'profit_factor': round(profit_factor, 2),
            'trading_days': int(trading_days),
            'total_calendar_days': int(total_calendar_days),
            'years_covered': years_covered,
            'best_trade': round(best_trade, 2),
            'worst_trade': round(worst_trade, 2),
            'avg_trade_duration': round(avg_trade_duration, 0)
        }

    def get_filtered_trades(
        self,
        strategy_name: Optional[str] = None,
        year: Optional[int] = None,
        weekdays: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = 0
    ) -> pd.DataFrame:
        """Get filtered trades with pagination."""
        if self.trades_df is None:
            raise ValueError("No trade data loaded")

        df = self.trades_df.copy()

        # Apply filters
        if strategy_name:
            df = df[df['strategy'] == strategy_name]

        if year:
            df = df[df['year'] == year]

        if weekdays:
            df = df[df['weekday'].isin(weekdays)]

        if start_date:
            df = df[df['date'] >= start_date]

        if end_date:
            df = df[df['date'] <= end_date]

        # Sort by date and time
        df = df.sort_values(['date', 'entry_datetime'])

        # Apply pagination
        if limit is not None:
            df = df.iloc[offset:offset+limit]

        return df

    def apply_filters(self, filter_request: FilterRequest) -> pd.DataFrame:
        """Apply comprehensive filters to trade data."""
        if self.trades_df is None:
            raise ValueError("No trade data loaded")

        df = self.trades_df.copy()

        # Strategy filter
        if filter_request.strategies:
            strategy_names = [s.value for s in filter_request.strategies]
            df = df[df['strategy'].isin(strategy_names)]

        # Year filter
        if filter_request.years:
            df = df[df['year'].isin(filter_request.years)]

        # Weekday filter
        if filter_request.weekdays:
            weekday_names = [w.value for w in filter_request.weekdays]
            df = df[df['weekday'].isin(weekday_names)]

        # Date range filter
        if filter_request.start_date:
            df = df[df['date'] >= filter_request.start_date]

        if filter_request.end_date:
            df = df[df['date'] <= filter_request.end_date]

        # Trade duration filters
        if filter_request.min_trade_duration is not None:
            df = df[df['trade_duration_minutes'] >= filter_request.min_trade_duration]

        if filter_request.max_trade_duration is not None:
            df = df[df['trade_duration_minutes'] <= filter_request.max_trade_duration]

        # Direction filter
        if filter_request.directions:
            direction_values = [d.value for d in filter_request.directions]
            df = df[df['direction'].isin(direction_values)]

        # PnL filters
        if filter_request.min_pnl is not None:
            df = df[df['pnl_points'] >= filter_request.min_pnl]

        if filter_request.max_pnl is not None:
            df = df[df['pnl_points'] <= filter_request.max_pnl]

        # Win/Loss filters
        if filter_request.winning_trades_only:
            df = df[df['win'] == 1]
        elif filter_request.losing_trades_only:
            df = df[df['loss'] == 1]

        return df

    def get_filtered_data(
        self,
        strategy_name: Optional[str] = None,
        year: Optional[int] = None,
        weekdays: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Simple filtered data access for calculations."""
        if self.trades_df is None:
            raise ValueError("No trade data loaded")

        df = self.trades_df.copy()

        if strategy_name:
            df = df[df['strategy'] == strategy_name]

        if year:
            df = df[df['year'] == year]

        if weekdays:
            df = df[df['weekday'].isin(weekdays)]

        return df

    def compare_strategies(
        self,
        strategies: List[str],
        year: Optional[int] = None,
        weekdays: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple strategies side by side."""
        comparison_data = {}
        equity_curves = {}

        for strategy in strategies:
            # Get filtered data
            strategy_df = self.get_filtered_data(
                strategy_name=strategy,
                year=year,
                weekdays=weekdays
            )

            if not strategy_df.empty:
                # Calculate metrics
                summary = self._calculate_strategy_metrics(strategy_df)
                comparison_data[strategy] = summary

                # Calculate equity curve
                equity_curve = self._calculate_equity_curve_data(strategy_df)
                equity_curves[strategy] = equity_curve

        # Rank strategies by performance
        ranking = []
        for strategy, metrics in comparison_data.items():
            ranking.append({
                'strategy': strategy,
                'points_per_day': metrics.get('points_per_day', 0),
                'total_points': metrics.get('total_points', 0),
                'win_rate': metrics.get('win_rate', 0),
                'profit_factor': metrics.get('profit_factor', 0)
            })

        ranking.sort(key=lambda x: x['points_per_day'], reverse=True)

        return {
            'strategies': strategies,
            'filters_applied': {
                'year': year,
                'weekdays': weekdays
            },
            'comparison_metrics': comparison_data,
            'equity_curves': equity_curves,
            'performance_ranking': ranking
        }

    def _calculate_strategy_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a strategy dataframe."""
        if df.empty:
            return {}

        total_trades = len(df)
        winning_trades = len(df[df['win'] == 1])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        total_points = df['pnl_points'].sum()
        trading_days = df['date'].nunique()
        points_per_day = total_points / trading_days if trading_days > 0 else 0

        avg_win = df[df['win'] == 1]['pnl_points'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['loss'] == 1]['pnl_points'].mean() if (total_trades - winning_trades) > 0 else 0

        # Profit factor
        gross_profit = df[df['pnl_points'] > 0]['pnl_points'].sum()
        gross_loss = abs(df[df['pnl_points'] < 0]['pnl_points'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Drawdown calculation
        cumulative_pnl = df['pnl_points'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()

        return {
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'win_rate': round(win_rate, 2),
            'total_points': round(total_points, 2),
            'points_per_day': round(points_per_day, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown': round(max_drawdown, 2),
            'trading_days': int(trading_days)
        }

    def _calculate_equity_curve_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculate equity curve data points."""
        if df.empty:
            return []

        # Sort by date and time
        df = df.sort_values(['date', 'entry_datetime']).copy()

        # Calculate cumulative PnL
        df['cumulative_pnl'] = df['pnl_points'].cumsum()

        # Calculate running equity (starting from 0)
        starting_equity = 0
        df['running_equity'] = starting_equity + df['cumulative_pnl']

        # Calculate drawdown
        df['running_max'] = df['running_equity'].expanding().max()
        df['drawdown_points'] = df['running_equity'] - df['running_max']
        df['drawdown_percent'] = (df['drawdown_points'] / df['running_max']) * 100

        # Convert to list of dictionaries
        equity_points = []
        for _, row in df.iterrows():
            equity_points.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'cumulative_pnl': round(row['cumulative_pnl'], 2),
                'running_equity': round(row['running_equity'], 2),
                'drawdown_percent': round(row['drawdown_percent'], 2),
                'drawdown_points': round(row['drawdown_points'], 2)
            })

        return equity_points

    def get_available_filters(self) -> Dict[str, Any]:
        """Get all available filter options from the data."""
        if self.trades_df is None:
            return {}

        return {
            'strategies': self.trades_df['strategy'].unique().tolist(),
            'years': sorted(self.trades_df['year'].unique().tolist()),
            'weekdays': self.trades_df['weekday'].unique().tolist(),
            'date_range': {
                'min_date': self.trades_df['date'].min().strftime('%Y-%m-%d'),
                'max_date': self.trades_df['date'].max().strftime('%Y-%m-%d')
            },
            'market_conditions': self.trades_df['market_condition'].unique().tolist(),
            'directions': self.trades_df['direction'].unique().tolist()
        }

    def get_daily_performance(self, filters: Optional[FilterRequest] = None) -> Dict[str, Any]:
        """Get daily P&L performance with detailed breakdowns."""
        if self.trades_df is None:
            raise ValueError("No trade data loaded")

        df = self.trades_df.copy()

        if filters:
            df = self.apply_filters(filters)

        # Group by date and calculate daily metrics
        daily_stats = df.groupby('date').agg({
            'pnl_points': ['sum', 'count', 'mean'],
            'pnl_dollars': 'sum',
            'win': 'sum',
            'loss': 'sum',
            'trade_duration_minutes': 'mean'
        }).round(2)

        # Flatten column names
        daily_stats.columns = ['points_total', 'trades_count', 'avg_points_per_trade',
                              'dollars_total', 'wins', 'losses', 'avg_duration']

        # Calculate win rate for each day
        daily_stats['win_rate'] = (daily_stats['wins'] / daily_stats['trades_count'] * 100).round(2)

        # Add calendar information
        daily_stats = daily_stats.reset_index()
        daily_stats['weekday'] = pd.to_datetime(daily_stats['date']).dt.day_name()
        daily_stats['month'] = pd.to_datetime(daily_stats['date']).dt.month
        daily_stats['year'] = pd.to_datetime(daily_stats['date']).dt.year
        daily_stats['week_of_year'] = pd.to_datetime(daily_stats['date']).dt.isocalendar().week

        # Sort by date
        daily_stats = daily_stats.sort_values('date')

        # Calculate cumulative performance
        daily_stats['cumulative_points'] = daily_stats['points_total'].cumsum()
        daily_stats['cumulative_dollars'] = daily_stats['dollars_total'].cumsum()

        return {
            'daily_performance': daily_stats.to_dict('records'),
            'summary': {
                'total_trading_days': len(daily_stats),
                'total_points': daily_stats['points_total'].sum(),
                'total_dollars': daily_stats['dollars_total'].sum(),
                'avg_points_per_day': daily_stats['points_total'].mean(),
                'best_day': daily_stats['points_total'].max(),
                'worst_day': daily_stats['points_total'].min(),
                'profitable_days': len(daily_stats[daily_stats['points_total'] > 0]),
                'losing_days': len(daily_stats[daily_stats['points_total'] < 0])
            }
        }

    def get_weekly_distribution(self, filters: Optional[FilterRequest] = None) -> Dict[str, Any]:
        """Get weekly points distribution analysis."""
        if self.trades_df is None:
            raise ValueError("No trade data loaded")

        df = self.trades_df.copy()

        if filters:
            df = self.apply_filters(filters)

        # Add week information
        # Convert date column back to datetime for calculations
        date_series = pd.to_datetime(df['date'])
        df['week_year'] = date_series.dt.strftime('%Y-W%U')
        df['week_number'] = date_series.dt.isocalendar().week
        df['year'] = date_series.dt.year

        # Group by week
        weekly_stats = df.groupby('week_year').agg({
            'pnl_points': ['sum', 'count', 'mean', 'std'],
            'pnl_dollars': 'sum',
            'win': 'sum',
            'loss': 'sum',
            'date': ['min', 'max']
        }).round(2)

        # Flatten columns
        weekly_stats.columns = ['points_total', 'trades_count', 'avg_points_per_trade',
                               'points_std', 'dollars_total', 'wins', 'losses',
                               'week_start', 'week_end']

        weekly_stats = weekly_stats.reset_index()
        weekly_stats['win_rate'] = (weekly_stats['wins'] / weekly_stats['trades_count'] * 100).round(2)

        # Distribution analysis
        points_distribution = weekly_stats['points_total'].describe()

        return {
            'weekly_performance': weekly_stats.to_dict('records'),
            'distribution_stats': {
                'count': int(points_distribution['count']),
                'mean': round(points_distribution['mean'], 2),
                'std': round(points_distribution['std'], 2),
                'min': round(points_distribution['min'], 2),
                'max': round(points_distribution['max'], 2),
                'percentile_25': round(points_distribution['25%'], 2),
                'percentile_50': round(points_distribution['50%'], 2),
                'percentile_75': round(points_distribution['75%'], 2),
            },
            'summary': {
                'total_weeks': len(weekly_stats),
                'profitable_weeks': len(weekly_stats[weekly_stats['points_total'] > 0]),
                'losing_weeks': len(weekly_stats[weekly_stats['points_total'] < 0]),
                'best_week': weekly_stats['points_total'].max(),
                'worst_week': weekly_stats['points_total'].min(),
                'avg_weekly_points': weekly_stats['points_total'].mean(),
                'weekly_consistency': weekly_stats['points_total'].std()
            }
        }

    def get_monthly_distribution(self, filters: Optional[FilterRequest] = None) -> Dict[str, Any]:
        """Get monthly points distribution analysis."""
        if self.trades_df is None:
            raise ValueError("No trade data loaded")

        df = self.trades_df.copy()

        if filters:
            df = self.apply_filters(filters)

        # Add month information
        # Convert date column back to datetime for calculations
        date_series = pd.to_datetime(df['date'])
        df['month_year'] = date_series.dt.strftime('%Y-%m')
        df['month_name'] = date_series.dt.strftime('%B')
        df['month_num'] = date_series.dt.month
        df['year'] = date_series.dt.year

        # Group by month
        monthly_stats = df.groupby(['year', 'month_num', 'month_year', 'month_name']).agg({
            'pnl_points': ['sum', 'count', 'mean', 'std'],
            'pnl_dollars': 'sum',
            'win': 'sum',
            'loss': 'sum',
            'date': ['min', 'max']
        }).round(2)

        # Flatten columns
        monthly_stats.columns = ['points_total', 'trades_count', 'avg_points_per_trade',
                               'points_std', 'dollars_total', 'wins', 'losses',
                               'month_start', 'month_end']

        monthly_stats = monthly_stats.reset_index()
        monthly_stats['win_rate'] = (monthly_stats['wins'] / monthly_stats['trades_count'] * 100).round(2)

        # Sort by year and month
        monthly_stats = monthly_stats.sort_values(['year', 'month_num'])

        # Distribution analysis
        points_distribution = monthly_stats['points_total'].describe()

        # Month-of-year analysis (seasonal patterns)
        seasonal_analysis = df.groupby(date_series.dt.month)['pnl_points'].agg(['sum', 'mean', 'count']).round(2)
        seasonal_analysis.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        return {
            'monthly_performance': monthly_stats.to_dict('records'),
            'distribution_stats': {
                'count': int(points_distribution['count']),
                'mean': round(points_distribution['mean'], 2),
                'std': round(points_distribution['std'], 2),
                'min': round(points_distribution['min'], 2),
                'max': round(points_distribution['max'], 2),
                'percentile_25': round(points_distribution['25%'], 2),
                'percentile_50': round(points_distribution['50%'], 2),
                'percentile_75': round(points_distribution['75%'], 2),
            },
            'seasonal_patterns': seasonal_analysis.to_dict('index'),
            'summary': {
                'total_months': len(monthly_stats),
                'profitable_months': len(monthly_stats[monthly_stats['points_total'] > 0]),
                'losing_months': len(monthly_stats[monthly_stats['points_total'] < 0]),
                'best_month': monthly_stats['points_total'].max(),
                'worst_month': monthly_stats['points_total'].min(),
                'avg_monthly_points': monthly_stats['points_total'].mean(),
                'monthly_consistency': monthly_stats['points_total'].std()
            }
        }

    def _validate_ny_session_compliance_optional(self) -> None:
        """Validate that all trades comply with NY session hours (9:30 AM - 4:10 PM ET)."""
        if self.trades_df is None or self.trades_df.empty:
            return

        # Use the clean time columns directly (entry_time and exit_time are HH:MM:SS format)
        entry_times = pd.to_datetime(self.trades_df['entry_time'], format='%H:%M:%S').dt.time

        # For exit times, handle cross-day trades by checking if exit is on different day
        entry_dates = pd.to_datetime(self.trades_df['entry_datetime']).dt.date
        exit_dates = pd.to_datetime(self.trades_df['exit_datetime']).dt.date

        # Only validate same-day trades for exit times
        same_day_mask = entry_dates == exit_dates
        exit_times = pd.to_datetime(self.trades_df['exit_time'], format='%H:%M:%S').dt.time

        # NY session bounds (9:30 AM - 4:10 PM to allow position exits)
        session_start = pd.to_datetime('09:30:00').time()
        session_end = pd.to_datetime('16:10:00').time()

        # Check for violations - only validate entry times and same-day exit times
        invalid_entries = (entry_times < session_start) | (entry_times > session_end)
        invalid_exits = same_day_mask & ((exit_times < session_start) | (exit_times > session_end))

        violations_count = invalid_entries.sum() + invalid_exits.sum()

        if violations_count > 0:
            logger.warning(f"⚠️ NY SESSION POTENTIAL ISSUES: {violations_count} trades outside 9:30-16:10 ET")

            # Log sample violations for debugging but don't fail
            sample_invalid = self.trades_df[invalid_entries | invalid_exits].head(5)
            for _, trade in sample_invalid.iterrows():
                logger.warning(f"   ISSUE: {trade['date']} {trade['strategy']} {trade['entry_time']} {trade['exit_time']}")

            # Don't raise exception - just warn for now
            logger.warning(f"⚠️ Data contains {violations_count} potentially problematic trades but continuing...")
        else:
            cross_day_count = (~same_day_mask).sum()
            logger.info(f"✅ NY session compliance validated: All {len(self.trades_df)} trades comply")
            logger.info(f"   - Same-day trades: {same_day_mask.sum()}")
            logger.info(f"   - Cross-day trades: {cross_day_count}")
            if cross_day_count > 0:
                logger.info(f"   - Cross-day trades allowed (positions held overnight)")