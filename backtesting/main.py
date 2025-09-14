"""
Main Backtesting Engine
=======================
Advanced quantitative backtesting system with event-driven architecture.
Orchestrates data handling, strategy execution, and performance analysis.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Type
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from data_handler import DataHandler
from strategy import Strategy, ORBStrategy
from portfolio import Portfolio
from performance import PerformanceCalculator

class BacktestEngine:
    """
    Main backtesting engine orchestrating all components.
    Event-driven architecture for realistic simulation.
    """

    def __init__(self, data_path: str, strategy: Strategy,
                 initial_capital: float = 100000.0,
                 point_value: float = 20.0,
                 commission_per_trade: float = 2.50,
                 timeframe: str = "15min",
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None):
        """
        Initialize backtesting engine.

        Args:
            data_path: Path to market data file
            strategy: Trading strategy instance
            initial_capital: Starting capital
            point_value: Dollar value per point (for futures)
            commission_per_trade: Round-trip commission cost
            timeframe: Data timeframe (1min, 5min, 15min, etc.)
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
        """
        self.data_handler = DataHandler(
            data_path=data_path,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        self.strategy = strategy

        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            point_value=point_value,
            commission_per_trade=commission_per_trade,
            max_positions=1  # Single position for most strategies
        )

        self.performance_calc = PerformanceCalculator()

        # Tracking variables
        self.total_bars_processed = 0
        self.signals_generated = 0
        self.trades_executed = 0

    def run_backtest(self, verbose: bool = True, progress_interval: int = 1000) -> dict:
        """
        Run complete backtest with event-driven simulation.

        Args:
            verbose: Print progress updates
            progress_interval: Print progress every N bars

        Returns:
            Dictionary with backtest results and performance metrics
        """
        if verbose:
            print("\n🚀 Starting Backtesting Engine")
            print("=" * 50)

        # Load and prepare data
        start_time = time.time()
        self.data_handler.load_data()

        if verbose:
            print(f"⏱️  Data loaded in {time.time() - start_time:.2f} seconds")

        # Main event-driven loop
        loop_start = time.time()

        try:
            for bar_data in self.data_handler:
                self._process_bar(bar_data, verbose, progress_interval)

        except KeyboardInterrupt:
            print("\n⏹️  Backtest interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during backtest: {e}")
            raise

        # Calculate final performance
        final_metrics = self._finalize_backtest(verbose, loop_start)

        return final_metrics

    def _process_bar(self, bar_data: dict, verbose: bool, progress_interval: int) -> None:
        """Process single bar in event-driven loop."""
        self.total_bars_processed += 1

        # Step 1: Update existing positions and check exits
        closed_positions = self.portfolio.update_positions(bar_data)
        if closed_positions:
            self.trades_executed += len(closed_positions)

        # Step 2: Generate strategy signal
        signal = self.strategy.generate_signal(bar_data, self.portfolio)

        # Step 3: Validate and execute signal
        if self.strategy.validate_signal(signal) and signal['signal'] != 'HOLD':
            # Add current price to signal for execution
            signal['entry_price'] = bar_data['close']
            signal['risk_per_trade'] = self.strategy.risk_per_trade

            position_id = self.portfolio.execute_trade(signal)
            if position_id:
                self.signals_generated += 1

        # Step 4: Record equity curve
        self.portfolio.calculate_equity_curve()

        # Progress updates
        if verbose and self.total_bars_processed % progress_interval == 0:
            progress = self.data_handler.progress
            open_pos = len(self.portfolio.positions)
            total_trades = len(self.portfolio.closed_trades)

            print(f"📊 Progress: {progress:.1f}% | "
                  f"Bars: {self.total_bars_processed:,} | "
                  f"Trades: {total_trades} | "
                  f"Open: {open_pos}")

    def _finalize_backtest(self, verbose: bool, loop_start: float) -> dict:
        """Finalize backtest and calculate performance metrics."""
        # Close any remaining open positions at market
        final_bar = {
            'timestamp': self.portfolio.current_timestamp,
            'close': 0,  # Will use current unrealized P&L
            'high': 0,
            'low': 0
        }

        # Force close remaining positions
        for position_id in list(self.portfolio.positions.keys()):
            position = self.portfolio.positions[position_id]
            if position.unrealized_pnl != 0:  # Has P&L to realize
                final_bar['close'] = position.entry_price + (position.unrealized_pnl / (position.quantity * self.portfolio.point_value))
                closed_trade = self.portfolio._close_position(position, final_bar, "End of Backtest")
                self.portfolio.closed_trades.append(closed_trade)
                del self.portfolio.positions[position_id]

        # Calculate comprehensive performance metrics
        performance_metrics = self.performance_calc.calculate_comprehensive_metrics(self.portfolio)

        # Add execution statistics
        execution_stats = {
            'execution_time_seconds': time.time() - loop_start,
            'bars_processed': self.total_bars_processed,
            'signals_generated': self.signals_generated,
            'bars_per_second': self.total_bars_processed / (time.time() - loop_start),
            'strategy_name': self.strategy.name
        }

        performance_metrics.update(execution_stats)

        if verbose:
            print(f"\n✅ Backtest completed in {execution_stats['execution_time_seconds']:.2f} seconds")
            print(f"📈 Processed {self.total_bars_processed:,} bars at {execution_stats['bars_per_second']:.0f} bars/sec")
            print(f"📊 Generated {self.signals_generated} signals, executed {len(self.portfolio.closed_trades)} trades")

        return performance_metrics

    def generate_report(self) -> str:
        """Generate comprehensive performance report."""
        return self.performance_calc.generate_performance_report(self.portfolio)

    def get_trades_dataframe(self):
        """Get trades as DataFrame for analysis."""
        return self.portfolio.get_trades_dataframe()

    def get_equity_curve(self):
        """Get equity curve as DataFrame for plotting."""
        return self.portfolio.get_equity_series()


def run_orb_backtest(data_path: str,
                    initial_capital: float = 100000.0,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    risk_per_trade: float = 0.02,
                    or_minutes: int = 15) -> dict:
    """
    Convenience function to run ORB strategy backtest.

    Args:
        data_path: Path to NQ 15min data file
        initial_capital: Starting capital
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        risk_per_trade: Risk per trade (0.02 = 2%)
        or_minutes: Opening range minutes

    Returns:
        Performance metrics dictionary
    """
    # Initialize ORB strategy
    strategy = ORBStrategy(risk_per_trade=risk_per_trade, or_minutes=or_minutes)

    # Create backtest engine
    engine = BacktestEngine(
        data_path=data_path,
        strategy=strategy,
        initial_capital=initial_capital,
        point_value=20.0,  # NQ point value
        commission_per_trade=2.50,
        timeframe="15min",
        start_date=start_date,
        end_date=end_date
    )

    # Run backtest
    results = engine.run_backtest(verbose=True)

    # Print performance report
    print(engine.generate_report())

    return results


if __name__ == "__main__":
    """
    Example usage of the backtesting system.
    """
    # Example: Run ORB strategy on NQ data
    DATA_PATH = "/Users/shubhamshanker/bt_/nq_data_15min.csv"  # Update path as needed

    print("🔬 Running Example ORB Backtest")
    print("=" * 40)

    results = run_orb_backtest(
        data_path=DATA_PATH,
        initial_capital=100000.0,
        start_date="2020-01-01",
        end_date="2023-12-31",
        risk_per_trade=0.02,
        or_minutes=15
    )

    print(f"\n🎯 Final Results Summary:")
    print(f"Total Return: {results.get('total_return_percent', 0):.2f}%")
    print(f"Win Rate: {results.get('win_rate', 0):.2f}%")
    print(f"Total Trades: {results.get('total_trades', 0)}")
    print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
    print(f"Max Drawdown: {results.get('max_drawdown_percent', 0):.2f}%")