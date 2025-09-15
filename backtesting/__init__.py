"""
Advanced Quantitative Backtesting System
========================================
Professional-grade event-driven backtesting framework for trading strategies.

Modules:
- data_handler: Multi-timeframe data loading and streaming
- strategy: Abstract strategy base classes and implementations
- portfolio: Position management and risk control
- performance: Comprehensive performance analysis
- main: Orchestration engine for backtests

Example Usage:
    from backtesting.main import run_orb_backtest

    results = run_orb_backtest(
        data_path=get_data_path("5min"),
        initial_capital=100000,
        start_date="2020-01-01",
        end_date="2023-12-31"
    )
"""

from .data_handler import DataHandler
from .parquet_data_handler import ParquetDataHandler
from .strategy import Strategy, ORBStrategy
from .portfolio import Portfolio, Position, Trade
from .performance import PerformanceCalculator
from .main import BacktestEngine, run_orb_backtest
from .ultimate_orb_strategy import UltimateORBStrategy

__version__ = "1.0.0"
__author__ = "Quantitative Trading System"

__all__ = [
    'DataHandler',
    'ParquetDataHandler',
    'Strategy', 'ORBStrategy',
    'UltimateORBStrategy',
    'Portfolio', 'Position', 'Trade',
    'PerformanceCalculator',
    'BacktestEngine', 'run_orb_backtest'
]