"""
Pydantic Models for Trading Dashboard API
=======================================
Data models for request/response validation and serialization
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import date, datetime
from enum import Enum

class WeekDay(str, Enum):
    """Valid weekdays for filtering."""
    MONDAY = "Monday"
    TUESDAY = "Tuesday"
    WEDNESDAY = "Wednesday"
    THURSDAY = "Thursday"
    FRIDAY = "Friday"

class StrategyName(str, Enum):
    """Available trading strategies."""
    CHAMPION = "Champion"
    STABLE = "Stable"
    BALANCED = "Balanced"

class Direction(str, Enum):
    """Trade directions."""
    LONG = "LONG"
    SHORT = "SHORT"

# Request Models
class FilterRequest(BaseModel):
    """Request model for filtering trades and calculations."""
    strategies: Optional[List[StrategyName]] = Field(None, description="List of strategies to include")
    years: Optional[List[int]] = Field(None, description="Years to include (2020-2025)")
    weekdays: Optional[List[WeekDay]] = Field(None, description="Weekdays to include")
    start_date: Optional[date] = Field(None, description="Start date filter (YYYY-MM-DD)")
    end_date: Optional[date] = Field(None, description="End date filter (YYYY-MM-DD)")
    min_trade_duration: Optional[int] = Field(None, description="Minimum trade duration in minutes")
    max_trade_duration: Optional[int] = Field(None, description="Maximum trade duration in minutes")
    directions: Optional[List[Direction]] = Field(None, description="Trade directions to include")
    min_pnl: Optional[float] = Field(None, description="Minimum PnL filter")
    max_pnl: Optional[float] = Field(None, description="Maximum PnL filter")
    winning_trades_only: Optional[bool] = Field(False, description="Include only winning trades")
    losing_trades_only: Optional[bool] = Field(False, description="Include only losing trades")

    @validator('years')
    def validate_years(cls, v):
        if v is not None:
            for year in v:
                if year < 2020 or year > 2025:
                    raise ValueError('Years must be between 2020 and 2025')
        return v

    @validator('winning_trades_only', 'losing_trades_only')
    def validate_win_loss_filter(cls, v, values):
        if v and values.get('winning_trades_only') and values.get('losing_trades_only'):
            raise ValueError('Cannot filter for both winning and losing trades only')
        return v

# Response Models
class TradeResponse(BaseModel):
    """Individual trade data response."""
    strategy: str
    year: int
    date: str
    entry_time: str
    exit_time: str
    entry_datetime: str
    exit_datetime: str
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    points_captured: float
    max_potential: float
    capture_ratio: float
    pnl_points: float
    pnl_dollars: float
    commission: float
    trade_duration_minutes: float
    weekday: str
    hour: int
    market_condition: str
    trade_number: int
    reason: str
    win: int
    loss: int
    quantity: int
    partial_profit_1: Optional[float]
    partial_profit_1_pct: Optional[float]

class StrategyResponse(BaseModel):
    """Strategy summary response."""
    strategy: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_points: float
    total_pnl: float
    points_per_day: float
    avg_win: float
    avg_loss: float
    expectancy: float
    profit_factor: float
    trading_days: int
    years_covered: str
    best_trade: float
    worst_trade: float
    avg_trade_duration: float

class StatisticsResponse(BaseModel):
    """Comprehensive statistics response."""
    # Basic metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L metrics
    total_pnl_points: float
    total_pnl_dollars: float
    points_per_day: float
    avg_trade_points: float
    avg_winning_trade: float
    avg_losing_trade: float

    # Risk metrics
    expectancy: float
    profit_factor: float
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    calmar_ratio: Optional[float]
    max_drawdown_percent: float
    max_drawdown_points: float

    # Streak analysis
    max_consecutive_wins: int
    max_consecutive_losses: int
    current_streak: int
    current_streak_type: str  # "wins" or "losses"

    # Distribution metrics
    best_trade: float
    worst_trade: float
    median_trade: float
    std_deviation: float

    # Time-based metrics
    total_trading_days: int
    avg_trades_per_day: float
    avg_trade_duration_minutes: float

    # Additional metrics
    recovery_factor: Optional[float]
    ulcer_index: Optional[float]
    var_95: Optional[float]  # Value at Risk 95%

class EquityCurvePoint(BaseModel):
    """Single point on equity curve."""
    date: str
    cumulative_pnl: float
    running_equity: float
    drawdown_percent: float
    drawdown_points: float

class EquityCurveResponse(BaseModel):
    """Equity curve data response."""
    strategy: Optional[str]
    total_points: float
    max_equity: float
    final_equity: float
    max_drawdown: float
    curve_data: List[EquityCurvePoint]

class DrawdownPeriod(BaseModel):
    """Drawdown period information."""
    start_date: str
    end_date: str
    duration_days: int
    max_drawdown_percent: float
    max_drawdown_points: float
    recovery_date: Optional[str]
    recovery_duration_days: Optional[int]

class DrawdownAnalysis(BaseModel):
    """Detailed drawdown analysis."""
    max_drawdown_percent: float
    max_drawdown_points: float
    current_drawdown_percent: float
    current_drawdown_points: float
    drawdown_periods: List[DrawdownPeriod]
    average_drawdown_duration: float
    average_recovery_time: float

class MonthlyReturn(BaseModel):
    """Monthly return data."""
    year: int
    month: int
    month_name: str
    pnl_points: float
    pnl_dollars: float
    trades_count: int
    win_rate: float

class PerformanceByPeriod(BaseModel):
    """Performance breakdown by time periods."""
    yearly: Dict[str, Dict[str, Union[float, int]]]
    monthly: List[MonthlyReturn]
    by_weekday: Dict[str, Dict[str, Union[float, int]]]
    by_hour: Dict[str, Dict[str, Union[float, int]]]

class ComparisonResponse(BaseModel):
    """Strategy comparison response."""
    strategies: List[str]
    filters_applied: Dict[str, Any]
    comparison_metrics: Dict[str, Dict[str, Union[float, int]]]
    equity_curves: Dict[str, List[EquityCurvePoint]]
    performance_ranking: List[Dict[str, Union[str, float]]]

class FilterOptions(BaseModel):
    """Available filter options."""
    strategies: List[str]
    years: List[int]
    weekdays: List[str]
    date_range: Dict[str, str]  # min_date, max_date
    market_conditions: List[str]
    directions: List[str]

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    data_loaded: bool
    total_trades: int

class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str
    error_code: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# Utility models for data processing
class TradeFilters(BaseModel):
    """Internal model for trade filtering."""
    strategy_name: Optional[str] = None
    year: Optional[int] = None
    weekdays: Optional[List[str]] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    direction: Optional[str] = None
    min_duration: Optional[int] = None
    max_duration: Optional[int] = None
    profitable_only: Optional[bool] = None

class CalculationResult(BaseModel):
    """Generic calculation result."""
    metric_name: str
    value: Union[float, int, str]
    calculation_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    data_points: int
    filters_applied: Optional[Dict[str, Any]] = None