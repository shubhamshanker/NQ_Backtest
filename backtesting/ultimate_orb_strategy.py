"""
Ultimate ORB Strategy - Comprehensive Optimization for 30+ Points
================================================================
- MAX 3 trades per day ONLY
- Risk: 1-10% per trade OR points-based 10-100 points
- R:R ratios: 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5
- ORB periods: 15min, 30min, 12min, 45min, 1hr, 90min
- Half-size booking, trailing stops
- Single contract only
- Robust loss prevention
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
import pandas as pd
from datetime import datetime, time, timedelta
import numpy as np

if TYPE_CHECKING:
    from portfolio import Portfolio

from strategy import Strategy

class UltimateORBStrategy(Strategy):
    """Ultimate ORB strategy with all requested optimizations."""

    def __init__(self, risk_per_trade: float = 0.02, or_minutes: int = 15,
                 fixed_stop_points: float = 50.0, target_multiplier: float = 1.5,
                 max_trades_per_day: int = 3, risk_in_points: Optional[float] = None,
                 half_size_booking: bool = True, trailing_stop: bool = True,
                 trailing_bars: int = 2, intraday_only: bool = True):
        """
        Initialize Ultimate ORB Strategy.

        Args:
            risk_per_trade: Risk as percentage (1-10% = 0.01-0.10)
            risk_in_points: Alternative - risk in points (10-100)
            or_minutes: Opening range period (12, 15, 30, 45, 60, 90)
            fixed_stop_points: Stop loss in points
            target_multiplier: R:R ratio (0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5)
            max_trades_per_day: Maximum trades per day (1-3 only)
            half_size_booking: Book half position at 50% of target
            trailing_stop: Use trailing stop loss
        """
        super().__init__(risk_per_trade)
        self.or_minutes = or_minutes
        self.fixed_stop = fixed_stop_points
        self.target_mult = target_multiplier
        self.max_trades_per_day = min(max_trades_per_day, 3)  # Hard cap at 3
        self.risk_in_points = risk_in_points
        self.half_size_booking = half_size_booking
        self.trailing_stop = trailing_stop
        self.trailing_bars = trailing_bars
        self.intraday_only = intraday_only

        # Always 1 contract
        self.contracts_per_trade = 1

        # State tracking
        self.daily_ranges = {}
        self.daily_trade_count = {}
        self.position_states = {}
        self.price_history = []
        self.daily_pnl = {}
        self.losing_days = []
        self.current_date = None

        # Name for identification
        risk_str = f"R{risk_in_points}pts" if risk_in_points else f"R{risk_per_trade*100:.0f}%"
        self.name = f"Ultimate_OR{or_minutes}_{risk_str}_TM{target_multiplier}_MT{max_trades_per_day}"

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """Generate ultimate ORB signals with all optimizations."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()

        signal = {
            'signal': 'HOLD',
            'entry_time': timestamp,
            'stop_loss': 0,
            'take_profit': 0,
            'reason': 'No signal',
            'contracts': 1,
            'half_size_booking': self.half_size_booking
        }

        # CRITICAL: Validate NY session timing before any signal generation
        if not self._validate_session_timing(bar_data, allow_extended=False):
            signal['reason'] = f'Outside regular trading session: {timestamp.time()}'
            return signal

        # Update daily data
        if current_date != self.current_date:
            self.current_date = current_date
            self._reset_daily_data(current_date)

        self._update_opening_range(bar_data)
        self._update_price_history(bar_data)

        # Check OR data
        if current_date not in self.daily_ranges:
            return signal

        or_data = self.daily_ranges[current_date]
        if or_data['bars_counted'] == 0:
            return signal

        # Trading hours - dynamic based on OR period
        current_time = timestamp.time()
        or_end_time = self._calculate_or_end_time()
        trade_start = or_end_time
        close_cutoff = time(15, 45) if self.intraday_only else time(16, 0)

        if not (trade_start <= current_time <= close_cutoff):
            return signal

        # Hard limit: MAX 3 trades per day
        if self.daily_trade_count[current_date] >= self.max_trades_per_day:
            return signal

        # Handle existing positions with advanced management
        existing_positions = getattr(portfolio, 'positions', {})
        if existing_positions:
            self._manage_existing_positions(bar_data, portfolio)
            return signal  # One position at a time

        # Avoid losing days pattern
        if self._is_high_risk_day(bar_data):
            return signal

        # Generate entry signals
        current_price = bar_data['close']
        or_high = or_data['high']
        or_low = or_data['low']
        or_range = or_high - or_low

        # Dynamic position sizing
        position_size = self._calculate_position_size(current_price)

        # Long breakout
        if current_price > or_high:
            target_points = self.fixed_stop * self.target_mult

            # Enhanced reason with session timing info
            is_prime_time = self._is_orb_prime_time(timestamp)
            session_info = "ORB Prime Time" if is_prime_time else "Regular Session"

            signal = {
                'signal': 'BUY',
                'entry_time': timestamp,
                'entry_price': current_price,
                'stop_loss': current_price - self.fixed_stop,
                'take_profit': current_price + target_points,
                'risk_per_trade': self.risk_per_trade,
                'contracts': 1,
                'position_size': position_size,
                'half_target': current_price + (target_points * 0.5) if self.half_size_booking else None,
                'reason': f'Ultimate Long breakout above {or_high:.2f}, OR range: {or_range:.1f} ({session_info})',
                'or_range': or_range,
                'direction': 'LONG'
            }
            self.daily_trade_count[current_date] += 1

        # Short breakdown
        elif current_price < or_low:
            target_points = self.fixed_stop * self.target_mult

            # Enhanced reason with session timing info
            is_prime_time = self._is_orb_prime_time(timestamp)
            session_info = "ORB Prime Time" if is_prime_time else "Regular Session"

            signal = {
                'signal': 'SELL',
                'entry_time': timestamp,
                'entry_price': current_price,
                'stop_loss': current_price + self.fixed_stop,
                'take_profit': current_price - target_points,
                'risk_per_trade': self.risk_per_trade,
                'contracts': 1,
                'position_size': position_size,
                'half_target': current_price - (target_points * 0.5) if self.half_size_booking else None,
                'reason': f'Ultimate Short breakdown below {or_low:.2f}, OR range: {or_range:.1f} ({session_info})',
                'or_range': or_range,
                'direction': 'SHORT'
            }
            self.daily_trade_count[current_date] += 1

        return signal

    def _calculate_position_size(self, current_price: float) -> float:
        """Calculate position size based on risk parameters."""
        if self.risk_in_points:
            # Points-based risk (10-100 points)
            risk_amount = self.risk_in_points * 20  # Convert to dollars
            return min(risk_amount / (self.fixed_stop * 20), 1.0)  # Cap at 1 contract
        else:
            # Percentage-based risk (1-10%)
            return 1.0  # Always 1 contract

    def _manage_existing_positions(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> None:
        """Manage existing positions with trailing stops and half-size booking."""
        current_price = bar_data['close']

        for pos_id, position in getattr(portfolio, 'positions', {}).items():
            pos_data = self.position_states.get(pos_id, {})

            # Half-size booking at 50% of target
            if (self.half_size_booking and not pos_data.get('half_booked', False) and
                hasattr(position, 'half_target')):

                if hasattr(position, 'direction'):
                    if (position.direction == 'LONG' and current_price >= position.half_target):
                        pos_data['half_booked'] = True
                        # Simulate half-size booking
                        print(f"Half-size booked at {current_price:.2f}")
                    elif (position.direction == 'SHORT' and current_price <= position.half_target):
                        pos_data['half_booked'] = True
                        print(f"Half-size booked at {current_price:.2f}")

            # Trailing stop management
            if self.trailing_stop and len(self.price_history) >= self.trailing_bars:
                self._update_trailing_stop(position, pos_data)

            self.position_states[pos_id] = pos_data

    def _update_trailing_stop(self, position, pos_data: Dict) -> None:
        """Update trailing stop based on recent price action."""
        if len(self.price_history) < self.trailing_bars:
            return

        recent_bars = self.price_history[-self.trailing_bars:]

        if hasattr(position, 'direction'):
            if position.direction == 'LONG':
                # Trail stop below recent lows
                recent_low = min(bar['low'] for bar in recent_bars)
                trail_stop = recent_low - 3.0  # 3-point buffer

                # Only move stop up, never down
                if trail_stop > position.stop_loss:
                    position.stop_loss = trail_stop

            elif position.direction == 'SHORT':
                # Trail stop above recent highs
                recent_high = max(bar['high'] for bar in recent_bars)
                trail_stop = recent_high + 3.0  # 3-point buffer

                # Only move stop down (less restrictive for shorts)
                if trail_stop < position.stop_loss:
                    position.stop_loss = trail_stop

    def _is_high_risk_day(self, bar_data: Dict[str, Any]) -> bool:
        """Identify high-risk days to avoid based on patterns."""
        current_date = bar_data['timestamp'].date()

        # Avoid trading on days following big losing days
        yesterday = current_date - timedelta(days=1)
        if yesterday in self.daily_pnl and self.daily_pnl[yesterday] < -20:
            return True

        # Avoid low volatility days (small OR)
        if current_date in self.daily_ranges:
            or_data = self.daily_ranges[current_date]
            if or_data['bars_counted'] > 0:
                or_range = or_data['high'] - or_data['low']
                if or_range < 20:  # Avoid days with < 20 point OR
                    return True

        return False

    def _calculate_or_end_time(self) -> time:
        """Calculate when OR period ends based on OR minutes."""
        market_open = datetime.strptime("09:30", "%H:%M")
        or_end = market_open + timedelta(minutes=self.or_minutes)
        return or_end.time()

    def _update_opening_range(self, bar_data: Dict[str, Any]) -> None:
        """Update opening range based on OR period."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()
        current_time = timestamp.time()

        if current_date not in self.daily_ranges:
            self._reset_daily_data(current_date)

        # OR calculation based on period
        market_open = time(9, 30)
        or_end_time = self._calculate_or_end_time()

        if market_open <= current_time <= or_end_time:
            or_data = self.daily_ranges[current_date]

            if or_data['bars_counted'] == 0:
                or_data['high'] = bar_data['high']
                or_data['low'] = bar_data['low']
                or_data['bars_counted'] = 1
            else:
                or_data['high'] = max(or_data['high'], bar_data['high'])
                or_data['low'] = min(or_data['low'], bar_data['low'])
                or_data['bars_counted'] += 1

    def _update_price_history(self, bar_data: Dict[str, Any]) -> None:
        """Update price history for trailing stops."""
        self.price_history.append({
            'timestamp': bar_data['timestamp'],
            'high': bar_data['high'],
            'low': bar_data['low'],
            'close': bar_data['close']
        })

        # Keep only last 10 bars
        if len(self.price_history) > 10:
            self.price_history = self.price_history[-10:]

    def _reset_daily_data(self, current_date) -> None:
        """Reset daily data."""
        self.daily_ranges[current_date] = {
            'high': float('-inf'),
            'low': float('inf'),
            'bars_counted': 0
        }
        self.daily_trade_count[current_date] = 0
        self.daily_pnl[current_date] = 0

    def update_daily_pnl(self, date, pnl: float) -> None:
        """Update daily P&L for loss pattern analysis."""
        self.daily_pnl[date] = pnl
        if pnl < -15:  # Track losing days
            self.losing_days.append(date)