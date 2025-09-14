"""
ORB Reversion Strategy - Mean Reversion Logic
=============================================
Counter-trend strategy that enters when price reverts back toward opening range
after initial breakout attempts fail. Uses next-candle entry with proper risk management.
"""

from typing import Dict, Any, TYPE_CHECKING
import pandas as pd
from datetime import datetime, time

if TYPE_CHECKING:
    from portfolio import Portfolio

from strategy import Strategy

class ORBReversionStrategy(Strategy):
    """
    Opening Range Reversion Strategy with mean reversion logic.

    Logic:
    1. Calculate opening range for first N minutes
    2. Wait for initial breakout attempt (price moves beyond OR)
    3. Enter when price reverts back toward OR (failed breakout)
    4. Target is OR center or opposite OR boundary
    5. Stop beyond the breakout failure point
    6. Dynamic position sizing based on risk percentage
    """

    def __init__(self, risk_per_trade: float = 0.02, or_minutes: int = 15,
                 fixed_stop_points: float = 80.0, reversion_target_ratio: float = 1.5,
                 max_trades_per_day: int = 3, intraday_only: bool = True,
                 breakout_threshold: float = 0.5):
        """
        Initialize ORB Reversion Strategy.

        Args:
            risk_per_trade: Risk as percentage of capital (0.01 = 1%)
            or_minutes: Opening range period in minutes
            fixed_stop_points: Stop loss distance in points
            reversion_target_ratio: Target multiplier for mean reversion
            max_trades_per_day: Maximum trades per day
            intraday_only: Close all positions before market close
            breakout_threshold: Minimum breakout distance to consider reversion (ratio of OR range)
        """
        super().__init__(risk_per_trade)
        self.or_minutes = or_minutes
        self.fixed_stop = fixed_stop_points
        self.target_ratio = reversion_target_ratio
        self.max_trades_per_day = max_trades_per_day
        self.intraday_only = intraday_only
        self.breakout_threshold = breakout_threshold

        # State tracking
        self.daily_ranges = {}
        self.daily_trade_count = {}
        self.pending_signals = {}  # Store signals for next bar entry
        self.breakout_tracker = {}  # Track breakout attempts
        self.current_date = None

        self.name = f"ORB_Reversion_OR{or_minutes}_SL{fixed_stop_points}_TR{reversion_target_ratio}_MT{max_trades_per_day}"

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """Generate ORB reversion signals with next-candle entry."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()

        # Default HOLD signal
        signal = {
            'signal': 'HOLD',
            'entry_time': timestamp,
            'entry_price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'reason': 'No signal',
            'risk_per_trade': self.risk_per_trade,
            'contracts': 1  # Fixed contract size
        }

        # CRITICAL: Validate NY session timing before any signal generation
        if not self._validate_session_timing(bar_data, allow_extended=False):
            signal['reason'] = f'Outside regular trading session: {timestamp.time()}'
            return signal

        # Check for pending signal from previous bar (realistic entry)
        if current_date in self.pending_signals:
            pending_signal = self.pending_signals[current_date]
            entry_price = bar_data['open']

            # Fixed position size - exactly 1 contract always
            position_size = 1  # No multiple contracts allowed

            # Update signal with actual entry price and position size
            pending_signal['entry_price'] = entry_price
            pending_signal['contracts'] = position_size

            if pending_signal['signal'] == 'BUY':
                pending_signal['stop_loss'] = entry_price - self.fixed_stop
                pending_signal['take_profit'] = entry_price + (self.fixed_stop * self.target_ratio)
            else:  # SELL
                pending_signal['stop_loss'] = entry_price + self.fixed_stop
                pending_signal['take_profit'] = entry_price - (self.fixed_stop * self.target_ratio)

            # Clear pending signal and return it
            del self.pending_signals[current_date]
            return pending_signal

        # Update opening range
        if current_date != self.current_date:
            self.current_date = current_date
            self._reset_daily_data(current_date)

        self._update_opening_range(bar_data)

        # Check if we have a valid OR
        if current_date not in self.daily_ranges:
            return signal

        or_data = self.daily_ranges[current_date]
        if or_data['bars_counted'] == 0:
            return signal

        # Only trade during active hours (after OR period)
        current_time = timestamp.time()
        or_end_hour = 9 + (30 + self.or_minutes) // 60
        or_end_minute = (30 + self.or_minutes) % 60
        or_end_time = time(or_end_hour, or_end_minute)  # OR end time
        close_cutoff = time(15, 45) if self.intraday_only else time(16, 0)

        if not (or_end_time <= current_time <= close_cutoff):
            return signal

        # Check daily trade limit
        if current_date not in self.daily_trade_count:
            self.daily_trade_count[current_date] = 0

        if self.daily_trade_count[current_date] >= self.max_trades_per_day:
            return signal

        # Only trade if no existing positions (one at a time)
        if len(portfolio.positions) > 0:
            return signal

        # Track breakout attempts and generate reversion signals
        self._track_breakout_attempts(bar_data)

        current_price = bar_data['close']
        or_high = or_data['high']
        or_low = or_data['low']
        or_range = or_high - or_low
        or_center = (or_high + or_low) / 2

        breakout_data = self.breakout_tracker[current_date]

        # Long reversion: Price was above OR, now reverting back down
        if (breakout_data['failed_upside_breakout'] and
            current_price < or_high and
            current_price > or_center and
            not or_data.get('long_reversion_triggered', False)):

            # Store pending long reversion signal
            self.pending_signals[current_date] = {
                'signal': 'BUY',
                'entry_time': timestamp,
                'entry_price': 0,  # Will be filled with next bar's open
                'stop_loss': 0,    # Will be calculated with actual entry
                'take_profit': 0,  # Will be calculated with actual entry
                'reason': f'ORB Long reversion after failed breakout above {or_high:.2f}',
                'risk_per_trade': self.risk_per_trade,
                'contracts': 1  # Will be recalculated
            }
            or_data['long_reversion_triggered'] = True
            self.daily_trade_count[current_date] += 1

        # Short reversion: Price was below OR, now reverting back up
        elif (breakout_data['failed_downside_breakout'] and
              current_price > or_low and
              current_price < or_center and
              not or_data.get('short_reversion_triggered', False)):

            # Store pending short reversion signal
            self.pending_signals[current_date] = {
                'signal': 'SELL',
                'entry_time': timestamp,
                'entry_price': 0,  # Will be filled with next bar's open
                'stop_loss': 0,    # Will be calculated with actual entry
                'take_profit': 0,  # Will be calculated with actual entry
                'reason': f'ORB Short reversion after failed breakout below {or_low:.2f}',
                'risk_per_trade': self.risk_per_trade,
                'contracts': 1  # Will be recalculated
            }
            or_data['short_reversion_triggered'] = True
            self.daily_trade_count[current_date] += 1

        return signal

    def _track_breakout_attempts(self, bar_data: Dict[str, Any]) -> None:
        """Track breakout attempts to identify reversion opportunities."""
        current_date = bar_data['timestamp'].date()
        current_price = bar_data['close']

        if current_date not in self.breakout_tracker:
            return

        if current_date not in self.daily_ranges:
            return

        or_data = self.daily_ranges[current_date]
        or_high = or_data['high']
        or_low = or_data['low']
        or_range = or_high - or_low

        breakout_data = self.breakout_tracker[current_date]

        # Track upside breakout attempts
        if current_price > or_high + (or_range * self.breakout_threshold):
            breakout_data['upside_breakout_attempted'] = True
            breakout_data['max_upside_extension'] = max(
                breakout_data['max_upside_extension'],
                current_price - or_high
            )

        # Track downside breakout attempts
        if current_price < or_low - (or_range * self.breakout_threshold):
            breakout_data['downside_breakout_attempted'] = True
            breakout_data['max_downside_extension'] = max(
                breakout_data['max_downside_extension'],
                or_low - current_price
            )

        # Detect failed upside breakout (price came back into OR)
        if (breakout_data['upside_breakout_attempted'] and
            current_price <= or_high):
            breakout_data['failed_upside_breakout'] = True

        # Detect failed downside breakout (price came back into OR)
        if (breakout_data['downside_breakout_attempted'] and
            current_price >= or_low):
            breakout_data['failed_downside_breakout'] = True

    def _reset_daily_data(self, current_date) -> None:
        """Reset daily tracking data."""
        self.daily_ranges[current_date] = {
            'high': float('-inf'),
            'low': float('inf'),
            'bars_counted': 0,
            'long_reversion_triggered': False,
            'short_reversion_triggered': False
        }
        self.daily_trade_count[current_date] = 0
        self.breakout_tracker[current_date] = {
            'upside_breakout_attempted': False,
            'downside_breakout_attempted': False,
            'failed_upside_breakout': False,
            'failed_downside_breakout': False,
            'max_upside_extension': 0,
            'max_downside_extension': 0
        }
        # Clear any pending signals from previous day
        if current_date in self.pending_signals:
            del self.pending_signals[current_date]

    def _update_opening_range(self, bar_data: Dict[str, Any]) -> None:
        """Update opening range for current day."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()
        current_time = timestamp.time()

        # Market opens at 9:30 AM ET
        market_open = time(9, 30)
        or_end_hour = 9 + (30 + self.or_minutes) // 60
        or_end_minute = (30 + self.or_minutes) % 60
        or_end_time = time(or_end_hour, or_end_minute)

        # Only count bars within opening range period
        if market_open <= current_time < or_end_time:
            if current_date in self.daily_ranges:
                or_data = self.daily_ranges[current_date]

                # Initialize OR with first bar
                if or_data['bars_counted'] == 0:
                    or_data['high'] = bar_data['high']
                    or_data['low'] = bar_data['low']
                else:
                    # Expand range with subsequent bars
                    or_data['high'] = max(or_data['high'], bar_data['high'])
                    or_data['low'] = min(or_data['low'], bar_data['low'])

                or_data['bars_counted'] += 1

        # Handle market open detection even if not a new day
        elif current_time == market_open:
            if current_date in self.daily_ranges:
                or_data = self.daily_ranges[current_date]
                or_data['high'] = bar_data['high']
                or_data['low'] = bar_data['low']
                or_data['bars_counted'] = 1