"""
ORB Breakout Strategy - Sound Entry/Exit Logic
==============================================
Momentum-based strategy that enters when price breaks above/below opening range.
Uses next-candle entry with proper risk management and dynamic capital sizing.
"""

from typing import Dict, Any, TYPE_CHECKING
import pandas as pd
from datetime import datetime, time

if TYPE_CHECKING:
    from portfolio import Portfolio

from strategy import Strategy

class ORBBreakoutStrategy(Strategy):
    """
    Opening Range Breakout Strategy with momentum continuation logic.

    Logic:
    1. Calculate opening range for first N minutes of trading day
    2. Wait for price to break above OR high (long) or below OR low (short)
    3. Enter at next candle open (realistic execution)
    4. Use fixed point stops and R:R based targets
    5. Dynamic position sizing based on risk percentage
    """

    def __init__(self, risk_per_trade: float = 0.02, or_minutes: int = 15,
                 fixed_stop_points: float = 80.0, rr_ratio: float = 1.5,
                 max_trades_per_day: int = 3, intraday_only: bool = True,
                 partial_profit_pct: float = 0.5, move_to_breakeven_pct: float = 0.6):
        """
        Initialize ORB Breakout Strategy.

        Args:
            risk_per_trade: Risk as percentage of capital (0.01 = 1%)
            or_minutes: Opening range period in minutes
            fixed_stop_points: Stop loss distance in points
            rr_ratio: Risk/Reward ratio for targets
            max_trades_per_day: Maximum trades per day
            intraday_only: Close all positions before market close
            partial_profit_pct: Percentage of position to close at 50% target (0.5 = 50%)
            move_to_breakeven_pct: Move SL to breakeven when this % of target reached
        """
        super().__init__(risk_per_trade)
        self.or_minutes = or_minutes
        self.fixed_stop = fixed_stop_points
        self.rr_ratio = rr_ratio
        self.max_trades_per_day = max_trades_per_day
        self.intraday_only = intraday_only
        self.partial_profit_pct = partial_profit_pct
        self.move_to_breakeven_pct = move_to_breakeven_pct

        # State tracking
        self.daily_ranges = {}
        self.daily_trade_count = {}
        self.pending_signals = {}  # Store signals for next bar entry
        self.current_date = None

        self.name = f"ORB_Breakout_OR{or_minutes}_SL{fixed_stop_points}_RR{rr_ratio}_MT{max_trades_per_day}"

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """Generate ORB breakout signals with next-candle entry."""
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
                pending_signal['take_profit'] = entry_price + (self.fixed_stop * self.rr_ratio)
            else:  # SELL
                pending_signal['stop_loss'] = entry_price + self.fixed_stop
                pending_signal['take_profit'] = entry_price - (self.fixed_stop * self.rr_ratio)

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

        # Generate breakout signals
        current_price = bar_data['close']
        or_high = or_data['high']
        or_low = or_data['low']

        # Long breakout above OR high
        if (current_price > or_high and
            not or_data.get('long_breakout_triggered', False)):

            # Enhanced reason with session timing info
            is_prime_time = self._is_orb_prime_time(timestamp)
            session_info = "ORB Prime Time" if is_prime_time else "Regular Session"

            # Store pending long signal for next bar
            self.pending_signals[current_date] = {
                'signal': 'BUY',
                'entry_time': timestamp,
                'entry_price': 0,  # Will be filled with next bar's open
                'stop_loss': 0,    # Will be calculated with actual entry
                'take_profit': 0,  # Will be calculated with actual entry
                'reason': f'ORB Long breakout above {or_high:.2f} ({session_info})',
                'risk_per_trade': self.risk_per_trade,
                'contracts': 1  # Will be recalculated
            }
            or_data['long_breakout_triggered'] = True
            self.daily_trade_count[current_date] += 1

        # Short breakdown below OR low
        elif (current_price < or_low and
              not or_data.get('short_breakout_triggered', False)):

            # Enhanced reason with session timing info
            is_prime_time = self._is_orb_prime_time(timestamp)
            session_info = "ORB Prime Time" if is_prime_time else "Regular Session"

            # Store pending short signal for next bar
            self.pending_signals[current_date] = {
                'signal': 'SELL',
                'entry_time': timestamp,
                'entry_price': 0,  # Will be filled with next bar's open
                'stop_loss': 0,    # Will be calculated with actual entry
                'take_profit': 0,  # Will be calculated with actual entry
                'reason': f'ORB Short breakdown below {or_low:.2f} ({session_info})',
                'risk_per_trade': self.risk_per_trade,
                'contracts': 1  # Will be recalculated
            }
            or_data['short_breakout_triggered'] = True
            self.daily_trade_count[current_date] += 1

        return signal

    def _reset_daily_data(self, current_date) -> None:
        """Reset daily tracking data."""
        self.daily_ranges[current_date] = {
            'high': float('-inf'),
            'low': float('inf'),
            'bars_counted': 0,
            'long_breakout_triggered': False,
            'short_breakout_triggered': False
        }
        self.daily_trade_count[current_date] = 0
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