"""
High-Frequency NQ Strategies for 30+ Points/Day Target
=====================================================
Multiple trade opportunities per day to achieve aggressive point targets.
"""

from typing import Dict, Any, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from portfolio import Portfolio

from strategy import Strategy

class ScalpingORBStrategy(Strategy):
    """
    Aggressive ORB scalping with multiple entries and tight profit targets.
    """

    def __init__(self, risk_per_trade: float = 0.02, or_minutes: int = 15,
                 scalp_target_points: float = 20.0, stop_loss_points: float = 30.0,
                 max_trades_per_day: int = 15, contracts_per_trade: int = 1):
        """
        Initialize Scalping ORB Strategy.

        Args:
            scalp_target_points: Quick profit target in points
            stop_loss_points: Stop loss in points
            max_trades_per_day: Max trades allowed per day
        """
        super().__init__(risk_per_trade)
        self.or_minutes = or_minutes
        self.scalp_target = scalp_target_points
        self.stop_loss_points = stop_loss_points
        self.max_trades_per_day = max_trades_per_day
        self.contracts_per_trade = contracts_per_trade
        self.daily_ranges = {}
        self.daily_trade_count = {}
        self.daily_direction_bias = {}
        self.current_date = None
        self.name = f"ScalpORB_T{scalp_target_points}_SL{stop_loss_points}_MT{max_trades_per_day}"

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """Generate scalping signals with multiple opportunities."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()

        signal = {
            'signal': 'HOLD',
            'entry_time': timestamp,
            'stop_loss': 0,
            'take_profit': 0,
            'reason': 'No signal'
        }

        # Update opening range
        if current_date != self.current_date:
            self.current_date = current_date
            self._update_opening_range(bar_data)
        else:
            market_open = pd.to_datetime("09:30", format="%H:%M").time()
            if timestamp.time() == market_open:
                self._update_opening_range(bar_data)

        if current_date not in self.daily_ranges:
            return signal

        # Trading hours - longer window for more opportunities
        current_time = timestamp.time()
        trade_start = pd.to_datetime("09:45", format="%H:%M").time()
        close_cutoff = pd.to_datetime("15:45", format="%H:%M").time()

        if not (trade_start <= current_time <= close_cutoff):
            return signal

        # Check trade limits
        if current_date not in self.daily_trade_count:
            self.daily_trade_count[current_date] = 0

        if self.daily_trade_count[current_date] >= self.max_trades_per_day:
            return signal

        # One position at a time
        if hasattr(portfolio, 'positions') and len(portfolio.positions) > 0:
            return signal

        or_data = self.daily_ranges[current_date]
        if or_data['bars_counted'] == 0:
            return signal

        current_price = bar_data['close']
        or_high = or_data['high']
        or_low = or_data['low']
        or_mid = (or_high + or_low) / 2

        # Establish daily bias based on first breakout
        if current_date not in self.daily_direction_bias:
            if current_price > or_high:
                self.daily_direction_bias[current_date] = 'BULLISH'
            elif current_price < or_low:
                self.daily_direction_bias[current_date] = 'BEARISH'
            else:
                self.daily_direction_bias[current_date] = 'NEUTRAL'

        daily_bias = self.daily_direction_bias[current_date]

        # Scalping opportunities based on bias and OR levels
        if daily_bias == 'BULLISH' or daily_bias == 'NEUTRAL':
            # Look for long entries on pullbacks to OR levels
            if (or_low <= current_price <= or_mid and
                bar_data['low'] <= or_low and
                current_price > or_low):

                signal = {
                    'signal': 'BUY',
                    'entry_time': timestamp,
                    'entry_price': current_price,
                    'stop_loss': current_price - self.stop_loss_points,
                    'take_profit': current_price + self.scalp_target,
                    'risk_per_trade': self.risk_per_trade,
                    'contracts': self.contracts_per_trade,
                    'reason': f'Scalp Long from OR support {or_low:.2f}'
                }
                self.daily_trade_count[current_date] += 1

        if daily_bias == 'BEARISH' or daily_bias == 'NEUTRAL':
            # Look for short entries on bounces to OR levels
            if (or_mid <= current_price <= or_high and
                bar_data['high'] >= or_high and
                current_price < or_high):

                signal = {
                    'signal': 'SELL',
                    'entry_time': timestamp,
                    'entry_price': current_price,
                    'stop_loss': current_price + self.stop_loss_points,
                    'take_profit': current_price - self.scalp_target,
                    'risk_per_trade': self.risk_per_trade,
                    'contracts': self.contracts_per_trade,
                    'reason': f'Scalp Short from OR resistance {or_high:.2f}'
                }
                self.daily_trade_count[current_date] += 1

        return signal

    def _update_opening_range(self, bar_data: Dict[str, Any]) -> None:
        """Update opening range data."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()
        current_time = timestamp.time()

        if current_date not in self.daily_ranges:
            self.daily_ranges[current_date] = {
                'high': float('-inf'),
                'low': float('inf'),
                'bars_counted': 0
            }

        market_open = pd.to_datetime("09:30", format="%H:%M").time()
        if current_time == market_open:
            or_data = self.daily_ranges[current_date]
            or_data['high'] = bar_data['high']
            or_data['low'] = bar_data['low']
            or_data['bars_counted'] = 1


class MomentumBreakoutStrategy(Strategy):
    """
    High-frequency momentum breakouts with multiple timeframe confirmation.
    """

    def __init__(self, risk_per_trade: float = 0.02, momentum_threshold: float = 25.0,
                 target_points: float = 35.0, stop_points: float = 25.0,
                 max_trades_per_day: int = 12, contracts_per_trade: int = 1):
        """
        Initialize Momentum Breakout Strategy.

        Args:
            momentum_threshold: Points move required for momentum signal
            target_points: Profit target in points
            stop_points: Stop loss in points
        """
        super().__init__(risk_per_trade)
        self.momentum_threshold = momentum_threshold
        self.target_points = target_points
        self.stop_points = stop_points
        self.max_trades_per_day = max_trades_per_day
        self.contracts_per_trade = contracts_per_trade
        self.daily_trade_count = {}
        self.price_history = []
        self.current_date = None
        self.name = f"Momentum_T{target_points}_SL{stop_points}_MT{max_trades_per_day}"

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """Generate momentum breakout signals."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()

        signal = {
            'signal': 'HOLD',
            'entry_time': timestamp,
            'stop_loss': 0,
            'take_profit': 0,
            'reason': 'No signal'
        }

        # Reset daily data
        if current_date != self.current_date:
            self.current_date = current_date
            self.price_history = []

        # Trading hours
        current_time = timestamp.time()
        trade_start = pd.to_datetime("09:30", format="%H:%M").time()
        close_cutoff = pd.to_datetime("15:30", format="%H:%M").time()

        if not (trade_start <= current_time <= close_cutoff):
            return signal

        # Check trade limits
        if current_date not in self.daily_trade_count:
            self.daily_trade_count[current_date] = 0

        if self.daily_trade_count[current_date] >= self.max_trades_per_day:
            return signal

        if hasattr(portfolio, 'positions') and len(portfolio.positions) > 0:
            return signal

        # Track price history for momentum calculation
        current_price = bar_data['close']
        self.price_history.append({
            'timestamp': timestamp,
            'price': current_price,
            'high': bar_data['high'],
            'low': bar_data['low']
        })

        # Keep only last 8 bars (2 hours of 15min data)
        if len(self.price_history) > 8:
            self.price_history = self.price_history[-8:]

        # Need at least 4 bars for momentum calculation
        if len(self.price_history) < 4:
            return signal

        # Calculate momentum over last hour (4 bars)
        recent_high = max(p['high'] for p in self.price_history[-4:])
        recent_low = min(p['low'] for p in self.price_history[-4:])
        momentum_range = recent_high - recent_low

        # Look for momentum breakouts
        if momentum_range >= self.momentum_threshold:
            prev_bar = self.price_history[-2]

            # Long momentum: price breaks above recent high
            if (current_price > recent_high and
                prev_bar['price'] <= recent_high):

                signal = {
                    'signal': 'BUY',
                    'entry_time': timestamp,
                    'entry_price': current_price,
                    'stop_loss': current_price - self.stop_points,
                    'take_profit': current_price + self.target_points,
                    'risk_per_trade': self.risk_per_trade,
                    'contracts': self.contracts_per_trade,
                    'reason': f'Momentum Long breakout above {recent_high:.2f}'
                }
                self.daily_trade_count[current_date] += 1

            # Short momentum: price breaks below recent low
            elif (current_price < recent_low and
                  prev_bar['price'] >= recent_low):

                signal = {
                    'signal': 'SELL',
                    'entry_time': timestamp,
                    'entry_price': current_price,
                    'stop_loss': current_price + self.stop_points,
                    'take_profit': current_price - self.target_points,
                    'risk_per_trade': self.risk_per_trade,
                    'contracts': self.contracts_per_trade,
                    'reason': f'Momentum Short breakdown below {recent_low:.2f}'
                }
                self.daily_trade_count[current_date] += 1

        return signal


class MultiSessionORBStrategy(Strategy):
    """
    Multiple opening range sessions throughout the day (9:30, 10:30, 1:30).
    """

    def __init__(self, risk_per_trade: float = 0.02, or_duration_minutes: int = 15,
                 fixed_stop_points: float = 40.0, target_multiplier: float = 1.5,
                 max_trades_per_session: int = 2, contracts_per_trade: int = 1):
        """Initialize Multi-Session ORB Strategy."""
        super().__init__(risk_per_trade)
        self.or_duration = or_duration_minutes
        self.fixed_stop = fixed_stop_points
        self.target_mult = target_multiplier
        self.max_per_session = max_trades_per_session
        self.contracts_per_trade = contracts_per_trade

        # Define session times
        self.sessions = [
            ('morning', pd.to_datetime("09:30", format="%H:%M").time(), pd.to_datetime("12:00", format="%H:%M").time()),
            ('midday', pd.to_datetime("12:00", format="%H:%M").time(), pd.to_datetime("15:00", format="%H:%M").time()),
            ('afternoon', pd.to_datetime("15:00", format="%H:%M").time(), pd.to_datetime("16:00", format="%H:%M").time())
        ]

        self.session_ranges = {}
        self.session_trade_counts = {}
        self.current_date = None
        self.name = f"MultiSession_SL{fixed_stop_points}_TM{target_multiplier}_MS{max_trades_per_session}"

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """Generate signals from multiple OR sessions."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()

        signal = {
            'signal': 'HOLD',
            'entry_time': timestamp,
            'stop_loss': 0,
            'take_profit': 0,
            'reason': 'No signal'
        }

        # Reset daily data
        if current_date != self.current_date:
            self.current_date = current_date
            self.session_ranges[current_date] = {}
            self.session_trade_counts[current_date] = {}
            for session_name, _, _ in self.sessions:
                self.session_ranges[current_date][session_name] = {
                    'high': float('-inf'), 'low': float('inf'), 'established': False, 'traded': 0
                }
                self.session_trade_counts[current_date][session_name] = 0

        current_time = timestamp.time()
        current_session = self._get_current_session(current_time)

        if not current_session:
            return signal

        if hasattr(portfolio, 'positions') and len(portfolio.positions) > 0:
            return signal

        session_name = current_session[0]
        session_data = self.session_ranges[current_date][session_name]

        # Establish OR for current session (first 15 minutes)
        session_start = current_session[1]
        or_end_minutes = 15
        or_end_time = pd.to_datetime(f"{session_start.hour}:{session_start.minute + or_end_minutes}", format="%H:%M").time()

        if session_start <= current_time <= or_end_time:
            if not session_data['established']:
                session_data['high'] = bar_data['high']
                session_data['low'] = bar_data['low']
                session_data['established'] = True
            else:
                session_data['high'] = max(session_data['high'], bar_data['high'])
                session_data['low'] = min(session_data['low'], bar_data['low'])
            return signal

        # Trade the session OR breakouts
        if session_data['established'] and current_time > or_end_time:
            if self.session_trade_counts[current_date][session_name] >= self.max_per_session:
                return signal

            current_price = bar_data['close']
            or_high = session_data['high']
            or_low = session_data['low']

            # Long breakout
            if current_price > or_high and session_data['traded'] < self.max_per_session:
                signal = {
                    'signal': 'BUY',
                    'entry_time': timestamp,
                    'entry_price': current_price,
                    'stop_loss': current_price - self.fixed_stop,
                    'take_profit': current_price + (self.fixed_stop * self.target_mult),
                    'risk_per_trade': self.risk_per_trade,
                    'contracts': self.contracts_per_trade,
                    'reason': f'{session_name.title()} session long breakout'
                }
                self.session_trade_counts[current_date][session_name] += 1
                session_data['traded'] += 1

            # Short breakdown
            elif current_price < or_low and session_data['traded'] < self.max_per_session:
                signal = {
                    'signal': 'SELL',
                    'entry_time': timestamp,
                    'entry_price': current_price,
                    'stop_loss': current_price + self.fixed_stop,
                    'take_profit': current_price - (self.fixed_stop * self.target_mult),
                    'risk_per_trade': self.risk_per_trade,
                    'contracts': self.contracts_per_trade,
                    'reason': f'{session_name.title()} session short breakdown'
                }
                self.session_trade_counts[current_date][session_name] += 1
                session_data['traded'] += 1

        return signal

    def _get_current_session(self, current_time):
        """Determine which trading session we're in."""
        for session in self.sessions:
            session_name, start_time, end_time = session
            if start_time <= current_time <= end_time:
                return session
        return None