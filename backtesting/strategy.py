"""
Strategy Module
===============
Abstract base class for trading strategies with mandatory risk management.
All strategies must implement the required interface for signal generation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING
from datetime import time
import pandas as pd

if TYPE_CHECKING:
    from portfolio import Portfolio

class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    All strategies must implement generate_signal() method and include
    mandatory risk management parameters (stop_loss, take_profit).
    """

    def __init__(self, risk_per_trade: float):
        """
        Initialize strategy with risk management parameter.

        Args:
            risk_per_trade: Risk per trade as decimal (e.g., 0.02 for 2%)
        """
        if not 0 < risk_per_trade <= 0.1:  # Max 10% risk
            raise ValueError("risk_per_trade must be between 0 and 0.1 (0% and 10%)")

        self.risk_per_trade = risk_per_trade
        self.name = self.__class__.__name__

    @abstractmethod
    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """
        Generate trading signal based on current bar and portfolio state.

        Args:
            bar_data: Current bar data dictionary
            portfolio: Portfolio object with current state

        Returns:
            Signal dictionary with required keys:
            {
                'signal': 'BUY' | 'SELL' | 'HOLD',
                'entry_time': timestamp,
                'stop_loss': float,
                'take_profit': float,
                'reason': str (optional)
            }
        """
        pass

    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Validate signal dictionary has required fields.

        Args:
            signal: Signal dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        required_keys = ['signal', 'entry_time', 'stop_loss', 'take_profit']

        if not all(key in signal for key in required_keys):
            return False

        if signal['signal'] not in ['BUY', 'SELL', 'HOLD']:
            return False

        if signal['signal'] != 'HOLD':
            if signal['stop_loss'] <= 0 or signal['take_profit'] <= 0:
                return False

        return True

    def _is_regular_trading_session(self, timestamp) -> bool:
        """
        Check if timestamp is within regular NY trading session.

        Args:
            timestamp: datetime object to check

        Returns:
            True if within 9:30 AM - 3:45 PM ET, False otherwise
        """
        current_time = timestamp.time()
        session_start = time(9, 30)  # 9:30 AM ET
        session_end = time(15, 45)   # 3:45 PM ET (15 min buffer before close)

        return session_start <= current_time <= session_end

    def _is_orb_prime_time(self, timestamp) -> bool:
        """
        Check if timestamp is within optimal ORB trading hours.

        Args:
            timestamp: datetime object to check

        Returns:
            True if within 9:30 AM - 11:30 AM ET (ORB prime time), False otherwise
        """
        current_time = timestamp.time()
        orb_start = time(9, 30)   # 9:30 AM ET
        orb_end = time(11, 30)    # 11:30 AM ET

        return orb_start <= current_time <= orb_end

    def _validate_session_timing(self, bar_data: Dict[str, Any], allow_extended: bool = False) -> bool:
        """
        Validate that bar_data timestamp is within appropriate trading session.

        Args:
            bar_data: Current bar data dictionary
            allow_extended: If True, allows extended hours (9:15-16:15), default False

        Returns:
            True if timing is valid for signal generation, False otherwise
        """
        timestamp = bar_data.get('timestamp')
        if not timestamp:
            return False

        if allow_extended:
            # Extended hours: 9:15 AM - 4:15 PM ET (for data validation)
            current_time = timestamp.time()
            return time(9, 15) <= current_time <= time(16, 15)
        else:
            # Regular session: 9:30 AM - 3:45 PM ET (for signal generation)
            return self._is_regular_trading_session(timestamp)


class ORBStrategy(Strategy):
    """
    Opening Range Breakout Strategy - Example implementation.

    Trades breakouts from the first 15 minutes of NY session.
    """

    def __init__(self, risk_per_trade: float = 0.02, or_minutes: int = 15, max_trades_per_day: int = 999, intraday_only: bool = True):
        """
        Initialize ORB Strategy.

        Args:
            risk_per_trade: Risk per trade as decimal
            or_minutes: Opening range minutes (default 15)
            max_trades_per_day: Maximum trades per day (default unlimited)
            intraday_only: Only allow intraday trades (close at market close)
        """
        super().__init__(risk_per_trade)
        self.or_minutes = or_minutes
        self.max_trades_per_day = max_trades_per_day
        self.intraday_only = intraday_only
        self.daily_ranges = {}
        self.daily_trade_count = {}
        self.current_date = None

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """
        Generate ORB signals based on opening range breakout logic.
        """
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()

        # Default HOLD signal
        signal = {
            'signal': 'HOLD',
            'entry_time': timestamp,
            'stop_loss': 0,
            'take_profit': 0,
            'reason': 'No signal'
        }

        # Update opening range if new day
        if current_date != self.current_date:
            self.current_date = current_date
            self._update_opening_range(bar_data)
        else:
            # Also check for 9:30 AM bar even if not new day
            market_open = pd.to_datetime("09:30", format="%H:%M").time()
            if timestamp.time() == market_open:
                self._update_opening_range(bar_data)

        # Check if we have opening range for today
        if current_date not in self.daily_ranges:
            return signal

        or_data = self.daily_ranges[current_date]

        # Only trade during active hours (after OR period)
        current_time = timestamp.time()
        market_open = pd.to_datetime("09:30", format="%H:%M").time()
        market_close = pd.to_datetime("16:00", format="%H:%M").time()

        # For 15-min bars, we can trade starting from 9:45 (second bar after market open)
        trade_start = pd.to_datetime("09:45", format="%H:%M").time()

        # For intraday only, don't allow new trades in last 15 minutes
        if self.intraday_only:
            close_cutoff = pd.to_datetime("15:45", format="%H:%M").time()
            if current_time > close_cutoff:
                return signal

        # Check if we're in valid trading hours
        if not (trade_start <= current_time <= market_close):
            return signal

        # Check daily trade limit
        if current_date not in self.daily_trade_count:
            self.daily_trade_count[current_date] = 0

        if self.daily_trade_count[current_date] >= self.max_trades_per_day:
            return signal

        # Check for breakout
        current_price = bar_data['close']
        or_high = or_data['high']
        or_low = or_data['low']

        # Check if position is already open to enforce one at a time
        if hasattr(portfolio, 'positions') and len(portfolio.positions) > 0:
            return signal

        # Long breakout above OR high
        if current_price > or_high and not or_data.get('breakout_triggered', False):
            stop_distance = or_high - or_low
            signal = {
                'signal': 'BUY',
                'entry_time': timestamp,
                'entry_price': current_price,
                'stop_loss': or_low,
                'take_profit': current_price + (stop_distance * 1.5),  # 1.5:1 R/R
                'risk_per_trade': self.risk_per_trade,
                'reason': f'ORB Long breakout above {or_high:.2f}'
            }
            or_data['breakout_triggered'] = True
            self.daily_trade_count[current_date] += 1

        # Short breakdown below OR low
        elif current_price < or_low and not or_data.get('breakdown_triggered', False):
            stop_distance = or_high - or_low
            signal = {
                'signal': 'SELL',
                'entry_time': timestamp,
                'entry_price': current_price,
                'stop_loss': or_high,
                'take_profit': current_price - (stop_distance * 1.5),  # 1.5:1 R/R
                'risk_per_trade': self.risk_per_trade,
                'reason': f'ORB Short breakdown below {or_low:.2f}'
            }
            or_data['breakdown_triggered'] = True
            self.daily_trade_count[current_date] += 1

        return signal

    def _update_opening_range(self, bar_data: Dict[str, Any]) -> None:
        """Update opening range data for current day."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()
        current_time = timestamp.time()

        # Initialize daily range if new day
        if current_date not in self.daily_ranges:
            self.daily_ranges[current_date] = {
                'high': float('-inf'),
                'low': float('inf'),
                'bars_counted': 0,
                'breakout_triggered': False,
                'breakdown_triggered': False
            }

        # For 15-minute bars, the opening range is just the 9:30 AM bar
        market_open = pd.to_datetime("09:30", format="%H:%M").time()

        # Check if this is the market open bar (9:30 AM)
        if current_time == market_open:
            or_data = self.daily_ranges[current_date]
            or_data['high'] = bar_data['high']
            or_data['low'] = bar_data['low']
            or_data['bars_counted'] = 1


class MovingAverageCrossStrategy(Strategy):
    """
    Simple Moving Average Cross Strategy - Example implementation.
    """

    def __init__(self, risk_per_trade: float = 0.02, fast_period: int = 10, slow_period: int = 20):
        """
        Initialize MA Cross Strategy.

        Args:
            risk_per_trade: Risk per trade as decimal
            fast_period: Fast MA period
            slow_period: Slow MA period
        """
        super().__init__(risk_per_trade)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """
        Generate MA cross signals.

        Note: This is a simplified example. In practice, you'd want to
        use the DataHandler's get_historical_data() method.
        """
        # Default HOLD signal
        signal = {
            'signal': 'HOLD',
            'entry_time': bar_data['timestamp'],
            'stop_loss': 0,
            'take_profit': 0,
            'reason': 'Insufficient data for MA calculation'
        }

        # This is where you'd implement MA cross logic
        # For brevity, returning HOLD signal
        return signal


class BuyAndHoldStrategy(Strategy):
    """
    Simple Buy and Hold Strategy - Example implementation.
    """

    def __init__(self, risk_per_trade: float = 0.02):
        super().__init__(risk_per_trade)
        self.position_opened = False

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """Generate buy and hold signal (buy once and hold)."""

        if not self.position_opened and len(portfolio.closed_trades) == 0:
            # Open initial position
            current_price = bar_data['close']
            self.position_opened = True

            return {
                'signal': 'BUY',
                'entry_time': bar_data['timestamp'],
                'stop_loss': current_price * 0.9,  # 10% stop loss
                'take_profit': current_price * 2.0,  # 100% take profit
                'reason': 'Initial buy and hold position'
            }

        return {
            'signal': 'HOLD',
            'entry_time': bar_data['timestamp'],
            'stop_loss': 0,
            'take_profit': 0,
            'reason': 'Holding position'
        }