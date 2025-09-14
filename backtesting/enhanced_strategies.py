"""
Enhanced Trading Strategies for 30+ Points/Day Optimization
============================================================
Multiple advanced strategies to achieve target performance.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from portfolio import Portfolio

from strategy import Strategy

class OptimizedORBStrategy(Strategy):
    """
    Enhanced ORB Strategy with fixed point-based stops and optimized parameters.
    """

    def __init__(self, risk_per_trade: float = 0.02, or_minutes: int = 15,
                 fixed_stop_points: float = None, target_multiplier: float = 1.5,
                 max_trades_per_day: int = 5, intraday_only: bool = True,
                 contracts_per_trade: int = 1):
        """
        Initialize Optimized ORB Strategy.

        Args:
            risk_per_trade: Risk per trade as decimal (ignored if fixed contracts used)
            or_minutes: Opening range minutes
            fixed_stop_points: Fixed stop loss in points (overrides OR-based stops)
            target_multiplier: Risk/reward ratio
            max_trades_per_day: Maximum trades per day
            intraday_only: Force intraday only
            contracts_per_trade: Fixed contract size (0 = use risk-based sizing)
        """
        super().__init__(risk_per_trade)
        self.or_minutes = or_minutes
        self.fixed_stop_points = fixed_stop_points
        self.target_multiplier = target_multiplier
        self.max_trades_per_day = max_trades_per_day
        self.intraday_only = intraday_only
        self.contracts_per_trade = contracts_per_trade
        self.daily_ranges = {}
        self.daily_trade_count = {}
        self.current_date = None
        self.pending_signals = {}  # Store signals for next bar entry
        self.name = f"OptimizedORB_SL{fixed_stop_points}_TM{target_multiplier}_C{contracts_per_trade}"

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """Generate optimized ORB signals."""
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

        # Check for pending signal from previous bar
        if current_date in self.pending_signals:
            pending_signal = self.pending_signals[current_date]
            # Execute with current bar's open price (next bar entry)
            entry_price = bar_data['open']

            # Update signal with correct entry price
            pending_signal['entry_price'] = entry_price
            if pending_signal['signal'] == 'BUY':
                if self.fixed_stop_points:
                    pending_signal['stop_loss'] = entry_price - self.fixed_stop_points
                    pending_signal['take_profit'] = entry_price + (self.fixed_stop_points * self.target_multiplier)
                else:
                    stop_distance = pending_signal['stop_distance']
                    pending_signal['stop_loss'] = entry_price - stop_distance
                    pending_signal['take_profit'] = entry_price + (stop_distance * self.target_multiplier)
            else:  # SELL
                if self.fixed_stop_points:
                    pending_signal['stop_loss'] = entry_price + self.fixed_stop_points
                    pending_signal['take_profit'] = entry_price - (self.fixed_stop_points * self.target_multiplier)
                else:
                    stop_distance = pending_signal['stop_distance']
                    pending_signal['stop_loss'] = entry_price + stop_distance
                    pending_signal['take_profit'] = entry_price - (stop_distance * self.target_multiplier)

            # Clear pending signal and return it
            del self.pending_signals[current_date]
            return pending_signal

        # Update opening range
        if current_date != self.current_date:
            self.current_date = current_date
            self._update_opening_range(bar_data)
        else:
            market_open = pd.to_datetime("09:30", format="%H:%M").time()
            if timestamp.time() == market_open:
                self._update_opening_range(bar_data)

        # Check if we have opening range for today
        if current_date not in self.daily_ranges:
            return signal

        or_data = self.daily_ranges[current_date]

        # Only trade during active hours
        current_time = timestamp.time()
        trade_start = pd.to_datetime("09:45", format="%H:%M").time()
        market_close = pd.to_datetime("16:00", format="%H:%M").time()

        # For intraday only, don't allow new trades in last 15 minutes
        if self.intraday_only:
            close_cutoff = pd.to_datetime("15:45", format="%H:%M").time()
            if current_time > close_cutoff:
                return signal

        if not (trade_start <= current_time <= market_close):
            return signal

        # Check daily trade limit
        if current_date not in self.daily_trade_count:
            self.daily_trade_count[current_date] = 0

        if self.daily_trade_count[current_date] >= self.max_trades_per_day:
            return signal

        # Check if position is already open (one at a time)
        if hasattr(portfolio, 'positions') and len(portfolio.positions) > 0:
            return signal

        # Check for valid opening range
        if or_data['bars_counted'] == 0:
            return signal

        current_price = bar_data['close']
        or_high = or_data['high']
        or_low = or_data['low']

        # Calculate stop loss and take profit
        if self.fixed_stop_points:
            # Use fixed point stops
            long_stop = current_price - self.fixed_stop_points
            short_stop = current_price + self.fixed_stop_points
            long_target = current_price + (self.fixed_stop_points * self.target_multiplier)
            short_target = current_price - (self.fixed_stop_points * self.target_multiplier)
        else:
            # Use OR-based stops
            stop_distance = or_high - or_low
            long_stop = or_low
            short_stop = or_high
            long_target = current_price + (stop_distance * self.target_multiplier)
            short_target = current_price - (stop_distance * self.target_multiplier)

        # Detect breakouts but store as pending signals (enter next bar)
        if current_price > or_high and not or_data.get('breakout_triggered', False):
            # Store pending long signal for next bar
            if self.fixed_stop_points:
                stop_distance = self.fixed_stop_points
            else:
                stop_distance = current_price - or_low

            self.pending_signals[current_date] = {
                'signal': 'BUY',
                'entry_time': timestamp,
                'entry_price': 0,  # Will be filled with next bar's open
                'stop_loss': 0,    # Will be calculated with actual entry price
                'take_profit': 0,  # Will be calculated with actual entry price
                'risk_per_trade': self.risk_per_trade,
                'contracts': self.contracts_per_trade,
                'reason': f'ORB Long breakout above {or_high:.2f}',
                'stop_distance': stop_distance
            }
            or_data['breakout_triggered'] = True
            self.daily_trade_count[current_date] += 1

        # Short breakdown below OR low
        elif current_price < or_low and not or_data.get('breakdown_triggered', False):
            # Store pending short signal for next bar
            if self.fixed_stop_points:
                stop_distance = self.fixed_stop_points
            else:
                stop_distance = or_high - current_price

            self.pending_signals[current_date] = {
                'signal': 'SELL',
                'entry_time': timestamp,
                'entry_price': 0,  # Will be filled with next bar's open
                'stop_loss': 0,    # Will be calculated with actual entry price
                'take_profit': 0,  # Will be calculated with actual entry price
                'risk_per_trade': self.risk_per_trade,
                'contracts': self.contracts_per_trade,
                'reason': f'ORB Short breakdown below {or_low:.2f}',
                'stop_distance': stop_distance
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

        if current_time == market_open:
            or_data = self.daily_ranges[current_date]
            or_data['high'] = bar_data['high']
            or_data['low'] = bar_data['low']
            or_data['bars_counted'] = 1


class ORBReversalStrategy(Strategy):
    """
    ORB Reversal Strategy - trades against failed breakouts.
    """

    def __init__(self, risk_per_trade: float = 0.02, or_minutes: int = 15,
                 reversal_threshold_points: float = 10.0, target_multiplier: float = 2.0,
                 max_trades_per_day: int = 4, intraday_only: bool = True,
                 contracts_per_trade: int = 1):
        """
        Initialize ORB Reversal Strategy.

        Args:
            reversal_threshold_points: How far past OR level before considering reversal
            target_multiplier: Risk/reward ratio for reversal trades
        """
        super().__init__(risk_per_trade)
        self.or_minutes = or_minutes
        self.reversal_threshold = reversal_threshold_points
        self.target_multiplier = target_multiplier
        self.max_trades_per_day = max_trades_per_day
        self.intraday_only = intraday_only
        self.contracts_per_trade = contracts_per_trade
        self.daily_ranges = {}
        self.daily_trade_count = {}
        self.current_date = None
        self.name = f"ORBReversal_RT{reversal_threshold_points}_TM{target_multiplier}_C{contracts_per_trade}"

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """Generate ORB reversal signals."""
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

        or_data = self.daily_ranges[current_date]

        # Trading hours check
        current_time = timestamp.time()
        trade_start = pd.to_datetime("10:00", format="%H:%M").time()  # Start later for reversals
        market_close = pd.to_datetime("16:00", format="%H:%M").time()

        if self.intraday_only:
            close_cutoff = pd.to_datetime("15:30", format="%H:%M").time()
            if current_time > close_cutoff:
                return signal

        if not (trade_start <= current_time <= market_close):
            return signal

        # Check daily trade limit and position limit
        if current_date not in self.daily_trade_count:
            self.daily_trade_count[current_date] = 0

        if self.daily_trade_count[current_date] >= self.max_trades_per_day:
            return signal

        if hasattr(portfolio, 'positions') and len(portfolio.positions) > 0:
            return signal

        if or_data['bars_counted'] == 0:
            return signal

        current_price = bar_data['close']
        or_high = or_data['high']
        or_low = or_data['low']

        # Look for failed breakouts (reversal opportunities)
        # Long reversal: price broke below OR low but is now reversing up
        if (current_price < or_low - self.reversal_threshold and
            bar_data['high'] > or_low and
            not or_data.get('long_reversal_triggered', False)):

            stop_distance = self.reversal_threshold + 5  # Buffer
            signal = {
                'signal': 'BUY',
                'entry_time': timestamp,
                'entry_price': current_price,
                'stop_loss': current_price - stop_distance,
                'take_profit': current_price + (stop_distance * self.target_multiplier),
                'risk_per_trade': self.risk_per_trade,
                'contracts': self.contracts_per_trade,
                'reason': f'ORB Reversal Long from {current_price:.2f}'
            }
            or_data['long_reversal_triggered'] = True
            self.daily_trade_count[current_date] += 1

        # Short reversal: price broke above OR high but is now reversing down
        elif (current_price > or_high + self.reversal_threshold and
              bar_data['low'] < or_high and
              not or_data.get('short_reversal_triggered', False)):

            stop_distance = self.reversal_threshold + 5
            signal = {
                'signal': 'SELL',
                'entry_time': timestamp,
                'entry_price': current_price,
                'stop_loss': current_price + stop_distance,
                'take_profit': current_price - (stop_distance * self.target_multiplier),
                'risk_per_trade': self.risk_per_trade,
                'contracts': self.contracts_per_trade,
                'reason': f'ORB Reversal Short from {current_price:.2f}'
            }
            or_data['short_reversal_triggered'] = True
            self.daily_trade_count[current_date] += 1

        return signal

    def _update_opening_range(self, bar_data: Dict[str, Any]) -> None:
        """Update opening range data for current day."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()
        current_time = timestamp.time()

        if current_date not in self.daily_ranges:
            self.daily_ranges[current_date] = {
                'high': float('-inf'),
                'low': float('inf'),
                'bars_counted': 0,
                'long_reversal_triggered': False,
                'short_reversal_triggered': False
            }

        market_open = pd.to_datetime("09:30", format="%H:%M").time()
        if current_time == market_open:
            or_data = self.daily_ranges[current_date]
            or_data['high'] = bar_data['high']
            or_data['low'] = bar_data['low']
            or_data['bars_counted'] = 1


class MultiTimeframeORBStrategy(Strategy):
    """
    Multi-timeframe ORB strategy using both 15min and 30min ranges.
    """

    def __init__(self, risk_per_trade: float = 0.02, or_minutes_1: int = 15,
                 or_minutes_2: int = 30, fixed_stop_points: float = 15.0,
                 target_multiplier: float = 2.0, max_trades_per_day: int = 6,
                 intraday_only: bool = True, contracts_per_trade: int = 1):
        """Initialize Multi-timeframe ORB Strategy."""
        super().__init__(risk_per_trade)
        self.or_minutes_1 = or_minutes_1
        self.or_minutes_2 = or_minutes_2
        self.fixed_stop_points = fixed_stop_points
        self.target_multiplier = target_multiplier
        self.max_trades_per_day = max_trades_per_day
        self.intraday_only = intraday_only
        self.contracts_per_trade = contracts_per_trade
        self.daily_ranges = {}
        self.daily_trade_count = {}
        self.current_date = None
        self.name = f"MultiTF_ORB_{or_minutes_1}_{or_minutes_2}_SL{fixed_stop_points}_C{contracts_per_trade}"

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """Generate multi-timeframe ORB signals."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()

        signal = {
            'signal': 'HOLD',
            'entry_time': timestamp,
            'stop_loss': 0,
            'take_profit': 0,
            'reason': 'No signal'
        }

        # Update opening ranges
        if current_date != self.current_date:
            self.current_date = current_date
            self._update_opening_ranges(bar_data)
        else:
            self._update_opening_ranges(bar_data)

        if current_date not in self.daily_ranges:
            return signal

        # Trading hours and limits
        current_time = timestamp.time()
        trade_start = pd.to_datetime("09:45", format="%H:%M").time()
        market_close = pd.to_datetime("16:00", format="%H:%M").time()

        if self.intraday_only:
            close_cutoff = pd.to_datetime("15:45", format="%H:%M").time()
            if current_time > close_cutoff:
                return signal

        if not (trade_start <= current_time <= market_close):
            return signal

        if current_date not in self.daily_trade_count:
            self.daily_trade_count[current_date] = 0

        if self.daily_trade_count[current_date] >= self.max_trades_per_day:
            return signal

        if hasattr(portfolio, 'positions') and len(portfolio.positions) > 0:
            return signal

        or_data = self.daily_ranges[current_date]
        current_price = bar_data['close']

        # Check both timeframes for alignment
        if ('or1_high' in or_data and 'or2_high' in or_data and
            or_data['or1_bars'] > 0 and or_data['or2_bars'] > 0):

            or1_high, or1_low = or_data['or1_high'], or_data['or1_low']
            or2_high, or2_low = or_data['or2_high'], or_data['or2_low']

            # Use the wider range for stops
            combined_high = max(or1_high, or2_high)
            combined_low = min(or1_low, or2_low)

            # Long signal: break above both ranges
            if (current_price > combined_high and
                not or_data.get('combined_long_triggered', False)):

                signal = {
                    'signal': 'BUY',
                    'entry_time': timestamp,
                    'entry_price': current_price,
                    'stop_loss': current_price - self.fixed_stop_points,
                    'take_profit': current_price + (self.fixed_stop_points * self.target_multiplier),
                    'risk_per_trade': self.risk_per_trade,
                    'contracts': self.contracts_per_trade,
                    'reason': f'Multi-TF Long above {combined_high:.2f}'
                }
                or_data['combined_long_triggered'] = True
                self.daily_trade_count[current_date] += 1

            # Short signal: break below both ranges
            elif (current_price < combined_low and
                  not or_data.get('combined_short_triggered', False)):

                signal = {
                    'signal': 'SELL',
                    'entry_time': timestamp,
                    'entry_price': current_price,
                    'stop_loss': current_price + self.fixed_stop_points,
                    'take_profit': current_price - (self.fixed_stop_points * self.target_multiplier),
                    'risk_per_trade': self.risk_per_trade,
                    'contracts': self.contracts_per_trade,
                    'reason': f'Multi-TF Short below {combined_low:.2f}'
                }
                or_data['combined_short_triggered'] = True
                self.daily_trade_count[current_date] += 1

        return signal

    def _update_opening_ranges(self, bar_data: Dict[str, Any]) -> None:
        """Update both 15min and 30min opening ranges."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()
        current_time = timestamp.time()

        if current_date not in self.daily_ranges:
            self.daily_ranges[current_date] = {
                'or1_high': float('-inf'), 'or1_low': float('inf'), 'or1_bars': 0,
                'or2_high': float('-inf'), 'or2_low': float('inf'), 'or2_bars': 0,
                'combined_long_triggered': False, 'combined_short_triggered': False
            }

        or_data = self.daily_ranges[current_date]

        # 15-min OR (single 9:30 bar)
        market_open = pd.to_datetime("09:30", format="%H:%M").time()
        if current_time == market_open:
            or_data['or1_high'] = bar_data['high']
            or_data['or1_low'] = bar_data['low']
            or_data['or1_bars'] = 1

        # 30-min OR (9:30 and 9:45 bars)
        or_30_end = pd.to_datetime("09:45", format="%H:%M").time()
        if current_time in [market_open, or_30_end]:
            if or_data['or2_bars'] == 0:
                or_data['or2_high'] = bar_data['high']
                or_data['or2_low'] = bar_data['low']
            else:
                or_data['or2_high'] = max(or_data['or2_high'], bar_data['high'])
                or_data['or2_low'] = min(or_data['or2_low'], bar_data['low'])
            or_data['or2_bars'] += 1