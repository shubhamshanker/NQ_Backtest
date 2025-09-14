"""
Advanced NQ ORB Strategies with Trailing Stops and Partial Profits
================================================================
Creative implementations to achieve 30+ points/day with low drawdown:
- Trailing stops based on 5-min candles
- Partial profit booking at 50%
- Breakeven moves
- Multiple R:R ratios including 0.8:1 and 1:1
- Advanced position management
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
import pandas as pd
from datetime import datetime, time

if TYPE_CHECKING:
    from portfolio import Portfolio

from strategy import Strategy

class AdvancedORBStrategy(Strategy):
    """
    Advanced ORB strategy with trailing stops, partial profits, and breakeven moves.
    """

    def __init__(self, risk_per_trade: float = 0.02, or_minutes: int = 15,
                 fixed_stop_points: float = 100.0, rr_ratio: float = 1.0,
                 max_trades_per_day: int = 3, contracts_per_trade: int = 2,
                 trailing_candles: int = 2, partial_profit_pct: float = 0.5,
                 move_to_breakeven_at_pct: float = 0.7, intraday_only: bool = True):
        """
        Initialize Advanced ORB Strategy.

        Args:
            fixed_stop_points: Stop loss distance in points
            rr_ratio: Risk/Reward ratio (0.8, 1.0, 1.5, etc.)
            trailing_candles: Number of 5-min candles to trail below/above
            partial_profit_pct: Percentage of position to close at partial target
            move_to_breakeven_at_pct: Move stop to breakeven when this % of target reached
        """
        super().__init__(risk_per_trade)
        self.or_minutes = or_minutes
        self.fixed_stop = fixed_stop_points
        self.rr_ratio = rr_ratio
        self.max_trades_per_day = max_trades_per_day
        self.contracts_per_trade = contracts_per_trade
        self.trailing_candles = trailing_candles
        self.partial_profit_pct = partial_profit_pct
        self.move_to_breakeven_at_pct = move_to_breakeven_at_pct
        self.intraday_only = intraday_only

        # State tracking
        self.daily_ranges = {}
        self.daily_trade_count = {}
        self.position_states = {}  # Track partial profits, breakeven moves, etc.
        self.price_history_5min = []  # For trailing stops
        self.current_date = None
        self.pending_signals = {}  # Store signals for next bar entry

        self.name = f"AdvORB_SL{fixed_stop_points}_RR{rr_ratio}_T{trailing_candles}_MT{max_trades_per_day}"

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """Generate advanced ORB signals with sophisticated position management."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()

        signal = {
            'signal': 'HOLD',
            'entry_time': timestamp,
            'stop_loss': 0,
            'take_profit': 0,
            'reason': 'No signal',
            'advanced_features': {
                'partial_profit_target': None,
                'breakeven_trigger': None,
                'trailing_stop_active': False
            }
        }

        # Check for pending signal from previous bar
        if current_date in self.pending_signals:
            pending_signal = self.pending_signals[current_date]
            # Execute with current bar's open price (next bar entry)
            entry_price = bar_data['open']

            # Update signal with correct entry price
            pending_signal['entry_price'] = entry_price
            if pending_signal['signal'] == 'BUY':
                pending_signal['stop_loss'] = entry_price - self.fixed_stop
                target_points = self.fixed_stop * self.rr_ratio
                pending_signal['take_profit'] = entry_price + target_points
                pending_signal['advanced_features']['partial_profit_target'] = entry_price + (target_points * self.partial_profit_pct)
                pending_signal['advanced_features']['breakeven_trigger'] = entry_price + (target_points * self.move_to_breakeven_at_pct)
                pending_signal['advanced_features']['original_stop'] = entry_price - self.fixed_stop
            else:  # SELL
                pending_signal['stop_loss'] = entry_price + self.fixed_stop
                target_points = self.fixed_stop * self.rr_ratio
                pending_signal['take_profit'] = entry_price - target_points
                pending_signal['advanced_features']['partial_profit_target'] = entry_price - (target_points * self.partial_profit_pct)
                pending_signal['advanced_features']['breakeven_trigger'] = entry_price - (target_points * self.move_to_breakeven_at_pct)
                pending_signal['advanced_features']['original_stop'] = entry_price + self.fixed_stop

            # Clear pending signal and return it
            del self.pending_signals[current_date]
            return pending_signal

        # Update opening range
        if current_date != self.current_date:
            self.current_date = current_date
            self._reset_daily_data(current_date)

        self._update_opening_range(bar_data)
        self._update_5min_history(bar_data)

        # Check if we have a valid OR
        if current_date not in self.daily_ranges:
            return signal

        or_data = self.daily_ranges[current_date]
        if or_data['bars_counted'] == 0:
            return signal

        # Trading hours
        current_time = timestamp.time()
        trade_start = time(9, 45)  # 9:45 AM
        close_cutoff = time(15, 45) if self.intraday_only else time(16, 0)

        if not (trade_start <= current_time <= close_cutoff):
            return signal

        # Check trade limits
        if self.daily_trade_count[current_date] >= self.max_trades_per_day:
            return signal

        # Check existing positions - handle advanced position management
        existing_positions = getattr(portfolio, 'positions', {})
        if existing_positions:
            self._handle_advanced_position_management(bar_data, portfolio)
            return signal  # One position at a time

        # Generate entry signals
        current_price = bar_data['close']
        or_high = or_data['high']
        or_low = or_data['low']

        # Detect breakouts but store as pending signals (enter next bar)
        if current_price > or_high:
            # Store pending long signal for next bar
            self.pending_signals[current_date] = {
                'signal': 'BUY',
                'entry_time': timestamp,
                'entry_price': 0,  # Will be filled with next bar's open
                'stop_loss': 0,    # Will be calculated with actual entry price
                'take_profit': 0,  # Will be calculated with actual entry price
                'risk_per_trade': self.risk_per_trade,
                'contracts': self.contracts_per_trade,
                'reason': f'ORB Long breakout above {or_high:.2f}',
                'advanced_features': {
                    'partial_profit_target': 0,  # Will be calculated
                    'breakeven_trigger': 0,      # Will be calculated
                    'trailing_stop_active': True,
                    'original_stop': 0,          # Will be calculated
                    'direction': 'LONG'
                }
            }
            self.daily_trade_count[current_date] += 1

        # Short breakdown
        elif current_price < or_low:
            # Store pending short signal for next bar
            self.pending_signals[current_date] = {
                'signal': 'SELL',
                'entry_time': timestamp,
                'entry_price': 0,  # Will be filled with next bar's open
                'stop_loss': 0,    # Will be calculated with actual entry price
                'take_profit': 0,  # Will be calculated with actual entry price
                'risk_per_trade': self.risk_per_trade,
                'contracts': self.contracts_per_trade,
                'reason': f'ORB Short breakdown below {or_low:.2f}',
                'advanced_features': {
                    'partial_profit_target': 0,  # Will be calculated
                    'breakeven_trigger': 0,      # Will be calculated
                    'trailing_stop_active': True,
                    'original_stop': 0,          # Will be calculated
                    'direction': 'SHORT'
                }
            }
            self.daily_trade_count[current_date] += 1

        return signal

    def _handle_advanced_position_management(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> None:
        """Handle trailing stops, partial profits, and breakeven moves."""
        if not hasattr(portfolio, 'positions') or not portfolio.positions:
            return

        for position_id, position in portfolio.positions.items():
            current_price = bar_data['close']

            # Get advanced features from position (if stored)
            adv_features = getattr(position, 'advanced_features', {})
            if not adv_features:
                continue

            direction = adv_features.get('direction', 'LONG')
            entry_price = position.entry_price

            # 1. Partial Profit Booking at 50%
            partial_target = adv_features.get('partial_profit_target')
            if partial_target and not adv_features.get('partial_booked', False):
                if direction == 'LONG' and current_price >= partial_target:
                    self._book_partial_profit(position, portfolio, 0.5)
                    adv_features['partial_booked'] = True
                elif direction == 'SHORT' and current_price <= partial_target:
                    self._book_partial_profit(position, portfolio, 0.5)
                    adv_features['partial_booked'] = True

            # 2. Move to Breakeven
            breakeven_trigger = adv_features.get('breakeven_trigger')
            if breakeven_trigger and not adv_features.get('moved_to_breakeven', False):
                if direction == 'LONG' and current_price >= breakeven_trigger:
                    position.stop_loss = entry_price + 2.0  # Small profit
                    adv_features['moved_to_breakeven'] = True
                elif direction == 'SHORT' and current_price <= breakeven_trigger:
                    position.stop_loss = entry_price - 2.0  # Small profit
                    adv_features['moved_to_breakeven'] = True

            # 3. Trailing Stop based on 5-min candles
            if adv_features.get('trailing_stop_active', False) and len(self.price_history_5min) >= self.trailing_candles:
                self._update_trailing_stop(position, direction, adv_features)

    def _book_partial_profit(self, position, portfolio, percentage: float) -> None:
        """Book partial profit at specified percentage."""
        partial_contracts = int(position.contracts * percentage)
        if partial_contracts > 0:
            # Create partial exit (simplified - would need portfolio integration)
            remaining_contracts = position.contracts - partial_contracts
            position.contracts = remaining_contracts

            # Log partial profit (would be integrated with portfolio)
            print(f"Partial profit booked: {partial_contracts} contracts at {percentage*100}% target")

    def _update_trailing_stop(self, position, direction: str, adv_features: Dict) -> None:
        """Update trailing stop based on recent 5-min candles."""
        if len(self.price_history_5min) < self.trailing_candles:
            return

        recent_candles = self.price_history_5min[-self.trailing_candles:]

        if direction == 'LONG':
            # Trail below recent lows
            recent_low = min(candle['low'] for candle in recent_candles)
            new_stop = recent_low - 5.0  # 5 point buffer

            # Only move stop up, never down
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop

        elif direction == 'SHORT':
            # Trail above recent highs
            recent_high = max(candle['high'] for candle in recent_candles)
            new_stop = recent_high + 5.0  # 5 point buffer

            # Only move stop down, never up
            if new_stop < position.stop_loss:
                position.stop_loss = new_stop

    def _update_5min_history(self, bar_data: Dict[str, Any]) -> None:
        """Update 5-minute price history for trailing stops."""
        # Convert 15-min bar to simulated 5-min data
        timestamp = bar_data['timestamp']

        # Create 3 simulated 5-min bars from each 15-min bar
        for i in range(3):
            sim_timestamp = timestamp + pd.Timedelta(minutes=i*5)
            sim_bar = {
                'timestamp': sim_timestamp,
                'high': bar_data['high'],
                'low': bar_data['low'],
                'close': bar_data['close']
            }
            self.price_history_5min.append(sim_bar)

        # Keep only last 20 bars (100 minutes of 5-min data)
        if len(self.price_history_5min) > 20:
            self.price_history_5min = self.price_history_5min[-20:]

    def _reset_daily_data(self, current_date) -> None:
        """Reset daily tracking data."""
        self.daily_ranges[current_date] = {
            'high': float('-inf'),
            'low': float('inf'),
            'bars_counted': 0
        }
        self.daily_trade_count[current_date] = 0
        self.price_history_5min = []
        # Clear any pending signals from previous day
        if current_date in self.pending_signals:
            del self.pending_signals[current_date]

    def _update_opening_range(self, bar_data: Dict[str, Any]) -> None:
        """Update opening range for current day."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()
        current_time = timestamp.time()

        if current_date not in self.daily_ranges:
            self._reset_daily_data(current_date)

        # For 15-minute bars, opening range is the 9:30 AM bar
        market_open = time(9, 30)
        if current_time == market_open:
            or_data = self.daily_ranges[current_date]
            or_data['high'] = bar_data['high']
            or_data['low'] = bar_data['low']
            or_data['bars_counted'] = 1


class DynamicRRORBStrategy(Strategy):
    """
    Dynamic Risk/Reward ORB strategy that adjusts R:R based on market conditions.
    """

    def __init__(self, risk_per_trade: float = 0.02, or_minutes: int = 15,
                 fixed_stop_points: float = 100.0, base_rr_ratio: float = 1.0,
                 max_trades_per_day: int = 3, contracts_per_trade: int = 1,
                 volatility_adjustment: bool = True):
        """Initialize Dynamic R:R ORB Strategy."""
        super().__init__(risk_per_trade)
        self.or_minutes = or_minutes
        self.fixed_stop = fixed_stop_points
        self.base_rr_ratio = base_rr_ratio
        self.max_trades_per_day = max_trades_per_day
        self.contracts_per_trade = contracts_per_trade
        self.volatility_adjustment = volatility_adjustment

        # State tracking
        self.daily_ranges = {}
        self.daily_trade_count = {}
        self.volatility_history = []
        self.current_date = None

        self.name = f"DynRR_ORB_SL{fixed_stop_points}_BaseRR{base_rr_ratio}_MT{max_trades_per_day}"

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """Generate signals with dynamic R:R adjustment."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()

        signal = {
            'signal': 'HOLD',
            'entry_time': timestamp,
            'stop_loss': 0,
            'take_profit': 0,
            'reason': 'No signal'
        }

        # Update daily data
        if current_date != self.current_date:
            self.current_date = current_date
            self._reset_daily_data(current_date)

        self._update_opening_range(bar_data)
        self._update_volatility_history(bar_data)

        if current_date not in self.daily_ranges:
            return signal

        or_data = self.daily_ranges[current_date]
        if or_data['bars_counted'] == 0:
            return signal

        # Trading hours
        current_time = timestamp.time()
        trade_start = time(9, 45)
        close_cutoff = time(15, 45)

        if not (trade_start <= current_time <= close_cutoff):
            return signal

        # Check limits
        if self.daily_trade_count[current_date] >= self.max_trades_per_day:
            return signal

        if hasattr(portfolio, 'positions') and len(portfolio.positions) > 0:
            return signal

        # Calculate dynamic R:R ratio
        dynamic_rr = self._calculate_dynamic_rr()

        current_price = bar_data['close']
        or_high = or_data['high']
        or_low = or_data['low']

        # Long breakout
        if current_price > or_high:
            target_points = self.fixed_stop * dynamic_rr

            signal = {
                'signal': 'BUY',
                'entry_time': timestamp,
                'entry_price': current_price,
                'stop_loss': current_price - self.fixed_stop,
                'take_profit': current_price + target_points,
                'risk_per_trade': self.risk_per_trade,
                'contracts': self.contracts_per_trade,
                'reason': f'Dynamic ORB Long (RR={dynamic_rr:.2f}) above {or_high:.2f}'
            }
            self.daily_trade_count[current_date] += 1

        # Short breakdown
        elif current_price < or_low:
            target_points = self.fixed_stop * dynamic_rr

            signal = {
                'signal': 'SELL',
                'entry_time': timestamp,
                'entry_price': current_price,
                'stop_loss': current_price + self.fixed_stop,
                'take_profit': current_price - target_points,
                'risk_per_trade': self.risk_per_trade,
                'contracts': self.contracts_per_trade,
                'reason': f'Dynamic ORB Short (RR={dynamic_rr:.2f}) below {or_low:.2f}'
            }
            self.daily_trade_count[current_date] += 1

        return signal

    def _calculate_dynamic_rr(self) -> float:
        """Calculate dynamic R:R ratio based on market conditions."""
        if not self.volatility_adjustment or len(self.volatility_history) < 5:
            return self.base_rr_ratio

        # Use recent volatility to adjust R:R
        recent_volatility = sum(self.volatility_history[-5:]) / 5
        avg_volatility = sum(self.volatility_history) / len(self.volatility_history)

        volatility_ratio = recent_volatility / avg_volatility if avg_volatility > 0 else 1.0

        # Adjust R:R: higher volatility = higher targets
        if volatility_ratio > 1.2:
            return min(self.base_rr_ratio * 1.5, 2.0)  # Cap at 2:1
        elif volatility_ratio < 0.8:
            return max(self.base_rr_ratio * 0.8, 0.8)  # Floor at 0.8:1
        else:
            return self.base_rr_ratio

    def _update_volatility_history(self, bar_data: Dict[str, Any]) -> None:
        """Update volatility tracking."""
        bar_range = bar_data['high'] - bar_data['low']
        self.volatility_history.append(bar_range)

        # Keep only last 20 bars
        if len(self.volatility_history) > 20:
            self.volatility_history = self.volatility_history[-20:]

    def _reset_daily_data(self, current_date) -> None:
        """Reset daily data."""
        self.daily_ranges[current_date] = {
            'high': float('-inf'),
            'low': float('inf'),
            'bars_counted': 0
        }
        self.daily_trade_count[current_date] = 0

    def _update_opening_range(self, bar_data: Dict[str, Any]) -> None:
        """Update opening range."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()
        current_time = timestamp.time()

        if current_date not in self.daily_ranges:
            self._reset_daily_data(current_date)

        market_open = time(9, 30)
        if current_time == market_open:
            or_data = self.daily_ranges[current_date]
            or_data['high'] = bar_data['high']
            or_data['low'] = bar_data['low']
            or_data['bars_counted'] = 1


class ScalperORBStrategy(Strategy):
    """
    High-frequency scalping ORB strategy with tight risk management.
    """

    def __init__(self, risk_per_trade: float = 0.015, or_minutes: int = 15,
                 scalp_target_points: float = 25.0, stop_points: float = 40.0,
                 max_trades_per_day: int = 8, contracts_per_trade: int = 1):
        """Initialize Scalper ORB Strategy."""
        super().__init__(risk_per_trade)
        self.or_minutes = or_minutes
        self.scalp_target = scalp_target_points
        self.stop_points = stop_points
        self.max_trades_per_day = max_trades_per_day
        self.contracts_per_trade = contracts_per_trade

        # Lower risk per trade for higher frequency
        self.risk_per_trade = risk_per_trade

        # State tracking
        self.daily_ranges = {}
        self.daily_trade_count = {}
        self.consecutive_losses = 0
        self.daily_pnl = {}
        self.current_date = None

        self.name = f"ScalpORB_T{scalp_target_points}_SL{stop_points}_MT{max_trades_per_day}"

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """Generate scalping signals with tight risk control."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()

        signal = {
            'signal': 'HOLD',
            'entry_time': timestamp,
            'stop_loss': 0,
            'take_profit': 0,
            'reason': 'No signal'
        }

        # Update daily data
        if current_date != self.current_date:
            self.current_date = current_date
            self._reset_daily_data(current_date)

        self._update_opening_range(bar_data)

        if current_date not in self.daily_ranges:
            return signal

        or_data = self.daily_ranges[current_date]
        if or_data['bars_counted'] == 0:
            return signal

        # Extended trading hours for scalping
        current_time = timestamp.time()
        trade_start = time(9, 45)
        close_cutoff = time(15, 30)

        if not (trade_start <= current_time <= close_cutoff):
            return signal

        # Dynamic trade limits based on performance
        effective_max_trades = self._get_effective_max_trades(current_date)

        if self.daily_trade_count[current_date] >= effective_max_trades:
            return signal

        if hasattr(portfolio, 'positions') and len(portfolio.positions) > 0:
            return signal

        # Risk adjustment based on consecutive losses
        current_risk = self.risk_per_trade * (0.8 ** min(self.consecutive_losses, 3))

        current_price = bar_data['close']
        or_high = or_data['high']
        or_low = or_data['low']
        or_mid = (or_high + or_low) / 2

        # Scalping opportunities near OR levels
        # Long setups
        if (or_low <= current_price <= or_mid + 10 and
            current_price > or_low + 2):  # Small buffer above OR low

            signal = {
                'signal': 'BUY',
                'entry_time': timestamp,
                'entry_price': current_price,
                'stop_loss': current_price - self.stop_points,
                'take_profit': current_price + self.scalp_target,
                'risk_per_trade': current_risk,
                'contracts': self.contracts_per_trade,
                'reason': f'Scalp Long near OR support {or_low:.2f}'
            }
            self.daily_trade_count[current_date] += 1

        # Short setups
        elif (or_mid - 10 <= current_price <= or_high and
              current_price < or_high - 2):  # Small buffer below OR high

            signal = {
                'signal': 'SELL',
                'entry_time': timestamp,
                'entry_price': current_price,
                'stop_loss': current_price + self.stop_points,
                'take_profit': current_price - self.scalp_target,
                'risk_per_trade': current_risk,
                'contracts': self.contracts_per_trade,
                'reason': f'Scalp Short near OR resistance {or_high:.2f}'
            }
            self.daily_trade_count[current_date] += 1

        return signal

    def _get_effective_max_trades(self, current_date) -> int:
        """Adjust max trades based on daily performance."""
        if current_date in self.daily_pnl:
            daily_profit = self.daily_pnl[current_date]
            if daily_profit < -50:  # If down 50 points today
                return max(self.max_trades_per_day - 2, 3)  # Reduce trades

        return self.max_trades_per_day

    def _reset_daily_data(self, current_date) -> None:
        """Reset daily data."""
        self.daily_ranges[current_date] = {
            'high': float('-inf'),
            'low': float('inf'),
            'bars_counted': 0
        }
        self.daily_trade_count[current_date] = 0
        self.daily_pnl[current_date] = 0
        self.consecutive_losses = 0

    def _update_opening_range(self, bar_data: Dict[str, Any]) -> None:
        """Update opening range."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()
        current_time = timestamp.time()

        if current_date not in self.daily_ranges:
            self._reset_daily_data(current_date)

        market_open = time(9, 30)
        if current_time == market_open:
            or_data = self.daily_ranges[current_date]
            or_data['high'] = bar_data['high']
            or_data['low'] = bar_data['low']
            or_data['bars_counted'] = 1