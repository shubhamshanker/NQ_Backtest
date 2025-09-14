"""
Enhanced Single Contract NQ Strategies for 30+ Points/Day
========================================================
- FIXED: Only 1 contract maximum
- Fair Value Gaps detection
- Bounces from key levels
- Enhanced entry techniques
- Optimized parameters for single contract performance
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
import pandas as pd
from datetime import datetime, time

if TYPE_CHECKING:
    from portfolio import Portfolio

from strategy import Strategy

class FairValueGapORBStrategy(Strategy):
    """
    ORB strategy enhanced with Fair Value Gap detection for better entries.
    Maximum 1 contract, partial profit at 50%.
    """

    def __init__(self, risk_per_trade: float = 0.02, or_minutes: int = 15,
                 fixed_stop_points: float = 65.0, rr_ratio: float = 1.8,
                 max_trades_per_day: int = 4, fvg_min_size: float = 20.0,
                 partial_profit_at_50pct: bool = True, aggressive_targets: bool = True):
        """
        Initialize Fair Value Gap ORB Strategy.

        Args:
            fixed_stop_points: Stop loss distance in points
            rr_ratio: Risk/reward ratio
            fvg_min_size: Minimum fair value gap size in points
            partial_profit_at_50pct: Book 50% profit at 50% of target
            aggressive_targets: Use higher R:R ratios for single contract
        """
        super().__init__(risk_per_trade)
        self.or_minutes = or_minutes
        self.fixed_stop = fixed_stop_points
        self.rr_ratio = rr_ratio
        self.max_trades_per_day = max_trades_per_day
        self.fvg_min_size = fvg_min_size
        self.partial_profit_at_50pct = partial_profit_at_50pct
        self.aggressive_targets = aggressive_targets

        # FIXED: Always 1 contract only
        self.contracts_per_trade = 1

        # State tracking
        self.daily_ranges = {}
        self.daily_trade_count = {}
        self.fair_value_gaps = []
        self.price_history = []
        self.position_states = {}
        self.current_date = None

        self.name = f"FVG_ORB_SL{fixed_stop_points}_RR{rr_ratio}_FVG{fvg_min_size}_MT{max_trades_per_day}"

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """Generate signals with FVG-enhanced entries."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()

        signal = {
            'signal': 'HOLD',
            'entry_time': timestamp,
            'stop_loss': 0,
            'take_profit': 0,
            'reason': 'No signal',
            'contracts': 1  # FIXED: Always 1 contract
        }

        # Update daily data
        if current_date != self.current_date:
            self.current_date = current_date
            self._reset_daily_data(current_date)

        self._update_opening_range(bar_data)
        self._update_price_history(bar_data)
        self._detect_fair_value_gaps()

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

        # Handle existing position
        existing_positions = getattr(portfolio, 'positions', {})
        if existing_positions:
            self._handle_partial_profits(bar_data, portfolio)
            return signal

        # Generate enhanced entry signals
        current_price = bar_data['close']
        or_high = or_data['high']
        or_low = or_data['low']

        # Enhanced long entry conditions
        long_signal = self._check_enhanced_long_entry(current_price, or_high, or_low, bar_data)
        if long_signal:
            target_points = self.fixed_stop * self.rr_ratio

            signal = {
                'signal': 'BUY',
                'entry_time': timestamp,
                'entry_price': current_price,
                'stop_loss': current_price - self.fixed_stop,
                'take_profit': current_price + target_points,
                'risk_per_trade': self.risk_per_trade,
                'contracts': 1,  # FIXED: Always 1 contract
                'reason': long_signal,
                'partial_target': current_price + (target_points * 0.5) if self.partial_profit_at_50pct else None
            }
            self.daily_trade_count[current_date] += 1

        # Enhanced short entry conditions
        short_signal = self._check_enhanced_short_entry(current_price, or_high, or_low, bar_data)
        if short_signal:
            target_points = self.fixed_stop * self.rr_ratio

            signal = {
                'signal': 'SELL',
                'entry_time': timestamp,
                'entry_price': current_price,
                'stop_loss': current_price + self.fixed_stop,
                'take_profit': current_price - target_points,
                'risk_per_trade': self.risk_per_trade,
                'contracts': 1,  # FIXED: Always 1 contract
                'reason': short_signal,
                'partial_target': current_price - (target_points * 0.5) if self.partial_profit_at_50pct else None
            }
            self.daily_trade_count[current_date] += 1

        return signal

    def _check_enhanced_long_entry(self, current_price: float, or_high: float, or_low: float, bar_data: Dict) -> Optional[str]:
        """Check for enhanced long entry conditions."""
        # 1. Basic ORB breakout
        if current_price > or_high:
            # Check for FVG support below
            fvg_support = self._find_nearest_fvg_support(current_price)
            if fvg_support:
                return f'ORB Long breakout above {or_high:.2f} with FVG support at {fvg_support:.2f}'
            else:
                return f'ORB Long breakout above {or_high:.2f}'

        # 2. Bounce from OR low with FVG confluence
        if or_low <= current_price <= or_low + 10:
            fvg_support = self._find_nearest_fvg_support(current_price)
            if fvg_support and abs(current_price - fvg_support) <= 15:
                return f'Long bounce from OR low {or_low:.2f} + FVG confluence {fvg_support:.2f}'

        # 3. Pullback to unfilled FVG in bullish context
        if current_price > or_high:  # Above OR, in bullish mode
            for fvg in self.fair_value_gaps:
                if (fvg['type'] == 'bullish' and not fvg['filled'] and
                    fvg['low'] <= current_price <= fvg['high']):
                    return f'Long entry filling bullish FVG {fvg["low"]:.2f}-{fvg["high"]:.2f}'

        return None

    def _check_enhanced_short_entry(self, current_price: float, or_high: float, or_low: float, bar_data: Dict) -> Optional[str]:
        """Check for enhanced short entry conditions."""
        # 1. Basic ORB breakdown
        if current_price < or_low:
            # Check for FVG resistance above
            fvg_resistance = self._find_nearest_fvg_resistance(current_price)
            if fvg_resistance:
                return f'ORB Short breakdown below {or_low:.2f} with FVG resistance at {fvg_resistance:.2f}'
            else:
                return f'ORB Short breakdown below {or_low:.2f}'

        # 2. Rejection from OR high with FVG confluence
        if or_high - 10 <= current_price <= or_high:
            fvg_resistance = self._find_nearest_fvg_resistance(current_price)
            if fvg_resistance and abs(current_price - fvg_resistance) <= 15:
                return f'Short rejection from OR high {or_high:.2f} + FVG confluence {fvg_resistance:.2f}'

        # 3. Rally to unfilled FVG in bearish context
        if current_price < or_low:  # Below OR, in bearish mode
            for fvg in self.fair_value_gaps:
                if (fvg['type'] == 'bearish' and not fvg['filled'] and
                    fvg['low'] <= current_price <= fvg['high']):
                    return f'Short entry filling bearish FVG {fvg["low"]:.2f}-{fvg["high"]:.2f}'

        return None

    def _detect_fair_value_gaps(self) -> None:
        """Detect fair value gaps in recent price action."""
        if len(self.price_history) < 3:
            return

        # Check last 3 bars for FVG patterns
        for i in range(len(self.price_history) - 2):
            bar1 = self.price_history[i]
            bar2 = self.price_history[i + 1]
            bar3 = self.price_history[i + 2]

            # Bullish FVG: bar1 low > bar3 high (gap between them)
            if bar1['low'] > bar3['high']:
                gap_size = bar1['low'] - bar3['high']
                if gap_size >= self.fvg_min_size:
                    fvg = {
                        'type': 'bullish',
                        'high': bar1['low'],
                        'low': bar3['high'],
                        'size': gap_size,
                        'timestamp': bar2['timestamp'],
                        'filled': False
                    }
                    self.fair_value_gaps.append(fvg)

            # Bearish FVG: bar1 high < bar3 low (gap between them)
            elif bar1['high'] < bar3['low']:
                gap_size = bar3['low'] - bar1['high']
                if gap_size >= self.fvg_min_size:
                    fvg = {
                        'type': 'bearish',
                        'high': bar3['low'],
                        'low': bar1['high'],
                        'size': gap_size,
                        'timestamp': bar2['timestamp'],
                        'filled': False
                    }
                    self.fair_value_gaps.append(fvg)

        # Mark filled FVGs
        current_price = self.price_history[-1]['close'] if self.price_history else 0
        for fvg in self.fair_value_gaps:
            if not fvg['filled'] and fvg['low'] <= current_price <= fvg['high']:
                fvg['filled'] = True

        # Keep only recent unfilled FVGs
        self.fair_value_gaps = [fvg for fvg in self.fair_value_gaps[-10:] if not fvg['filled']]

    def _find_nearest_fvg_support(self, price: float) -> Optional[float]:
        """Find nearest FVG support level below current price."""
        supports = []
        for fvg in self.fair_value_gaps:
            if fvg['type'] == 'bullish' and not fvg['filled'] and fvg['high'] < price:
                supports.append(fvg['high'])
        return max(supports) if supports else None

    def _find_nearest_fvg_resistance(self, price: float) -> Optional[float]:
        """Find nearest FVG resistance level above current price."""
        resistances = []
        for fvg in self.fair_value_gaps:
            if fvg['type'] == 'bearish' and not fvg['filled'] and fvg['low'] > price:
                resistances.append(fvg['low'])
        return min(resistances) if resistances else None

    def _handle_partial_profits(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> None:
        """Handle partial profit booking at 50% of target."""
        if not self.partial_profit_at_50pct:
            return

        current_price = bar_data['close']
        for position_id, position in getattr(portfolio, 'positions', {}).items():
            partial_target = getattr(position, 'partial_target', None)
            if partial_target and not getattr(position, 'partial_booked', False):

                # Check if partial target hit
                if hasattr(position, 'direction'):
                    if (position.direction == 'LONG' and current_price >= partial_target):
                        # Book 50% profit (simplified - would integrate with portfolio)
                        position.partial_booked = True
                        print(f"Partial profit booked at 50% target: {partial_target:.2f}")
                    elif (position.direction == 'SHORT' and current_price <= partial_target):
                        position.partial_booked = True
                        print(f"Partial profit booked at 50% target: {partial_target:.2f}")

    def _update_price_history(self, bar_data: Dict[str, Any]) -> None:
        """Update price history for FVG detection."""
        self.price_history.append({
            'timestamp': bar_data['timestamp'],
            'high': bar_data['high'],
            'low': bar_data['low'],
            'close': bar_data['close']
        })

        # Keep only last 20 bars
        if len(self.price_history) > 20:
            self.price_history = self.price_history[-20:]

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


class OptimizedSingleContractORB(Strategy):
    """
    Heavily optimized ORB strategy for single contract to achieve 30+ points/day.
    """

    def __init__(self, risk_per_trade: float = 0.025, or_minutes: int = 15,
                 fixed_stop_points: float = 55.0, rr_ratio: float = 2.2,
                 max_trades_per_day: int = 5, momentum_filter: bool = True,
                 volume_filter: bool = True, session_bias: bool = True):
        """
        Initialize Optimized Single Contract ORB.

        Higher risk per trade and R:R ratio to compensate for single contract.
        """
        super().__init__(risk_per_trade)
        self.or_minutes = or_minutes
        self.fixed_stop = fixed_stop_points
        self.rr_ratio = rr_ratio
        self.max_trades_per_day = max_trades_per_day
        self.momentum_filter = momentum_filter
        self.volume_filter = volume_filter
        self.session_bias = session_bias

        # FIXED: Always 1 contract
        self.contracts_per_trade = 1

        # State tracking
        self.daily_ranges = {}
        self.daily_trade_count = {}
        self.session_momentum = {}
        self.price_momentum = []
        self.current_date = None

        self.name = f"OptSingle_SL{fixed_stop_points}_RR{rr_ratio}_Risk{risk_per_trade}_MT{max_trades_per_day}"

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """Generate optimized signals for single contract performance."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()

        signal = {
            'signal': 'HOLD',
            'entry_time': timestamp,
            'stop_loss': 0,
            'take_profit': 0,
            'reason': 'No signal',
            'contracts': 1  # FIXED: Always 1 contract
        }

        # Update daily data
        if current_date != self.current_date:
            self.current_date = current_date
            self._reset_daily_data(current_date)

        self._update_opening_range(bar_data)
        self._update_momentum_tracking(bar_data)

        if current_date not in self.daily_ranges:
            return signal

        or_data = self.daily_ranges[current_date]
        if or_data['bars_counted'] == 0:
            return signal

        # Trading hours with session awareness
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

        # Apply filters
        if not self._passes_filters(bar_data, current_time):
            return signal

        current_price = bar_data['close']
        or_high = or_data['high']
        or_low = or_data['low']

        # Optimized entry conditions
        # Long breakout with momentum confirmation
        if current_price > or_high and self._confirm_bullish_momentum(bar_data):
            target_points = self.fixed_stop * self.rr_ratio

            signal = {
                'signal': 'BUY',
                'entry_time': timestamp,
                'entry_price': current_price,
                'stop_loss': current_price - self.fixed_stop,
                'take_profit': current_price + target_points,
                'risk_per_trade': self.risk_per_trade,
                'contracts': 1,
                'reason': f'Optimized ORB Long above {or_high:.2f} with momentum confirmation'
            }
            self.daily_trade_count[current_date] += 1

        # Short breakdown with momentum confirmation
        elif current_price < or_low and self._confirm_bearish_momentum(bar_data):
            target_points = self.fixed_stop * self.rr_ratio

            signal = {
                'signal': 'SELL',
                'entry_time': timestamp,
                'entry_price': current_price,
                'stop_loss': current_price + self.fixed_stop,
                'take_profit': current_price - target_points,
                'risk_per_trade': self.risk_per_trade,
                'contracts': 1,
                'reason': f'Optimized ORB Short below {or_low:.2f} with momentum confirmation'
            }
            self.daily_trade_count[current_date] += 1

        return signal

    def _passes_filters(self, bar_data: Dict[str, Any], current_time: time) -> bool:
        """Apply various filters to improve entry quality."""
        # Session bias filter - prefer certain times
        if self.session_bias:
            morning_power_hour = time(9, 45) <= current_time <= time(10, 45)
            lunch_breakout = time(13, 30) <= current_time <= time(14, 30)
            close_momentum = time(15, 0) <= current_time <= time(15, 45)

            if not (morning_power_hour or lunch_breakout or close_momentum):
                return False

        # Volume filter - ensure sufficient activity
        if self.volume_filter:
            bar_range = bar_data['high'] - bar_data['low']
            if bar_range < 15:  # Minimum 15 point range for quality
                return False

        return True

    def _confirm_bullish_momentum(self, bar_data: Dict[str, Any]) -> bool:
        """Confirm bullish momentum before entry."""
        if not self.momentum_filter or len(self.price_momentum) < 3:
            return True

        # Check if recent bars show upward momentum
        recent_closes = [bar['close'] for bar in self.price_momentum[-3:]]
        return recent_closes[-1] > recent_closes[0]  # Close higher than 3 bars ago

    def _confirm_bearish_momentum(self, bar_data: Dict[str, Any]) -> bool:
        """Confirm bearish momentum before entry."""
        if not self.momentum_filter or len(self.price_momentum) < 3:
            return True

        # Check if recent bars show downward momentum
        recent_closes = [bar['close'] for bar in self.price_momentum[-3:]]
        return recent_closes[-1] < recent_closes[0]  # Close lower than 3 bars ago

    def _update_momentum_tracking(self, bar_data: Dict[str, Any]) -> None:
        """Update momentum tracking data."""
        self.price_momentum.append({
            'timestamp': bar_data['timestamp'],
            'close': bar_data['close'],
            'high': bar_data['high'],
            'low': bar_data['low']
        })

        # Keep only last 10 bars
        if len(self.price_momentum) > 10:
            self.price_momentum = self.price_momentum[-10:]

    def _reset_daily_data(self, current_date) -> None:
        """Reset daily data."""
        self.daily_ranges[current_date] = {
            'high': float('-inf'),
            'low': float('inf'),
            'bars_counted': 0
        }
        self.daily_trade_count[current_date] = 0
        self.price_momentum = []

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


class HighFrequencySingleContract(Strategy):
    """
    High-frequency approach optimized for single contract to achieve 30+ points.
    """

    def __init__(self, risk_per_trade: float = 0.015, or_minutes: int = 15,
                 scalp_target: float = 35.0, stop_points: float = 25.0,
                 max_trades_per_day: int = 12, aggressive_entries: bool = True):
        """Initialize high-frequency single contract strategy."""
        super().__init__(risk_per_trade)
        self.or_minutes = or_minutes
        self.scalp_target = scalp_target
        self.stop_points = stop_points
        self.max_trades_per_day = max_trades_per_day
        self.aggressive_entries = aggressive_entries

        # FIXED: Always 1 contract
        self.contracts_per_trade = 1

        # State tracking
        self.daily_ranges = {}
        self.daily_trade_count = {}
        self.daily_pnl = {}
        self.current_date = None

        self.name = f"HF_Single_T{scalp_target}_SL{stop_points}_MT{max_trades_per_day}"

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """Generate high-frequency signals optimized for single contract."""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()

        signal = {
            'signal': 'HOLD',
            'entry_time': timestamp,
            'stop_loss': 0,
            'take_profit': 0,
            'reason': 'No signal',
            'contracts': 1  # FIXED: Always 1 contract
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

        # Extended trading hours for high frequency
        current_time = timestamp.time()
        trade_start = time(9, 45)
        close_cutoff = time(15, 30)

        if not (trade_start <= current_time <= close_cutoff):
            return signal

        # Dynamic trade limit based on performance
        effective_max_trades = self._get_effective_trade_limit(current_date)
        if self.daily_trade_count[current_date] >= effective_max_trades:
            return signal

        if hasattr(portfolio, 'positions') and len(portfolio.positions) > 0:
            return signal

        current_price = bar_data['close']
        or_high = or_data['high']
        or_low = or_data['low']
        or_mid = (or_high + or_low) / 2

        # High-frequency scalping opportunities
        # More aggressive entries around OR levels
        if self.aggressive_entries:
            # Long entries near OR support
            if (or_low <= current_price <= or_mid + 5 and
                current_price > or_low + 1):

                signal = {
                    'signal': 'BUY',
                    'entry_time': timestamp,
                    'entry_price': current_price,
                    'stop_loss': current_price - self.stop_points,
                    'take_profit': current_price + self.scalp_target,
                    'risk_per_trade': self.risk_per_trade,
                    'contracts': 1,
                    'reason': f'HF Long scalp near OR support {or_low:.2f}'
                }
                self.daily_trade_count[current_date] += 1

            # Short entries near OR resistance
            elif (or_mid - 5 <= current_price <= or_high and
                  current_price < or_high - 1):

                signal = {
                    'signal': 'SELL',
                    'entry_time': timestamp,
                    'entry_price': current_price,
                    'stop_loss': current_price + self.stop_points,
                    'take_profit': current_price - self.scalp_target,
                    'risk_per_trade': self.risk_per_trade,
                    'contracts': 1,
                    'reason': f'HF Short scalp near OR resistance {or_high:.2f}'
                }
                self.daily_trade_count[current_date] += 1

        return signal

    def _get_effective_trade_limit(self, current_date) -> int:
        """Adjust trade limit based on daily P&L."""
        base_limit = self.max_trades_per_day

        if current_date in self.daily_pnl:
            daily_profit = self.daily_pnl[current_date]
            if daily_profit >= 20:  # If up 20+ points, reduce frequency
                return max(base_limit - 3, 5)
            elif daily_profit <= -15:  # If down 15+ points, reduce frequency
                return max(base_limit - 4, 3)

        return base_limit

    def _reset_daily_data(self, current_date) -> None:
        """Reset daily data."""
        self.daily_ranges[current_date] = {
            'high': float('-inf'),
            'low': float('inf'),
            'bars_counted': 0
        }
        self.daily_trade_count[current_date] = 0
        self.daily_pnl[current_date] = 0

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