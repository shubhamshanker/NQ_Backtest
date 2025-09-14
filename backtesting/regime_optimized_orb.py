"""
Regime-Optimized ORB Strategy - Target 15-30+ Points/Day
=======================================================
Ultimate ORB strategy with:
- Market regime detection (Trending/Ranging/Volatile/Quiet)
- Dynamic risk-reward ratios (3:1 to 7:1)
- Advanced entry filters (momentum, volatility, volume)
- Maximum 3 trades per day, 1 contract only
- Sophisticated trade management and loss prevention
"""

from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from portfolio import Portfolio

from strategy import Strategy

class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"
    BREAKOUT_SETUP = "breakout_setup"

@dataclass
class RegimeParameters:
    """Parameters for each market regime"""
    or_minutes: int
    stop_points: float
    rr_ratio: float
    max_trades: int
    momentum_threshold: float
    volatility_threshold: float

class RegimeOptimizedORB(Strategy):
    """Ultimate ORB strategy optimized for 15-30+ points per day"""

    def __init__(self,
                 risk_per_trade: float = 0.02,
                 target_points_per_day: float = 20.0,
                 max_trades_per_day: int = 3,
                 adaptive_parameters: bool = True,
                 enable_regime_detection: bool = True):
        """
        Initialize Regime-Optimized ORB Strategy

        Args:
            risk_per_trade: Risk per trade (always 1 contract)
            target_points_per_day: Daily target in points (15-30+)
            max_trades_per_day: Maximum trades per day (1-3)
            adaptive_parameters: Use dynamic parameter adjustment
            enable_regime_detection: Enable market regime classification
        """
        super().__init__(risk_per_trade)
        self.target_points_per_day = target_points_per_day
        self.max_trades_per_day = min(max_trades_per_day, 3)  # Hard cap at 3
        self.adaptive_parameters = adaptive_parameters
        self.enable_regime_detection = enable_regime_detection

        # State tracking
        self.daily_data = {}
        self.daily_trade_count = {}
        self.daily_pnl = {}
        self.price_history = []
        self.volume_history = []
        self.regime_history = []
        self.current_date = None
        self.consecutive_losses = 0
        self.stop_trading_today = False

        # Regime-specific parameters
        self.regime_params = {
            MarketRegime.TRENDING_UP: RegimeParameters(
                or_minutes=15, stop_points=20.0, rr_ratio=6.0,
                max_trades=3, momentum_threshold=0.7, volatility_threshold=25.0
            ),
            MarketRegime.TRENDING_DOWN: RegimeParameters(
                or_minutes=15, stop_points=20.0, rr_ratio=6.0,
                max_trades=3, momentum_threshold=0.7, volatility_threshold=25.0
            ),
            MarketRegime.RANGING: RegimeParameters(
                or_minutes=30, stop_points=15.0, rr_ratio=3.0,
                max_trades=2, momentum_threshold=0.4, volatility_threshold=15.0
            ),
            MarketRegime.HIGH_VOLATILITY: RegimeParameters(
                or_minutes=30, stop_points=40.0, rr_ratio=7.0,
                max_trades=2, momentum_threshold=0.8, volatility_threshold=50.0
            ),
            MarketRegime.LOW_VOLATILITY: RegimeParameters(
                or_minutes=45, stop_points=10.0, rr_ratio=4.0,
                max_trades=1, momentum_threshold=0.3, volatility_threshold=8.0
            ),
            MarketRegime.BREAKOUT_SETUP: RegimeParameters(
                or_minutes=15, stop_points=25.0, rr_ratio=7.0,
                max_trades=3, momentum_threshold=0.9, volatility_threshold=30.0
            )
        }

        # Current regime and parameters
        self.current_regime = MarketRegime.RANGING
        self.current_params = self.regime_params[self.current_regime]

        self.name = f"RegimeORB_T{target_points_per_day}_MT{max_trades_per_day}"

    def generate_signal(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> Dict[str, Any]:
        """Generate optimized ORB signals based on market regime"""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()

        # Default signal
        signal = {
            'signal': 'HOLD',
            'entry_time': timestamp,
            'stop_loss': 0,
            'take_profit': 0,
            'reason': 'No signal',
            'contracts': 1,
            'regime': self.current_regime.value
        }

        # Session validation
        if not self._validate_session_timing(bar_data, allow_extended=False):
            return signal

        # Daily reset and initialization
        if current_date != self.current_date:
            self._reset_daily_data(current_date)
            self.current_date = current_date

        # Update market data
        self._update_market_data(bar_data)

        # Detect market regime
        if self.enable_regime_detection:
            self._detect_market_regime(bar_data)

        # Check if we should stop trading today
        if self._should_stop_trading_today():
            signal['reason'] = 'Daily trading stopped due to losses/limits'
            return signal

        # Trading session timing
        if not self._is_trading_time(timestamp):
            return signal

        # Maximum trades check
        if self.daily_trade_count[current_date] >= self.current_params.max_trades:
            signal['reason'] = f'Max trades ({self.current_params.max_trades}) reached for regime {self.current_regime.value}'
            return signal

        # Handle existing positions
        existing_positions = getattr(portfolio, 'positions', {})
        if existing_positions:
            self._manage_existing_positions(bar_data, portfolio)
            return signal

        # Generate entry signals if we have enough data
        if not self._has_sufficient_data():
            return signal

        # Get current regime parameters
        params = self.current_params

        # Calculate opening range
        or_data = self._get_opening_range_data(current_date, params.or_minutes)
        if not or_data or or_data['bars_counted'] == 0:
            return signal

        # Apply advanced filters
        if not self._passes_advanced_filters(bar_data, or_data, params):
            return signal

        # Generate entry signals
        current_price = bar_data['close']
        or_high = or_data['high']
        or_low = or_data['low']
        or_range = or_high - or_low

        # Dynamic target calculation based on regime and volatility
        target_points = self._calculate_dynamic_target(bar_data, params, or_range)

        # Long breakout signal
        if current_price > or_high and self._validate_long_entry(bar_data, or_data, params):
            signal = {
                'signal': 'BUY',
                'entry_time': timestamp,
                'entry_price': current_price,
                'stop_loss': current_price - params.stop_points,
                'take_profit': current_price + target_points,
                'contracts': 1,
                'regime': self.current_regime.value,
                'or_range': or_range,
                'target_points': target_points,
                'rr_ratio': target_points / params.stop_points,
                'half_target': current_price + (target_points * 0.5),
                'reason': f'{self.current_regime.value} Long breakout: {or_high:.2f}, Target: {target_points:.1f}pts, RR: {target_points/params.stop_points:.1f}',
                'direction': 'LONG',
                'momentum': self._calculate_momentum(bar_data),
                'volatility': self._calculate_volatility()
            }
            self.daily_trade_count[current_date] += 1

        # Short breakdown signal
        elif current_price < or_low and self._validate_short_entry(bar_data, or_data, params):
            signal = {
                'signal': 'SELL',
                'entry_time': timestamp,
                'entry_price': current_price,
                'stop_loss': current_price + params.stop_points,
                'take_profit': current_price - target_points,
                'contracts': 1,
                'regime': self.current_regime.value,
                'or_range': or_range,
                'target_points': target_points,
                'rr_ratio': target_points / params.stop_points,
                'half_target': current_price - (target_points * 0.5),
                'reason': f'{self.current_regime.value} Short breakdown: {or_low:.2f}, Target: {target_points:.1f}pts, RR: {target_points/params.stop_points:.1f}',
                'direction': 'SHORT',
                'momentum': self._calculate_momentum(bar_data),
                'volatility': self._calculate_volatility()
            }
            self.daily_trade_count[current_date] += 1

        return signal

    def _detect_market_regime(self, bar_data: Dict[str, Any]) -> None:
        """Detect current market regime based on price action and volatility"""
        if len(self.price_history) < 20:  # Need sufficient data
            return

        # Calculate key metrics
        volatility = self._calculate_volatility()
        momentum = self._calculate_momentum(bar_data)
        trend_strength = self._calculate_trend_strength()

        # Recent price action
        recent_bars = self.price_history[-10:]
        recent_highs = [bar['high'] for bar in recent_bars]
        recent_lows = [bar['low'] for bar in recent_bars]

        current_high = bar_data['high']
        current_low = bar_data['low']

        # Regime classification logic
        if volatility > 50 and momentum > 0.8:
            new_regime = MarketRegime.HIGH_VOLATILITY
        elif volatility < 15:
            new_regime = MarketRegime.LOW_VOLATILITY
        elif trend_strength > 0.7 and momentum > 0.6:
            if current_high > max(recent_highs):
                new_regime = MarketRegime.TRENDING_UP
            elif current_low < min(recent_lows):
                new_regime = MarketRegime.TRENDING_DOWN
            else:
                new_regime = MarketRegime.TRENDING_UP if momentum > 0 else MarketRegime.TRENDING_DOWN
        elif volatility > 25 and abs(momentum) > 0.5:
            new_regime = MarketRegime.BREAKOUT_SETUP
        else:
            new_regime = MarketRegime.RANGING

        # Update regime if changed
        if new_regime != self.current_regime:
            print(f"Regime change: {self.current_regime.value} -> {new_regime.value} (Vol: {volatility:.1f}, Mom: {momentum:.2f})")
            self.current_regime = new_regime
            self.current_params = self.regime_params[new_regime]

        # Track regime history
        self.regime_history.append({
            'timestamp': bar_data['timestamp'],
            'regime': new_regime,
            'volatility': volatility,
            'momentum': momentum,
            'trend_strength': trend_strength
        })

        # Keep only recent history
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]

    def _calculate_dynamic_target(self, bar_data: Dict[str, Any], params: RegimeParameters, or_range: float) -> float:
        """Calculate dynamic target based on regime, volatility, and market conditions"""
        base_target = params.stop_points * params.rr_ratio

        # Adjust based on volatility
        volatility = self._calculate_volatility()
        if volatility > 40:
            volatility_multiplier = 1.5  # Bigger targets in high vol
        elif volatility < 15:
            volatility_multiplier = 0.8  # Smaller targets in low vol
        else:
            volatility_multiplier = 1.0

        # Adjust based on opening range
        if or_range > 30:
            or_multiplier = 1.3  # Bigger targets with wide OR
        elif or_range < 15:
            or_multiplier = 0.9  # Smaller targets with narrow OR
        else:
            or_multiplier = 1.0

        # Adjust based on time of day (bigger targets in morning)
        hour = bar_data['timestamp'].hour
        if hour <= 11:
            time_multiplier = 1.2  # Morning ORB prime time
        elif hour >= 15:
            time_multiplier = 0.8  # Late day reduced targets
        else:
            time_multiplier = 1.0

        # Calculate final target
        dynamic_target = base_target * volatility_multiplier * or_multiplier * time_multiplier

        # Ensure minimum target for profitability
        min_target = 15.0  # Minimum 15 points target
        max_target = 100.0  # Maximum 100 points target

        return max(min_target, min(dynamic_target, max_target))

    def _passes_advanced_filters(self, bar_data: Dict[str, Any], or_data: Dict, params: RegimeParameters) -> bool:
        """Apply advanced entry filters"""
        # Momentum filter
        momentum = self._calculate_momentum(bar_data)
        if abs(momentum) < params.momentum_threshold:
            return False

        # Volatility filter
        volatility = self._calculate_volatility()
        if volatility < params.volatility_threshold:
            return False

        # Opening range filter - must be significant
        or_range = or_data['high'] - or_data['low']
        if or_range < 10:  # Minimum 10 points OR
            return False

        # Volume confirmation (if available)
        if 'volume' in bar_data:
            avg_volume = self._calculate_average_volume()
            if avg_volume > 0 and bar_data['volume'] < avg_volume * 0.8:
                return False  # Need above-average volume

        # Time-based filters
        hour = bar_data['timestamp'].hour
        minute = bar_data['timestamp'].minute

        # Skip lunch hour low activity (12:00-13:00)
        if hour == 12:
            return False

        # Favor morning sessions for breakouts
        if self.current_regime in [MarketRegime.BREAKOUT_SETUP, MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            if hour > 14:  # After 2 PM, be more selective
                return abs(momentum) > 0.8  # Need very strong momentum

        return True

    def _validate_long_entry(self, bar_data: Dict[str, Any], or_data: Dict, params: RegimeParameters) -> bool:
        """Additional validation for long entries"""
        current_price = bar_data['close']
        or_high = or_data['high']

        # Price must be clearly above OR high
        if current_price <= or_high + 2.0:  # Need at least 2 point breakout
            return False

        # Check momentum direction
        momentum = self._calculate_momentum(bar_data)
        if momentum < 0.3:  # Need positive momentum for longs
            return False

        # Recent price action should support upward move
        if len(self.price_history) >= 3:
            recent_closes = [bar['close'] for bar in self.price_history[-3:]]
            if not all(recent_closes[i] <= recent_closes[i+1] for i in range(len(recent_closes)-1)):
                return abs(momentum) > 0.7  # Need strong momentum if not trending up

        return True

    def _validate_short_entry(self, bar_data: Dict[str, Any], or_data: Dict, params: RegimeParameters) -> bool:
        """Additional validation for short entries"""
        current_price = bar_data['close']
        or_low = or_data['low']

        # Price must be clearly below OR low
        if current_price >= or_low - 2.0:  # Need at least 2 point breakdown
            return False

        # Check momentum direction
        momentum = self._calculate_momentum(bar_data)
        if momentum > -0.3:  # Need negative momentum for shorts
            return False

        # Recent price action should support downward move
        if len(self.price_history) >= 3:
            recent_closes = [bar['close'] for bar in self.price_history[-3:]]
            if not all(recent_closes[i] >= recent_closes[i+1] for i in range(len(recent_closes)-1)):
                return abs(momentum) > 0.7  # Need strong momentum if not trending down

        return True

    def _calculate_momentum(self, bar_data: Dict[str, Any]) -> float:
        """Calculate price momentum"""
        if len(self.price_history) < 5:
            return 0.0

        current_price = bar_data['close']
        past_prices = [bar['close'] for bar in self.price_history[-5:]]
        past_price = past_prices[0]

        if past_price == 0:
            return 0.0

        # Calculate rate of change over 5 bars
        momentum = (current_price - past_price) / past_price
        return momentum

    def _calculate_volatility(self) -> float:
        """Calculate recent volatility"""
        if len(self.price_history) < 10:
            return 20.0  # Default volatility

        recent_bars = self.price_history[-10:]
        ranges = [bar['high'] - bar['low'] for bar in recent_bars]
        avg_range = sum(ranges) / len(ranges)

        return avg_range

    def _calculate_trend_strength(self) -> float:
        """Calculate trend strength"""
        if len(self.price_history) < 10:
            return 0.0

        recent_bars = self.price_history[-10:]
        closes = [bar['close'] for bar in recent_bars]

        # Linear regression slope
        x = list(range(len(closes)))
        n = len(x)

        if n < 2:
            return 0.0

        x_mean = sum(x) / n
        y_mean = sum(closes) / n

        numerator = sum((x[i] - x_mean) * (closes[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        # Normalize slope to 0-1 scale
        return min(1.0, abs(slope) / 10.0)

    def _calculate_average_volume(self) -> float:
        """Calculate average volume"""
        if len(self.volume_history) < 10:
            return 0.0

        recent_volumes = self.volume_history[-10:]
        return sum(recent_volumes) / len(recent_volumes)

    def _should_stop_trading_today(self) -> bool:
        """Check if we should stop trading today due to losses"""
        current_date = self.current_date

        # Stop after 2 consecutive losses
        if self.consecutive_losses >= 2:
            return True

        # Stop if daily loss exceeds threshold
        daily_pnl = self.daily_pnl.get(current_date, 0)
        if daily_pnl < -30:  # Stop if down more than 30 points
            return True

        # Stop if target reached (lock in profits)
        if daily_pnl >= self.target_points_per_day:
            return True

        return False

    def _is_trading_time(self, timestamp: datetime) -> bool:
        """Check if it's valid trading time based on regime"""
        current_time = timestamp.time()

        # Calculate OR end time
        market_open = time(9, 30)
        or_end_minutes = self.current_params.or_minutes
        or_end_time = (datetime.combine(timestamp.date(), market_open) +
                      timedelta(minutes=or_end_minutes)).time()

        # Trading starts after OR period
        trading_start = or_end_time

        # Trading ends based on regime
        if self.current_regime == MarketRegime.LOW_VOLATILITY:
            trading_end = time(14, 0)  # End earlier in low vol
        else:
            trading_end = time(15, 45)  # Regular close

        return trading_start <= current_time <= trading_end

    def _get_opening_range_data(self, current_date, or_minutes: int) -> Optional[Dict]:
        """Get opening range data for specified period"""
        if current_date not in self.daily_data:
            return None

        # Calculate OR period end
        market_open = time(9, 30)
        or_end = (datetime.combine(current_date, market_open) +
                 timedelta(minutes=or_minutes)).time()

        or_data = self.daily_data[current_date].get(f'or_{or_minutes}', {})

        return or_data if or_data.get('bars_counted', 0) > 0 else None

    def _has_sufficient_data(self) -> bool:
        """Check if we have sufficient data for analysis"""
        return (len(self.price_history) >= 10 and
                len(self.regime_history) >= 5)

    def _update_market_data(self, bar_data: Dict[str, Any]) -> None:
        """Update market data tracking"""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()
        current_time = timestamp.time()

        # Update price history
        self.price_history.append({
            'timestamp': timestamp,
            'high': bar_data['high'],
            'low': bar_data['low'],
            'close': bar_data['close'],
            'open': bar_data.get('open', bar_data['close'])
        })

        # Keep only recent history
        if len(self.price_history) > 50:
            self.price_history = self.price_history[-50:]

        # Update volume history
        if 'volume' in bar_data:
            self.volume_history.append(bar_data['volume'])
            if len(self.volume_history) > 50:
                self.volume_history = self.volume_history[-50:]

        # Update opening ranges for different periods
        self._update_opening_ranges(bar_data)

    def _update_opening_ranges(self, bar_data: Dict[str, Any]) -> None:
        """Update opening ranges for all periods"""
        timestamp = bar_data['timestamp']
        current_date = timestamp.date()
        current_time = timestamp.time()
        market_open = time(9, 30)

        if current_date not in self.daily_data:
            self.daily_data[current_date] = {}

        # Update OR for different periods
        or_periods = [15, 30, 45, 60, 90]

        for period in or_periods:
            or_end = (datetime.combine(current_date, market_open) +
                     timedelta(minutes=period)).time()

            if market_open <= current_time <= or_end:
                or_key = f'or_{period}'

                if or_key not in self.daily_data[current_date]:
                    self.daily_data[current_date][or_key] = {
                        'high': bar_data['high'],
                        'low': bar_data['low'],
                        'bars_counted': 1
                    }
                else:
                    or_data = self.daily_data[current_date][or_key]
                    or_data['high'] = max(or_data['high'], bar_data['high'])
                    or_data['low'] = min(or_data['low'], bar_data['low'])
                    or_data['bars_counted'] += 1

    def _manage_existing_positions(self, bar_data: Dict[str, Any], portfolio: 'Portfolio') -> None:
        """Advanced position management"""
        current_price = bar_data['close']

        for pos_id, position in getattr(portfolio, 'positions', {}).items():
            # Implement trailing stops, half-size booking, etc.
            # This would integrate with the portfolio management system
            pass

    def _reset_daily_data(self, current_date) -> None:
        """Reset daily data structures"""
        if current_date not in self.daily_data:
            self.daily_data[current_date] = {}
        if current_date not in self.daily_trade_count:
            self.daily_trade_count[current_date] = 0
        if current_date not in self.daily_pnl:
            self.daily_pnl[current_date] = 0

        # Reset daily flags
        self.stop_trading_today = False
        self.consecutive_losses = 0

    def update_trade_result(self, pnl_points: float) -> None:
        """Update trade result for loss tracking"""
        current_date = self.current_date
        if current_date:
            self.daily_pnl[current_date] += pnl_points

            if pnl_points < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        total_days = len(self.daily_pnl)
        profitable_days = sum(1 for pnl in self.daily_pnl.values() if pnl > 0)
        total_pnl = sum(self.daily_pnl.values())

        avg_daily_pnl = total_pnl / total_days if total_days > 0 else 0
        win_rate_daily = (profitable_days / total_days * 100) if total_days > 0 else 0

        return {
            'total_days': total_days,
            'profitable_days': profitable_days,
            'total_pnl_points': total_pnl,
            'avg_daily_pnl': avg_daily_pnl,
            'win_rate_daily': win_rate_daily,
            'target_achievement_rate': sum(1 for pnl in self.daily_pnl.values() if pnl >= self.target_points_per_day) / total_days * 100 if total_days > 0 else 0,
            'current_regime': self.current_regime.value,
            'regime_distribution': self._get_regime_distribution()
        }

    def _get_regime_distribution(self) -> Dict[str, float]:
        """Get distribution of time spent in each regime"""
        if not self.regime_history:
            return {}

        regime_counts = {}
        for entry in self.regime_history:
            regime = entry['regime'].value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        total = len(self.regime_history)
        return {regime: count/total*100 for regime, count in regime_counts.items()}