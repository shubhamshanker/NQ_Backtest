"""
Advanced Trade Management System
===============================
Sophisticated position management for maximizing the 15-30+ points/day target:
- Half-size booking at profit targets
- Dynamic trailing stops
- Breakeven moves
- Profit protection
- Risk escalation management
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime, time
from dataclasses import dataclass, field
from enum import Enum
import math

if TYPE_CHECKING:
    from portfolio import Portfolio

class TradeStatus(Enum):
    ACTIVE = "active"
    HALF_CLOSED = "half_closed"
    BREAKEVEN = "breakeven"
    TRAILING = "trailing"
    CLOSED = "closed"

class ExitReason(Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    HALF_PROFIT = "half_profit"
    TRAILING_STOP = "trailing_stop"
    BREAKEVEN_STOP = "breakeven_stop"
    END_OF_DAY = "end_of_day"
    RISK_MANAGEMENT = "risk_management"

@dataclass
class TradeManagementState:
    """State tracking for active trade management"""
    trade_id: str
    entry_time: datetime
    entry_price: float
    direction: str  # LONG or SHORT
    stop_loss: float
    take_profit: float
    contracts: int = 1

    # Management levels
    half_profit_target: float = 0.0
    breakeven_trigger: float = 0.0
    trailing_trigger: float = 0.0

    # State tracking
    status: TradeStatus = TradeStatus.ACTIVE
    half_closed: bool = False
    moved_to_breakeven: bool = False
    trailing_active: bool = False
    highest_favorable: float = 0.0
    lowest_favorable: float = 0.0

    # Performance tracking
    current_profit: float = 0.0
    max_profit_seen: float = 0.0
    max_adverse_move: float = 0.0

    # Management parameters
    half_profit_pct: float = 0.5  # Take half at 50% of target
    breakeven_pct: float = 0.3    # Move to BE at 30% of target
    trailing_pct: float = 0.6     # Start trailing at 60% of target
    trailing_buffer: float = 5.0  # Trail by 5 points

    # Risk management
    max_loss_points: float = 0.0
    profit_protection_level: float = 0.0

class AdvancedTradeManager:
    """Advanced trade management system for ORB strategies"""

    def __init__(self, target_daily_points: float = 20.0):
        """
        Initialize trade manager

        Args:
            target_daily_points: Daily point target for aggressive profit taking
        """
        self.target_daily_points = target_daily_points
        self.active_trades = {}
        self.completed_trades = []
        self.daily_pnl = {}
        self.current_date = None

        # Management settings optimized for high daily targets
        self.management_config = {
            'aggressive_half_booking': True,   # Take profits quickly
            'breakeven_protection': True,      # Protect capital aggressively
            'dynamic_trailing': True,          # Adaptive trailing based on volatility
            'end_of_day_exit': True,          # Close all positions by EOD
            'profit_protection': True,         # Protect unrealized profits
            'daily_target_protection': True    # Lock profits when daily target hit
        }

    def add_trade(self, trade_signal: Dict[str, Any]) -> str:
        """
        Add new trade for management

        Args:
            trade_signal: Trade signal dictionary

        Returns:
            Trade ID for tracking
        """
        trade_id = f"{trade_signal['entry_time'].strftime('%Y%m%d_%H%M%S')}_{trade_signal['direction']}"

        # Calculate management levels
        entry_price = trade_signal['entry_price']
        direction = trade_signal['direction']
        stop_loss = trade_signal['stop_loss']
        take_profit = trade_signal['take_profit']

        target_points = abs(take_profit - entry_price)
        stop_points = abs(entry_price - stop_loss)

        # Calculate management triggers
        if direction == 'LONG':
            half_profit_target = entry_price + (target_points * 0.5)
            breakeven_trigger = entry_price + (target_points * 0.3)
            trailing_trigger = entry_price + (target_points * 0.6)
        else:  # SHORT
            half_profit_target = entry_price - (target_points * 0.5)
            breakeven_trigger = entry_price - (target_points * 0.3)
            trailing_trigger = entry_price - (target_points * 0.6)

        # Create trade management state
        trade_state = TradeManagementState(
            trade_id=trade_id,
            entry_time=trade_signal['entry_time'],
            entry_price=entry_price,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            half_profit_target=half_profit_target,
            breakeven_trigger=breakeven_trigger,
            trailing_trigger=trailing_trigger,
            max_loss_points=stop_points,
            highest_favorable=entry_price if direction == 'LONG' else entry_price,
            lowest_favorable=entry_price if direction == 'SHORT' else entry_price
        )

        self.active_trades[trade_id] = trade_state

        print(f"📈 Trade Added: {trade_id} - {direction} @ {entry_price:.2f}")
        print(f"   Half: {half_profit_target:.2f}, BE: {breakeven_trigger:.2f}, Trail: {trailing_trigger:.2f}")

        return trade_id

    def update_trade(self, trade_id: str, current_bar: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Update trade management based on current bar

        Args:
            trade_id: Trade to update
            current_bar: Current price bar data

        Returns:
            List of management actions (exits, modifications)
        """
        if trade_id not in self.active_trades:
            return []

        trade = self.active_trades[trade_id]
        current_price = current_bar['close']
        current_high = current_bar['high']
        current_low = current_bar['low']
        timestamp = current_bar['timestamp']

        actions = []

        # Update profit tracking
        if trade.direction == 'LONG':
            trade.current_profit = current_price - trade.entry_price
            trade.highest_favorable = max(trade.highest_favorable, current_high)
            trade.max_adverse_move = min(trade.max_adverse_move, current_low - trade.entry_price)
        else:  # SHORT
            trade.current_profit = trade.entry_price - current_price
            trade.lowest_favorable = min(trade.lowest_favorable, current_low)
            trade.max_adverse_move = max(trade.max_adverse_move, current_high - trade.entry_price)

        trade.max_profit_seen = max(trade.max_profit_seen, trade.current_profit)

        # Check for management actions

        # 1. Stop Loss Check
        if self._check_stop_loss_hit(trade, current_low, current_high):
            actions.append(self._create_exit_action(trade, trade.stop_loss, ExitReason.STOP_LOSS, timestamp))

        # 2. Take Profit Check
        elif self._check_take_profit_hit(trade, current_low, current_high):
            actions.append(self._create_exit_action(trade, trade.take_profit, ExitReason.TAKE_PROFIT, timestamp))

        # 3. Half-Size Booking
        elif self._should_book_half_size(trade, current_price) and not trade.half_closed:
            actions.append(self._create_half_exit_action(trade, timestamp))

        # 4. Breakeven Move
        elif self._should_move_to_breakeven(trade, current_price) and not trade.moved_to_breakeven:
            actions.append(self._create_breakeven_action(trade, timestamp))

        # 5. Trailing Stop Activation
        elif self._should_activate_trailing(trade, current_price):
            actions.append(self._create_trailing_action(trade, current_price, timestamp))

        # 6. End of Day Exit
        elif self._should_exit_end_of_day(timestamp):
            actions.append(self._create_exit_action(trade, current_price, ExitReason.END_OF_DAY, timestamp))

        # 7. Daily Target Protection
        elif self._should_protect_daily_target(timestamp.date()):
            actions.append(self._create_exit_action(trade, current_price, ExitReason.RISK_MANAGEMENT, timestamp))

        return actions

    def _check_stop_loss_hit(self, trade: TradeManagementState, low: float, high: float) -> bool:
        """Check if stop loss was hit"""
        if trade.direction == 'LONG':
            return low <= trade.stop_loss
        else:  # SHORT
            return high >= trade.stop_loss

    def _check_take_profit_hit(self, trade: TradeManagementState, low: float, high: float) -> bool:
        """Check if take profit was hit"""
        if trade.direction == 'LONG':
            return high >= trade.take_profit
        else:  # SHORT
            return low <= trade.take_profit

    def _should_book_half_size(self, trade: TradeManagementState, current_price: float) -> bool:
        """Check if should book half size"""
        if not self.management_config['aggressive_half_booking'] or trade.half_closed:
            return False

        if trade.direction == 'LONG':
            return current_price >= trade.half_profit_target
        else:  # SHORT
            return current_price <= trade.half_profit_target

    def _should_move_to_breakeven(self, trade: TradeManagementState, current_price: float) -> bool:
        """Check if should move stop to breakeven"""
        if not self.management_config['breakeven_protection'] or trade.moved_to_breakeven:
            return False

        if trade.direction == 'LONG':
            return current_price >= trade.breakeven_trigger
        else:  # SHORT
            return current_price <= trade.breakeven_trigger

    def _should_activate_trailing(self, trade: TradeManagementState, current_price: float) -> bool:
        """Check if should activate trailing stop"""
        if not self.management_config['dynamic_trailing']:
            return False

        # Already trailing - update trailing stop
        if trade.trailing_active:
            return self._update_trailing_stop(trade, current_price)

        # Check if should start trailing
        if trade.direction == 'LONG':
            return current_price >= trade.trailing_trigger
        else:  # SHORT
            return current_price <= trade.trailing_trigger

    def _update_trailing_stop(self, trade: TradeManagementState, current_price: float) -> bool:
        """Update trailing stop level"""
        if trade.direction == 'LONG':
            # Trail below recent high
            new_trailing_stop = trade.highest_favorable - trade.trailing_buffer
            if new_trailing_stop > trade.stop_loss:
                trade.stop_loss = new_trailing_stop
                return True
        else:  # SHORT
            # Trail above recent low
            new_trailing_stop = trade.lowest_favorable + trade.trailing_buffer
            if new_trailing_stop < trade.stop_loss:
                trade.stop_loss = new_trailing_stop
                return True

        return False

    def _should_exit_end_of_day(self, timestamp: datetime) -> bool:
        """Check if should exit for end of day"""
        if not self.management_config['end_of_day_exit']:
            return False

        exit_time = time(15, 50)  # Exit 10 minutes before close
        return timestamp.time() >= exit_time

    def _should_protect_daily_target(self, current_date) -> bool:
        """Check if should protect daily target achievement"""
        if not self.management_config['daily_target_protection']:
            return False

        daily_pnl = self.daily_pnl.get(current_date, 0)
        return daily_pnl >= self.target_daily_points

    def _create_exit_action(self, trade: TradeManagementState, exit_price: float,
                           reason: ExitReason, timestamp: datetime) -> Dict[str, Any]:
        """Create exit action"""
        profit_points = trade.current_profit
        profit_dollars = profit_points * 20.0  # Point value

        action = {
            'action_type': 'EXIT',
            'trade_id': trade.trade_id,
            'exit_price': exit_price,
            'exit_time': timestamp,
            'reason': reason.value,
            'profit_points': profit_points,
            'profit_dollars': profit_dollars,
            'contracts': trade.contracts,
            'duration_minutes': (timestamp - trade.entry_time).total_seconds() / 60
        }

        # Move to completed trades
        self.completed_trades.append({
            'trade_id': trade.trade_id,
            'entry_time': trade.entry_time,
            'exit_time': timestamp,
            'entry_price': trade.entry_price,
            'exit_price': exit_price,
            'direction': trade.direction,
            'profit_points': profit_points,
            'profit_dollars': profit_dollars,
            'exit_reason': reason.value,
            'max_profit_seen': trade.max_profit_seen,
            'max_adverse_move': trade.max_adverse_move
        })

        # Update daily P&L
        trade_date = timestamp.date()
        if trade_date not in self.daily_pnl:
            self.daily_pnl[trade_date] = 0
        self.daily_pnl[trade_date] += profit_points

        # Remove from active trades
        del self.active_trades[trade.trade_id]

        print(f"🔚 Trade Closed: {trade.trade_id} - {reason.value}")
        print(f"   Profit: {profit_points:+.1f} points (${profit_dollars:+.2f})")

        return action

    def _create_half_exit_action(self, trade: TradeManagementState, timestamp: datetime) -> Dict[str, Any]:
        """Create half-size exit action"""
        exit_price = trade.half_profit_target
        profit_points = abs(exit_price - trade.entry_price)
        profit_dollars = profit_points * 20.0 * 0.5  # Half position

        action = {
            'action_type': 'HALF_EXIT',
            'trade_id': trade.trade_id,
            'exit_price': exit_price,
            'exit_time': timestamp,
            'reason': ExitReason.HALF_PROFIT.value,
            'profit_points': profit_points,
            'profit_dollars': profit_dollars,
            'contracts': 0.5  # Half position
        }

        # Update trade state
        trade.half_closed = True
        trade.contracts = 0.5
        trade.status = TradeStatus.HALF_CLOSED

        # Update daily P&L for half position
        trade_date = timestamp.date()
        if trade_date not in self.daily_pnl:
            self.daily_pnl[trade_date] = 0
        self.daily_pnl[trade_date] += profit_points * 0.5

        print(f"📊 Half Position Closed: {trade.trade_id}")
        print(f"   Half Profit: {profit_points:.1f} points (${profit_dollars:.2f})")

        return action

    def _create_breakeven_action(self, trade: TradeManagementState, timestamp: datetime) -> Dict[str, Any]:
        """Create breakeven move action"""
        # Move stop to breakeven (entry price + small buffer)
        buffer = 2.0  # 2 point buffer above/below entry

        if trade.direction == 'LONG':
            new_stop = trade.entry_price + buffer
        else:  # SHORT
            new_stop = trade.entry_price - buffer

        old_stop = trade.stop_loss
        trade.stop_loss = new_stop
        trade.moved_to_breakeven = True
        trade.status = TradeStatus.BREAKEVEN

        action = {
            'action_type': 'MODIFY_STOP',
            'trade_id': trade.trade_id,
            'old_stop': old_stop,
            'new_stop': new_stop,
            'timestamp': timestamp,
            'reason': 'breakeven_protection'
        }

        print(f"🛡️ Moved to Breakeven: {trade.trade_id}")
        print(f"   Stop: {old_stop:.2f} -> {new_stop:.2f}")

        return action

    def _create_trailing_action(self, trade: TradeManagementState, current_price: float,
                               timestamp: datetime) -> Dict[str, Any]:
        """Create trailing stop action"""
        if not trade.trailing_active:
            # Activate trailing
            trade.trailing_active = True
            trade.status = TradeStatus.TRAILING

            action = {
                'action_type': 'ACTIVATE_TRAILING',
                'trade_id': trade.trade_id,
                'current_price': current_price,
                'timestamp': timestamp,
                'trailing_buffer': trade.trailing_buffer
            }

            print(f"🎯 Trailing Activated: {trade.trade_id}")
            print(f"   Buffer: {trade.trailing_buffer} points")

        else:
            # Update trailing stop
            old_stop = trade.stop_loss

            action = {
                'action_type': 'UPDATE_TRAILING',
                'trade_id': trade.trade_id,
                'old_stop': old_stop,
                'new_stop': trade.stop_loss,
                'timestamp': timestamp
            }

        return action

    def get_daily_performance(self, date) -> Dict[str, Any]:
        """Get daily performance metrics"""
        daily_pnl = self.daily_pnl.get(date, 0)

        # Get trades for this date
        daily_trades = [t for t in self.completed_trades
                       if t['exit_time'].date() == date]

        total_trades = len(daily_trades)
        winning_trades = len([t for t in daily_trades if t['profit_points'] > 0])

        return {
            'date': date,
            'total_pnl_points': daily_pnl,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'target_achieved': daily_pnl >= self.target_daily_points,
            'avg_profit_per_trade': daily_pnl / total_trades if total_trades > 0 else 0
        }

    def get_management_statistics(self) -> Dict[str, Any]:
        """Get trade management performance statistics"""
        if not self.completed_trades:
            return {}

        total_trades = len(self.completed_trades)

        # Exit reason analysis
        exit_reasons = {}
        for trade in self.completed_trades:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        # Half-size booking analysis
        half_booked_trades = [t for t in self.completed_trades if 'half_profit' in t.get('exit_reason', '')]

        # Profit protection analysis
        trades_with_profit = [t for t in self.completed_trades if t['profit_points'] > 0]
        avg_max_profit = sum(t['max_profit_seen'] for t in trades_with_profit) / len(trades_with_profit) if trades_with_profit else 0
        avg_realized_profit = sum(t['profit_points'] for t in trades_with_profit) / len(trades_with_profit) if trades_with_profit else 0

        profit_capture_ratio = avg_realized_profit / avg_max_profit if avg_max_profit > 0 else 0

        return {
            'total_managed_trades': total_trades,
            'exit_reason_breakdown': exit_reasons,
            'half_booking_rate': len(half_booked_trades) / total_trades * 100,
            'profit_capture_ratio': profit_capture_ratio * 100,
            'avg_max_profit_seen': avg_max_profit,
            'avg_realized_profit': avg_realized_profit,
            'breakeven_protection_rate': len([t for t in self.completed_trades if 'breakeven' in t.get('exit_reason', '')]) / total_trades * 100
        }