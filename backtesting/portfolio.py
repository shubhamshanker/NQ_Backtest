"""
Portfolio Module
================
Central state management for backtesting with position tracking,
risk management, and performance calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from pandas import DataFrame
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Position:
    """Represents an open trading position."""
    id: str
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_time: datetime
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float = 0.0
    max_favorable: float = 0.0
    max_adverse: float = 0.0

@dataclass
class Trade:
    """Represents a completed trade for analysis."""
    id: str
    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    duration_minutes: float
    exit_reason: str
    max_favorable: float = 0.0
    max_adverse: float = 0.0

class Portfolio:
    """
    Central portfolio management with position tracking and performance analysis.
    Single source of truth for all backtesting state.
    """

    def __init__(self, initial_capital: float, point_value: float = 20.0,
                 commission_per_trade: float = 2.50, max_positions: int = 1):
        """
        Initialize portfolio with capital and trading parameters.

        Args:
            initial_capital: Starting cash
            point_value: Dollar value per point (e.g., $20 for NQ)
            commission_per_trade: Commission per trade (round-trip)
            max_positions: Maximum concurrent positions
        """
        # Capital management
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.point_value = float(point_value)
        self.commission_per_trade = float(commission_per_trade)
        self.max_positions = max_positions

        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.position_counter = 0

        # Trade history
        self.closed_trades: List[Trade] = []

        # Performance tracking
        self.equity_curve: List[Dict[str, Any]] = []
        self.current_timestamp = None

        # Risk management
        self.max_risk_per_trade = 0.1  # Maximum 10% risk per trade

    def update_positions(self, bar_data: Dict[str, Any]) -> List[str]:
        """
        Update all open positions and close any that hit stop/target.

        Args:
            bar_data: Current bar data

        Returns:
            List of position IDs that were closed
        """
        self.current_timestamp = bar_data['timestamp']
        closed_position_ids = []

        # Update each open position
        for position_id, position in list(self.positions.items()):
            # Update unrealized P&L and MAE/MFE
            self._update_position_metrics(position, bar_data)

            # Check exit conditions
            exit_reason = self._check_exit_conditions(position, bar_data)

            if exit_reason:
                # Close position
                closed_trade = self._close_position(position, bar_data, exit_reason)
                self.closed_trades.append(closed_trade)
                closed_position_ids.append(position_id)
                del self.positions[position_id]

        return closed_position_ids

    def execute_trade(self, signal_dict: Dict[str, Any], symbol: str = "NQ") -> Optional[str]:
        """
        Execute trade based on strategy signal with position sizing.

        Args:
            signal_dict: Signal from strategy
            symbol: Trading symbol

        Returns:
            Position ID if trade executed, None otherwise
        """
        if signal_dict['signal'] == 'HOLD':
            return None

        # Check position limits
        if len(self.positions) >= self.max_positions:
            return None

        # Calculate position size based on risk management
        position_size = self._calculate_position_size(signal_dict)
        if position_size <= 0:
            return None

        # Create new position
        self.position_counter += 1
        position_id = f"POS_{self.position_counter:04d}"

        side = 'LONG' if signal_dict['signal'] == 'BUY' else 'SHORT'
        entry_price = signal_dict.get('entry_price', 0.0)  # Should be current price

        # If entry_price not provided, we need current market price
        if entry_price == 0.0:
            # This should be handled by the strategy, but as fallback
            raise ValueError("Entry price must be provided in signal")

        position = Position(
            id=position_id,
            symbol=symbol,
            side=side,
            entry_time=signal_dict['entry_time'],
            entry_price=entry_price,
            quantity=position_size,
            stop_loss=signal_dict['stop_loss'],
            take_profit=signal_dict['take_profit']
        )

        self.positions[position_id] = position

        # Deduct commission from cash
        self.cash -= self.commission_per_trade

        return position_id

    def _calculate_position_size(self, signal_dict: Dict[str, Any]) -> float:
        """
        Calculate position size based on risk management or fixed contracts.
        """
        # Check if fixed contract size is specified
        if 'contracts' in signal_dict and signal_dict['contracts'] > 0:
            return float(signal_dict['contracts'])

        # Otherwise use risk-based sizing
        entry_price = signal_dict.get('entry_price', 0.0)
        stop_loss = signal_dict['stop_loss']
        risk_per_trade = signal_dict.get('risk_per_trade', 0.02)  # Default 2%

        if entry_price <= 0 or stop_loss <= 0:
            return 0.0

        # Calculate risk amount in dollars
        risk_amount = self.get_total_value() * risk_per_trade
        risk_amount = min(risk_amount, self.get_total_value() * self.max_risk_per_trade)

        # Calculate stop distance in points
        stop_distance_points = abs(entry_price - stop_loss)

        if stop_distance_points <= 0:
            return 0.0

        # Calculate position size (number of contracts)
        # Risk Amount = Position Size * Stop Distance * Point Value
        position_size = risk_amount / (stop_distance_points * self.point_value)

        # Ensure we have enough cash (simplified check)
        # In futures, margin requirements would be checked here
        min_margin = position_size * entry_price * 0.1  # Assume 10% margin
        if min_margin > self.cash:
            position_size = self.cash / (entry_price * 0.1)

        return max(0.0, round(position_size, 2))

    def _update_position_metrics(self, position: Position, bar_data: Dict[str, Any]) -> None:
        """Update position metrics including unrealized P&L and MAE/MFE."""
        current_price = bar_data['close']

        if position.side == 'LONG':
            # Long position P&L
            points_pnl = current_price - position.entry_price
            unrealized_pnl = points_pnl * position.quantity * self.point_value

            # Track maximum favorable/adverse excursion
            favorable_points = bar_data['high'] - position.entry_price
            adverse_points = bar_data['low'] - position.entry_price

            position.max_favorable = max(position.max_favorable, favorable_points)
            position.max_adverse = min(position.max_adverse, adverse_points)

        else:  # SHORT
            # Short position P&L
            points_pnl = position.entry_price - current_price
            unrealized_pnl = points_pnl * position.quantity * self.point_value

            # Track maximum favorable/adverse excursion
            favorable_points = position.entry_price - bar_data['low']
            adverse_points = position.entry_price - bar_data['high']

            position.max_favorable = max(position.max_favorable, favorable_points)
            position.max_adverse = min(position.max_adverse, adverse_points)

        position.unrealized_pnl = unrealized_pnl

    def _check_exit_conditions(self, position: Position, bar_data: Dict[str, Any]) -> Optional[str]:
        """Check if position should be closed based on stop/target levels."""

        # Check for end-of-day forced exit (intraday only)
        current_time = bar_data['timestamp'].time()
        market_close = pd.to_datetime("16:00", format="%H:%M").time()

        if current_time >= market_close:
            return 'End of Day'

        if position.side == 'LONG':
            # Long position exits
            if bar_data['low'] <= position.stop_loss:
                return 'Stop Loss'
            elif bar_data['high'] >= position.take_profit:
                return 'Take Profit'

        else:  # SHORT
            # Short position exits
            if bar_data['high'] >= position.stop_loss:
                return 'Stop Loss'
            elif bar_data['low'] <= position.take_profit:
                return 'Take Profit'

        return None

    def _close_position(self, position: Position, bar_data: Dict[str, Any], exit_reason: str) -> Trade:
        """Close position and create trade record."""

        # Determine exit price based on exit reason
        if exit_reason == 'Stop Loss':
            exit_price = position.stop_loss
        elif exit_reason == 'Take Profit':
            exit_price = position.take_profit
        else:
            exit_price = bar_data['close']  # Market close

        # Calculate final P&L
        if position.side == 'LONG':
            points_pnl = exit_price - position.entry_price
        else:
            points_pnl = position.entry_price - exit_price

        dollar_pnl = points_pnl * position.quantity * self.point_value
        pnl_percent = (dollar_pnl / (position.entry_price * position.quantity * self.point_value)) * 100

        # Update cash
        self.cash += dollar_pnl - self.commission_per_trade  # Subtract exit commission

        # Calculate trade duration
        duration = (bar_data['timestamp'] - position.entry_time).total_seconds() / 60

        # Create trade record
        trade = Trade(
            id=position.id,
            symbol=position.symbol,
            side=position.side,
            entry_time=position.entry_time,
            exit_time=bar_data['timestamp'],
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl=dollar_pnl,
            pnl_percent=pnl_percent,
            duration_minutes=duration,
            exit_reason=exit_reason,
            max_favorable=position.max_favorable,
            max_adverse=position.max_adverse
        )

        return trade

    def calculate_equity_curve(self) -> None:
        """Calculate and record current portfolio equity."""
        if self.current_timestamp is None:
            return

        # Calculate total unrealized P&L
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())

        # Total equity = cash + unrealized P&L
        total_equity = self.cash + unrealized_pnl

        equity_point = {
            'timestamp': self.current_timestamp,
            'cash': self.cash,
            'unrealized_pnl': unrealized_pnl,
            'total_equity': total_equity,
            'open_positions': len(self.positions),
            'total_trades': len(self.closed_trades)
        }

        self.equity_curve.append(equity_point)

    def get_total_value(self) -> float:
        """Get current total portfolio value including unrealized P&L."""
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return self.cash + unrealized_pnl

    def get_equity_series(self) -> "DataFrame":
        """Get equity curve as pandas DataFrame for analysis."""
        if not self.equity_curve:
            return pd.DataFrame()

        df = pd.DataFrame(self.equity_curve)
        df.set_index('timestamp', inplace=True)
        return df

    def get_trades_dataframe(self) -> "DataFrame":
        """Get closed trades as pandas DataFrame for analysis."""
        if not self.closed_trades:
            return pd.DataFrame()

        trades_data = []
        for trade in self.closed_trades:
            trades_data.append({
                'trade_id': trade.id,
                'symbol': trade.symbol,
                'side': trade.side,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent,
                'duration_minutes': trade.duration_minutes,
                'exit_reason': trade.exit_reason,
                'max_favorable': trade.max_favorable,
                'max_adverse': trade.max_adverse
            })

        return pd.DataFrame(trades_data)

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive portfolio performance metrics."""
        if not self.closed_trades:
            return {'error': 'No trades to analyze'}

        trades_df = self.get_trades_dataframe()
        equity_df = self.get_equity_series()

        # Basic trade statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # P&L statistics
        total_pnl = trades_df['pnl'].sum()
        avg_trade = trades_df['pnl'].mean()
        avg_winner = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loser = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0

        # Portfolio metrics
        final_equity = equity_df['total_equity'].iloc[-1] if not equity_df.empty else self.initial_capital
        total_return = ((final_equity / self.initial_capital) - 1) * 100

        # Calculate additional metrics if we have equity curve
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_percent': total_return,
            'avg_trade_pnl': avg_trade,
            'avg_winning_trade': avg_winner,
            'avg_losing_trade': avg_loser,
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'profit_factor': abs(avg_winner / avg_loser) if avg_loser != 0 else 0
        }

        # Add advanced metrics if we have sufficient data
        if not equity_df.empty and len(equity_df) > 1:
            returns = equity_df['total_equity'].pct_change().dropna()

            # Risk metrics
            if len(returns) > 0:
                # Maximum drawdown
                peak = equity_df['total_equity'].cummax()
                drawdown = (equity_df['total_equity'] - peak) / peak
                max_drawdown = drawdown.min() * 100

                # Sharpe ratio (simplified, assuming daily returns)
                if returns.std() != 0:
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
                else:
                    sharpe_ratio = 0

                metrics.update({
                    'max_drawdown_percent': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'volatility_percent': returns.std() * np.sqrt(252) * 100
                })

        return metrics