"""
Data Handler Module
==================
Handles multi-timeframe data loading and streaming for event-driven backtesting.
Optimized for speed with Parquet support and efficient data structures.
"""

import pandas as pd
import numpy as np
from typing import Iterator, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from pandas import DataFrame
import warnings
warnings.filterwarnings('ignore')

class DataHandler:
    """
    Handles data loading, timeframe conversion, and event-driven data streaming.
    Provides one bar at a time for the main backtesting loop.
    """

    def __init__(self, data_path: str, timeframe: str = "15min",
                 start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Initialize DataHandler with data source and parameters.

        Args:
            data_path: Path to data file (CSV or Parquet)
            timeframe: Target timeframe (1min, 3min, 5min, 15min, etc.)
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
        """
        self.data_path = Path(data_path)
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.data: Optional["DataFrame"] = None
        self.current_index = 0
        self.total_bars = 0

    def load_data(self) -> None:
        """Load and prepare data from file with timezone handling."""
        try:
            # Attempt to load Parquet first (faster)
            if self.data_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(self.data_path)
            else:
                # Load CSV with proper datetime parsing
                df = pd.read_csv(self.data_path, parse_dates=['Datetime'], index_col='Datetime')

            print(f"✅ Loaded {len(df):,} bars from {self.data_path.name}")

        except Exception as e:
            raise FileNotFoundError(f"Failed to load data from {self.data_path}: {e}")

        # Handle timezone conversion (Chicago -> NY)
        if hasattr(df.index, 'tz'):
            if df.index.tz is None:
                df.index = df.index.tz_localize("America/Chicago", ambiguous='infer', nonexistent='shift_forward')
            df.index = df.index.tz_convert("America/New_York")
        else:
            # If not a DatetimeIndex, skip timezone handling
            print("⚠️ Non-datetime index detected, skipping timezone conversion")

        # CRITICAL: Filter out extended hours data to prevent strategy timing issues
        original_bars = len(df)
        df = self._filter_regular_session_hours(df)
        filtered_bars = len(df)

        if original_bars > filtered_bars:
            print(f"🔄 Filtered extended hours: {original_bars:,} → {filtered_bars:,} bars ({original_bars-filtered_bars:,} removed)")

        # Date filtering
        if self.start_date:
            df = df[df.index >= self.start_date]
        if self.end_date:
            df = df[df.index <= self.end_date]

        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert to target timeframe if needed
        df = self._convert_timeframe(df, self.timeframe)

        # Add technical indicators commonly needed
        df = self._add_basic_indicators(df)

        # Sort by datetime and clean
        df = df.sort_index().dropna()

        self.data = df
        self.total_bars = len(df)
        self.current_index = 0

        print(f"📊 Data prepared: {self.total_bars:,} bars, {df.index[0].date()} to {df.index[-1].date()}")

    def _convert_timeframe(self, df: "DataFrame", target_timeframe: str) -> "DataFrame":
        """Convert data to target timeframe using proper OHLCV aggregation."""

        # Parse timeframe
        if target_timeframe.endswith('min'):
            freq = target_timeframe.replace('min', 'T')
        elif target_timeframe.endswith('h'):
            freq = target_timeframe.replace('h', 'H')
        else:
            # Assume already correct timeframe
            return df

        # Resample with proper OHLCV logic
        resampled = df.resample(freq).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        print(f"🔄 Converted to {target_timeframe}: {len(resampled):,} bars")
        return resampled

    def _add_basic_indicators(self, df: "DataFrame") -> "DataFrame":
        """Add commonly used technical indicators for strategy use."""
        df = df.copy()

        # Price-based indicators
        df['HL2'] = (df['High'] + df['Low']) / 2
        df['HLC3'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['OHLC4'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

        # Basic volatility
        df['Range'] = df['High'] - df['Low']
        df['TrueRange'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )

        return df

    def next_bar(self) -> Optional[Dict[str, Any]]:
        """
        Get next bar of data for event-driven processing.

        Returns:
            Dictionary with bar data or None if no more data
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if self.current_index >= self.total_bars:
            return None

        # Get current bar
        timestamp = self.data.index[self.current_index]
        row = self.data.iloc[self.current_index]

        # Create bar dictionary
        bar_data = {
            'timestamp': timestamp,
            'datetime': timestamp,
            'open': float(row['Open']),
            'high': float(row['High']),
            'low': float(row['Low']),
            'close': float(row['Close']),
            'volume': float(row['Volume']),
            'range': float(row['Range']),
            'true_range': float(row['TrueRange']),
            'hl2': float(row['HL2']),
            'hlc3': float(row['HLC3']),
            'ohlc4': float(row['OHLC4'])
        }

        self.current_index += 1
        return bar_data

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Make DataHandler iterable for main loop."""
        while True:
            bar = self.next_bar()
            if bar is None:
                break
            yield bar

    def reset(self) -> None:
        """Reset iterator to beginning."""
        self.current_index = 0

    def get_historical_data(self, lookback: int) -> Optional["DataFrame"]:
        """
        Get historical data for strategy calculations (e.g., moving averages).

        Args:
            lookback: Number of bars to look back from current position

        Returns:
            DataFrame with historical data or None if insufficient data
        """
        if self.data is None or self.current_index < lookback:
            return None

        start_idx = max(0, self.current_index - lookback)
        end_idx = self.current_index

        return self.data.iloc[start_idx:end_idx].copy()

    def _filter_regular_session_hours(self, df: "DataFrame") -> "DataFrame":
        """
        Filter out extended hours data, keeping only regular NY session hours.

        Args:
            df: DataFrame with datetime index

        Returns:
            Filtered DataFrame with only regular session hours (9:15 AM - 4:15 PM ET)
        """
        if not hasattr(df.index, 'time'):
            return df  # Not a datetime index, return as-is

        # Use slightly extended hours (9:15-16:15) to provide buffer for strategies
        # Strategies will further filter to (9:30-15:45) for signal generation
        session_start = pd.to_datetime("09:15", format="%H:%M").time()
        session_end = pd.to_datetime("16:15", format="%H:%M").time()

        # Create mask for regular session hours
        session_mask = (df.index.time >= session_start) & (df.index.time <= session_end)

        # Apply mask and return filtered data
        return df[session_mask].copy()

    @property
    def progress(self) -> float:
        """Get progress as percentage."""
        if self.total_bars == 0:
            return 0.0
        return (self.current_index / self.total_bars) * 100