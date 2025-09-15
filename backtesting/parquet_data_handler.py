"""
Parquet Data Handler with DuckDB
===============================
High-performance data loading and querying for backtesting.
- DuckDB-based SQL queries on Parquet files
- Automatic resampling from 1min to higher timeframes
- NY session filtering built-in
- Memory-efficient streaming for large datasets
"""

import duckdb
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Iterator
from datetime import datetime, time, date
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ParquetDataHandler:
    """High-performance data handler using DuckDB queries on Parquet files."""

    def __init__(self, data_root: str = "/Users/shubhamshanker/bt_/data_parquet",
                 connection: Optional[duckdb.DuckDBPyConnection] = None):
        """
        Initialize Parquet data handler.

        Args:
            data_root: Root directory containing Parquet files
            connection: Optional existing DuckDB connection
        """
        self.data_root = Path(data_root)
        self.conn = connection or duckdb.connect(":memory:")
        self.current_data: Optional[pd.DataFrame] = None
        self.current_index = 0

        # NY trading session hours
        self.session_start = time(9, 30)
        self.session_end = time(16, 0)

        # Cache for query optimization
        self._schema_cache = {}

        # Set up DuckDB for optimal Parquet reading
        self._configure_duckdb()

    def _configure_duckdb(self):
        """Configure DuckDB for optimal performance."""
        try:
            # Set timezone to NY for proper datetime handling
            self.conn.execute("SET TimeZone='America/New_York'")
            # Set threads to 1 to avoid the error
            self.conn.execute("SET threads TO 1")
            # Optimize for analytical workloads
            self.conn.execute("SET memory_limit='1GB'")
            logger.info("✅ DuckDB configured for optimal performance (NY timezone)")
        except Exception as e:
            logger.warning(f"DuckDB configuration warning: {e}")

    def get_available_data(self) -> Dict[str, Dict]:
        """Get information about available data files."""
        available_data = {}

        symbol_dir = self.data_root / 'nq'
        if not symbol_dir.exists():
            return available_data

        for timeframe_dir in symbol_dir.iterdir():
            if timeframe_dir.is_dir():
                timeframe = timeframe_dir.name
                parquet_files = list(timeframe_dir.rglob("*.parquet"))

                if parquet_files:
                    # Try to get date range from first file
                    try:
                        sample_df = pd.read_parquet(parquet_files[0])
                        date_range = {
                            'start': sample_df['datetime'].min(),
                            'end': sample_df['datetime'].max(),
                            'files': len(parquet_files)
                        }
                    except Exception:
                        date_range = {'files': len(parquet_files)}

                    available_data[timeframe] = date_range

        return available_data

    def load_data(self, symbol: str = "NQ", timeframe: str = "1min",
                  start_date: Optional[Union[str, date]] = None,
                  end_date: Optional[Union[str, date]] = None,
                  session_filter: bool = True) -> pd.DataFrame:
        """
        Load data using DuckDB SQL queries on Parquet files.

        Args:
            symbol: Trading symbol (default: "NQ")
            timeframe: Timeframe (1min, 3min, 5min, 15min)
            start_date: Start date filter
            end_date: End date filter
            session_filter: Filter to NY session hours

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"🔍 Loading {symbol} {timeframe} data...")

        # Build file pattern
        pattern = str(self.data_root / symbol.lower() / timeframe / "**" / "*.parquet")

        # Build SQL query
        query_parts = [
            "SELECT datetime, open, high, low, close, volume",
            f"FROM read_parquet('{pattern}')",
            "WHERE 1=1"
        ]

        # Add date filters
        if start_date:
            if isinstance(start_date, str):
                start_date_str = start_date
            else:
                start_date_str = str(start_date)
            query_parts.append(f"AND datetime >= '{start_date_str}'")

        if end_date:
            if isinstance(end_date, str):
                end_date_str = end_date
            else:
                end_date_str = str(end_date)
            query_parts.append(f"AND datetime <= '{end_date_str} 23:59:59'")

        # Add session filter - data is already in NY time
        if session_filter:
            query_parts.extend([
                "AND EXTRACT(hour FROM datetime) * 60 + EXTRACT(minute FROM datetime) >= 570",  # 9:30 NY
                "AND EXTRACT(hour FROM datetime) * 60 + EXTRACT(minute FROM datetime) <= 960",   # 16:00 NY
                "AND EXTRACT(dow FROM datetime) BETWEEN 1 AND 5"  # Monday=1, Friday=5
            ])

        query_parts.append("ORDER BY datetime")
        query = " ".join(query_parts)

        try:
            # Execute query
            result = self.conn.execute(query).fetchdf()

            if len(result) == 0:
                logger.warning(f"No data found for {symbol} {timeframe}")
                return pd.DataFrame()

            # Ensure datetime index
            result['datetime'] = pd.to_datetime(result['datetime'])
            result = result.set_index('datetime')

            # Convert to NY timezone
            if result.index.tz is None:
                result.index = result.index.tz_localize('America/New_York')
            else:
                result.index = result.index.tz_convert('America/New_York')

            # Standardize column names
            result.columns = [col.title() for col in result.columns]

            logger.info(f"✅ Loaded {len(result):,} rows ({result.index[0].date()} to {result.index[-1].date()})")
            return result

        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            # Fallback: try direct file reading
            return self._fallback_load(symbol, timeframe, start_date, end_date, session_filter)

    def _fallback_load(self, symbol: str, timeframe: str, start_date: Optional[date],
                      end_date: Optional[date], session_filter: bool) -> pd.DataFrame:
        """Fallback method using direct Parquet file reading."""
        logger.info("🔄 Using fallback loading method...")

        timeframe_dir = self.data_root / symbol.lower() / timeframe
        if not timeframe_dir.exists():
            logger.error(f"Timeframe directory not found: {timeframe_dir}")
            return pd.DataFrame()

        # Find all Parquet files
        parquet_files = list(timeframe_dir.rglob("*.parquet"))
        if not parquet_files:
            logger.error(f"No Parquet files found in {timeframe_dir}")
            return pd.DataFrame()

        # Load and concatenate all files
        dfs = []
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to read {file}: {e}")

        if not dfs:
            return pd.DataFrame()

        # Combine all data
        result = pd.concat(dfs, ignore_index=True)
        result['datetime'] = pd.to_datetime(result['datetime'])
        result = result.set_index('datetime').sort_index()

        # Apply filters
        if start_date:
            result = result[result.index.date >= start_date]
        if end_date:
            result = result[result.index.date <= end_date]

        if session_filter:
            session_mask = (
                (result.index.time >= self.session_start) &
                (result.index.time <= self.session_end) &
                (result.index.dayofweek < 5)
            )
            result = result[session_mask]

        # Standardize columns
        if 'open' in result.columns:
            result.columns = [col.title() for col in result.columns]

        return result

    def resample_data(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        Resample data to target timeframe using proper OHLCV aggregation.

        Args:
            df: Source DataFrame
            target_timeframe: Target timeframe (3min, 5min, 15min, etc.)

        Returns:
            Resampled DataFrame
        """
        logger.info(f"🔄 Resampling to {target_timeframe}")

        if len(df) == 0:
            return df

        # Parse timeframe
        if target_timeframe.endswith('min'):
            freq = target_timeframe.replace('min', 'T')
        elif target_timeframe.endswith('h'):
            freq = target_timeframe.replace('h', 'H')
        else:
            logger.warning(f"Unknown timeframe format: {target_timeframe}")
            return df

        # Resample with proper OHLCV logic
        try:
            resampled = df.resample(freq).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

            logger.info(f"✅ Resampled: {len(df):,} → {len(resampled):,} bars")
            return resampled

        except Exception as e:
            logger.error(f"❌ Resampling failed: {e}")
            return df

    def query_data(self, sql: str) -> pd.DataFrame:
        """
        Execute custom SQL query on Parquet data.

        Args:
            sql: SQL query string

        Returns:
            Query results as DataFrame
        """
        try:
            result = self.conn.execute(sql).fetchdf()
            logger.info(f"✅ Query executed: {len(result)} rows returned")
            return result
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return pd.DataFrame()

    def get_data_stats(self, symbol: str = "NQ", timeframe: str = "1min") -> Dict[str, Any]:
        """Get statistical summary of available data."""
        pattern = str(self.data_root / symbol.lower() / timeframe / "**" / "*.parquet")

        query = f"""
        SELECT
            COUNT(*) as total_rows,
            MIN(datetime) as start_date,
            MAX(datetime) as end_date,
            MIN(low) as min_price,
            MAX(high) as max_price,
            AVG(volume) as avg_volume,
            SUM(volume) as total_volume
        FROM read_parquet('{pattern}')
        """

        try:
            result = self.conn.execute(query).fetchone()
            return {
                'total_rows': result[0],
                'date_range': {'start': result[1], 'end': result[2]},
                'price_range': {'min': result[3], 'max': result[4]},
                'volume': {'average': result[5], 'total': result[6]}
            }
        except Exception as e:
            logger.error(f"❌ Stats query failed: {e}")
            return {}

    def create_data_iterator(self, symbol: str = "NQ", timeframe: str = "1min",
                           start_date: Optional[str] = None, end_date: Optional[str] = None,
                           chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """
        Create memory-efficient iterator for large datasets.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date
            end_date: End date
            chunk_size: Number of rows per chunk

        Yields:
            DataFrame chunks
        """
        pattern = str(self.data_root / symbol.lower() / timeframe / "**" / "*.parquet")

        # Build base query
        query_base = f"""
        SELECT datetime, open, high, low, close, volume
        FROM read_parquet('{pattern}')
        WHERE 1=1
        """

        if start_date:
            query_base += f" AND datetime >= '{start_date}'"
        if end_date:
            query_base += f" AND datetime <= '{end_date}'"

        query_base += " ORDER BY datetime"

        try:
            # Get total count first
            count_query = f"SELECT COUNT(*) FROM ({query_base}) t"
            total_rows = self.conn.execute(count_query).fetchone()[0]

            logger.info(f"📊 Creating iterator for {total_rows:,} rows in chunks of {chunk_size:,}")

            # Iterate through chunks
            offset = 0
            while offset < total_rows:
                chunk_query = f"{query_base} LIMIT {chunk_size} OFFSET {offset}"
                chunk = self.conn.execute(chunk_query).fetchdf()

                if len(chunk) == 0:
                    break

                # Process chunk
                chunk['datetime'] = pd.to_datetime(chunk['datetime'])
                chunk = chunk.set_index('datetime')
                chunk.columns = [col.title() for col in chunk.columns]

                yield chunk
                offset += chunk_size

        except Exception as e:
            logger.error(f"❌ Iterator creation failed: {e}")

    def close(self):
        """Close DuckDB connection."""
        if self.conn:
            self.conn.close()
            logger.info("🔒 DuckDB connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Convenience functions
def load_nq_data(timeframe: str = "1min", start_date: Optional[str] = None,
                 end_date: Optional[str] = None, data_root: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to load NQ data.

    Args:
        timeframe: Data timeframe
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        data_root: Root data directory

    Returns:
        OHLCV DataFrame
    """
    with ParquetDataHandler(data_root or "/Users/shubhamshanker/bt_/data_parquet") as handler:
        return handler.load_data("NQ", timeframe, start_date, end_date)

def get_data_summary(data_root: Optional[str] = None) -> Dict[str, Any]:
    """Get summary of all available data."""
    with ParquetDataHandler(data_root or "/Users/shubhamshanker/bt_/data_parquet") as handler:
        return handler.get_available_data()

# Example usage
if __name__ == "__main__":
    # Test the data handler
    with ParquetDataHandler() as handler:
        # Get available data summary
        print("📊 Available data:")
        summary = handler.get_available_data()
        for tf, info in summary.items():
            print(f"  {tf}: {info}")

        # Load sample data
        print("\n📈 Loading 1-minute data for 2024...")
        df = handler.load_data("NQ", "1min", "2024-01-01", "2024-01-31")
        print(f"Loaded {len(df)} rows")
        if len(df) > 0:
            print(df.head())

        # Test resampling
        if len(df) > 0:
            print("\n🔄 Testing resampling to 5min...")
            df_5min = handler.resample_data(df, "5min")
            print(f"Resampled to {len(df_5min)} rows")