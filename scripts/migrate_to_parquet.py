#!/usr/bin/env python3
"""
Data Migration Script: CSV → Parquet + DuckDB
=============================================
Migrates NQ historical data from CSV to optimized Parquet format with:
- Chicago → New York timezone conversion
- NY session filtering (09:30-16:00)
- Data validation and cleaning
- Partitioned storage for optimal query performance
"""

import pandas as pd
import numpy as np
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, time
import logging
from typing import Dict, List, Tuple, Optional
import hashlib
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/shubhamshanker/bt_/migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataMigrationEngine:
    """Handles migration from CSV to Parquet with timezone conversion."""

    def __init__(self, source_dir: str = "/Users/shubhamshanker/bt_/data",
                 target_dir: str = "/Users/shubhamshanker/bt_/data_parquet"):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.validation_report = {}

        # Define file mappings
        self.csv_files = {
            '1min': 'NQ_M1_standard.csv',
            '3min': 'NQ_M3.csv',
            '5min': 'NQ_M5.csv',
            '15min': 'NQ_M15.csv'
        }

        # NY trading session hours
        self.session_start = time(9, 30)
        self.session_end = time(16, 0)

    def validate_csv_data(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Validate CSV data quality and generate report."""
        logger.info(f"🔍 Validating {timeframe} data...")

        validation = {
            'timeframe': timeframe,
            'total_rows': len(df),
            'date_range': {
                'start': df.index.min().isoformat() if len(df) > 0 else None,
                'end': df.index.max().isoformat() if len(df) > 0 else None
            },
            'issues': []
        }

        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            validation['issues'].append(f"Missing columns: {missing_cols}")

        # Check for OHLC consistency
        if not missing_cols:
            ohlc_issues = (
                (df['Low'] > df['Open']) |
                (df['Low'] > df['Close']) |
                (df['High'] < df['Open']) |
                (df['High'] < df['Close'])
            ).sum()

            if ohlc_issues > 0:
                validation['issues'].append(f"OHLC inconsistencies: {ohlc_issues} rows")

            # Check for negative values
            negative_prices = (df[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1).sum()
            if negative_prices > 0:
                validation['issues'].append(f"Negative/zero prices: {negative_prices} rows")

            # Check for extreme values (potential errors)
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                extreme_values = ((df[col] > q99 * 2) | (df[col] < q01 * 0.5)).sum()
                if extreme_values > 0:
                    validation['issues'].append(f"Extreme {col} values: {extreme_values} rows")

        # Check for duplicates
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            validation['issues'].append(f"Duplicate timestamps: {duplicates}")

        # Check for gaps (missing data)
        if len(df) > 1:
            time_diff = df.index.to_series().diff()
            expected_freq = pd.Timedelta(minutes=int(timeframe.replace('min', '')))
            large_gaps = (time_diff > expected_freq * 2).sum()
            validation['issues'].append(f"Large time gaps: {large_gaps}")

        validation['is_valid'] = len(validation['issues']) == 0
        logger.info(f"✅ Validation complete: {len(validation['issues'])} issues found")

        return validation

    def clean_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Clean and prepare data for migration."""
        logger.info(f"🧹 Cleaning {timeframe} data...")

        original_size = len(df)

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        logger.info(f"Removed {original_size - len(df)} duplicate timestamps")

        # Fix OHLC inconsistencies by adjusting High/Low
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Adjust High to be at least max(Open, Close)
            df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))
            # Adjust Low to be at most min(Open, Close)
            df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))

        # Remove rows with negative or zero prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        before_filter = len(df)
        df = df[(df[price_cols] > 0).all(axis=1)]
        logger.info(f"Removed {before_filter - len(df)} rows with invalid prices")

        # Remove extreme outliers (beyond 3 standard deviations)
        for col in price_cols:
            mean_price = df[col].mean()
            std_price = df[col].std()
            df = df[np.abs(df[col] - mean_price) <= 3 * std_price]

        # Ensure Volume is non-negative
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].clip(lower=0)

        logger.info(f"✅ Cleaning complete: {original_size} → {len(df)} rows")
        return df.sort_index()

    def convert_timezone(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Convert Chicago time to New York time with proper DST handling."""
        logger.info(f"🌐 Converting {timeframe} timezone: Chicago → New York...")

        original_size = len(df)

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        # Handle timezone conversion
        if df.index.tz is None:
            # Localize to Chicago time
            df.index = df.index.tz_localize(
                'America/Chicago',
                ambiguous='infer',  # Handle DST ambiguity
                nonexistent='shift_forward'  # Handle DST gaps
            )
        elif df.index.tz.zone != 'America/Chicago':
            logger.warning(f"Data already has timezone {df.index.tz}, converting to Chicago first")
            df.index = df.index.tz_convert('America/Chicago')

        # Convert to New York time
        df.index = df.index.tz_convert('America/New_York')

        # Filter to NY regular session hours (09:30-16:00 ET)
        session_mask = (
            (df.index.time >= self.session_start) &
            (df.index.time <= self.session_end) &
            (df.index.dayofweek < 5)  # Monday=0, Friday=4
        )
        df = df[session_mask]

        logger.info(f"✅ Timezone conversion complete: {original_size} → {len(df)} rows (filtered to NY session)")
        return df

    def save_as_parquet(self, df: pd.DataFrame, timeframe: str) -> str:
        """Save DataFrame as partitioned Parquet files."""
        logger.info(f"💾 Saving {timeframe} data as Parquet...")

        if len(df) == 0:
            logger.warning(f"Empty DataFrame for {timeframe}, skipping...")
            return ""

        # Add partitioning columns
        df = df.copy()
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['symbol'] = 'NQ'
        df['timeframe'] = timeframe

        # Reset index to make datetime a column
        df = df.reset_index()
        df = df.rename(columns={'Datetime': 'datetime'})

        # Standardize column names (lowercase)
        df.columns = df.columns.str.lower()

        # Define output path
        output_path = self.target_dir / 'nq' / timeframe
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as partitioned Parquet
        table = pa.Table.from_pandas(df)

        # Use Hive-style partitioning
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=['year', 'month'],
            compression='snappy',
            use_dictionary=True,
            row_group_size=100000  # Optimize for query performance
        )

        logger.info(f"✅ Saved {len(df)} rows to {output_path}")
        return str(output_path)

    def generate_data_hash(self, df: pd.DataFrame) -> str:
        """Generate hash for data validation."""
        # Create deterministic hash from data
        data_str = f"{len(df)}_{df.index.min()}_{df.index.max()}_{df.sum().sum()}"
        return hashlib.md5(data_str.encode()).hexdigest()

    def migrate_timeframe(self, timeframe: str) -> Dict:
        """Migrate a single timeframe from CSV to Parquet."""
        logger.info(f"🚀 Starting migration for {timeframe}")

        csv_file = self.source_dir / self.csv_files[timeframe]
        if not csv_file.exists():
            logger.error(f"CSV file not found: {csv_file}")
            return {'success': False, 'error': f'File not found: {csv_file}'}

        try:
            # Load CSV data
            logger.info(f"📂 Loading {csv_file}")
            df = pd.read_csv(csv_file, parse_dates=['Datetime'], index_col='Datetime')
            logger.info(f"Loaded {len(df):,} rows from {timeframe} CSV")

            # Validate original data
            validation = self.validate_csv_data(df, timeframe)
            original_hash = self.generate_data_hash(df)

            # Clean data
            df_cleaned = self.clean_data(df, timeframe)

            # Convert timezone and filter to NY session
            df_final = self.convert_timezone(df_cleaned, timeframe)

            # Save as Parquet
            parquet_path = self.save_as_parquet(df_final, timeframe)
            final_hash = self.generate_data_hash(df_final)

            # Create migration report
            migration_report = {
                'success': True,
                'timeframe': timeframe,
                'source_file': str(csv_file),
                'target_path': parquet_path,
                'original_rows': len(df),
                'final_rows': len(df_final),
                'compression_ratio': len(df_final) / len(df) if len(df) > 0 else 0,
                'original_hash': original_hash,
                'final_hash': final_hash,
                'validation': validation,
                'migration_time': datetime.now().isoformat()
            }

            logger.info(f"✅ Migration complete for {timeframe}: {len(df):,} → {len(df_final):,} rows")
            return migration_report

        except Exception as e:
            logger.error(f"❌ Migration failed for {timeframe}: {e}")
            return {
                'success': False,
                'timeframe': timeframe,
                'error': str(e),
                'migration_time': datetime.now().isoformat()
            }

    def migrate_all(self) -> Dict:
        """Migrate all timeframes and generate comprehensive report."""
        logger.info("🚀 Starting full data migration...")

        start_time = datetime.now()
        migration_reports = {}

        for timeframe in self.csv_files.keys():
            migration_reports[timeframe] = self.migrate_timeframe(timeframe)

        # Generate summary report
        successful_migrations = sum(1 for report in migration_reports.values() if report.get('success', False))
        total_original_rows = sum(report.get('original_rows', 0) for report in migration_reports.values())
        total_final_rows = sum(report.get('final_rows', 0) for report in migration_reports.values())

        summary_report = {
            'migration_summary': {
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_minutes': (datetime.now() - start_time).total_seconds() / 60,
                'successful_migrations': successful_migrations,
                'total_migrations': len(self.csv_files),
                'total_original_rows': total_original_rows,
                'total_final_rows': total_final_rows,
                'overall_compression_ratio': total_final_rows / total_original_rows if total_original_rows > 0 else 0
            },
            'timeframe_reports': migration_reports,
            'data_structure': {
                'target_directory': str(self.target_dir),
                'partitioning_scheme': 'symbol/timeframe/year/month',
                'timezone': 'America/New_York',
                'session_hours': f"{self.session_start} - {self.session_end}",
                'format': 'Parquet with Snappy compression'
            }
        }

        # Save report
        report_file = self.target_dir / 'migration_report.json'
        with open(report_file, 'w') as f:
            json.dump(summary_report, f, indent=2)

        logger.info(f"🎉 Migration complete! {successful_migrations}/{len(self.csv_files)} timeframes migrated successfully")
        logger.info(f"📊 Total rows: {total_original_rows:,} → {total_final_rows:,}")
        logger.info(f"📋 Report saved to: {report_file}")

        return summary_report

def main():
    """Main migration function."""
    print("🚀 Starting CSV to Parquet Migration...")
    print("=" * 50)

    # Initialize migration engine
    migrator = DataMigrationEngine()

    # Run migration
    report = migrator.migrate_all()

    # Print summary
    summary = report['migration_summary']
    print(f"\n✅ Migration Complete!")
    print(f"Duration: {summary['duration_minutes']:.2f} minutes")
    print(f"Success rate: {summary['successful_migrations']}/{summary['total_migrations']}")
    print(f"Data processed: {summary['total_original_rows']:,} → {summary['total_final_rows']:,} rows")
    print(f"Compression ratio: {summary['overall_compression_ratio']:.2%}")

    # Show any issues
    for timeframe, tf_report in report['timeframe_reports'].items():
        if not tf_report.get('success', False):
            print(f"❌ Failed: {timeframe} - {tf_report.get('error', 'Unknown error')}")
        elif tf_report.get('validation', {}).get('issues'):
            print(f"⚠️  Issues in {timeframe}: {tf_report['validation']['issues']}")

if __name__ == "__main__":
    main()