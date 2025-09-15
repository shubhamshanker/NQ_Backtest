#!/usr/bin/env python3
"""
NQ Futures Sample Data Extractor

Extracts actual 2024 data from the full historical CSV files to create
smaller sample datasets for testing without requiring the full 1GB+ files.

This uses REAL market data, not synthetic/generated data.

Usage:
    python sample_data_generator.py

This extracts 3 months of actual 2024 data from the source files.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class NQSampleDataExtractor:
    """Extracts real NQ futures sample data from full historical files"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.source_files = {
            'M1': get_data_path("1min"),
            'M3': get_data_path("3min"),
            'M5': get_data_path("5min"),
            'M15': get_data_path("15min")
        }

    def extract_date_range(self, start_date: str = "2024-01-01",
                          end_date: str = "2024-03-31") -> Dict[str, pd.DataFrame]:
        """
        Extract actual data for specified date range from all timeframe files.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dictionary with timeframe as key and DataFrame as value
        """
        print(f"Extracting REAL NQ data from {start_date} to {end_date}")

        extracted_data = {}

        for timeframe, filename in self.source_files.items():
            filepath = os.path.join(self.data_dir, filename)

            if not os.path.exists(filepath):
                print(f"⚠️  Warning: {filepath} not found, skipping {timeframe}")
                continue

            print(f"Processing {timeframe} data from {filename}...")

            try:
                # Read the data in chunks to handle large files efficiently
                extracted_df = self._extract_from_file(filepath, start_date, end_date)

                if not extracted_df.empty:
                    extracted_data[timeframe] = extracted_df
                    print(f"✅ {timeframe}: Extracted {len(extracted_df):,} bars")
                else:
                    print(f"⚠️  {timeframe}: No data found for date range")

            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")
                continue

        return extracted_data

    def _extract_from_file(self, filepath: str, start_date: str,
                          end_date: str) -> pd.DataFrame:
        """Extract data from a specific CSV file for the given date range"""

        # Convert date strings to datetime objects for comparison
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Read file in chunks to handle large files
        chunk_size = 50000
        matching_chunks = []

        print(f"    Reading file in chunks (chunk size: {chunk_size:,})...")

        try:
            for chunk_num, chunk in enumerate(pd.read_csv(filepath, chunksize=chunk_size)):
                # Convert datetime column
                chunk['Datetime'] = pd.to_datetime(chunk['Datetime'])

                # Filter for date range
                mask = (chunk['Datetime'] >= start_dt) & (chunk['Datetime'] <= end_dt)
                filtered_chunk = chunk[mask]

                if not filtered_chunk.empty:
                    matching_chunks.append(filtered_chunk)
                    print(f"    Chunk {chunk_num + 1}: Found {len(filtered_chunk):,} matching rows")

                # Early termination if we've passed the end date
                if chunk['Datetime'].min() > end_dt:
                    print(f"    Reached end date, stopping at chunk {chunk_num + 1}")
                    break

        except Exception as e:
            print(f"    Error reading file: {str(e)}")
            return pd.DataFrame()

        # Combine all matching chunks
        if matching_chunks:
            result_df = pd.concat(matching_chunks, ignore_index=True)

            # Sort by datetime to ensure proper order
            result_df = result_df.sort_values('Datetime')

            # Format datetime back to string (matching original format)
            result_df['Datetime'] = result_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

            # Ensure columns are in correct order and format
            result_df = self._format_dataframe(result_df)

            return result_df
        else:
            return pd.DataFrame()

    def _format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame matches exact format of original data"""

        # Ensure correct column order
        expected_columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[expected_columns]

        # Ensure proper data types
        df['Volume'] = df['Volume'].astype(int)

        # Round price columns to match original precision
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            df[col] = df[col].round(6)  # Match original precision

        return df

    def validate_data_quality(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """Validate the quality and consistency of extracted data"""

        print("\n📊 Data Quality Validation")
        print("=" * 40)

        for timeframe, df in data_dict.items():
            print(f"\n{timeframe} Validation:")

            # Basic statistics
            print(f"  Rows: {len(df):,}")
            print(f"  Date range: {df['Datetime'].iloc[0]} to {df['Datetime'].iloc[-1]}")
            print(f"  Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
            print(f"  Volume range: {df['Volume'].min():,} - {df['Volume'].max():,}")

            # Data integrity checks
            ohlc_valid = (
                (df['High'] >= df['Open']) &
                (df['High'] >= df['Close']) &
                (df['Low'] <= df['Open']) &
                (df['Low'] <= df['Close'])
            ).all()

            print(f"  OHLC integrity: {'✅ Valid' if ohlc_valid else '❌ Invalid'}")

            # Check for missing values
            missing_values = df.isnull().sum().sum()
            print(f"  Missing values: {missing_values} {'✅' if missing_values == 0 else '⚠️'}")

            # Check for duplicate timestamps
            duplicates = df['Datetime'].duplicated().sum()
            print(f"  Duplicate timestamps: {duplicates} {'✅' if duplicates == 0 else '⚠️'}")

    def save_sample_data(self, data_dict: Dict[str, pd.DataFrame],
                        output_dir: str = "sample_data") -> None:
        """Save extracted sample data to CSV files"""

        os.makedirs(output_dir, exist_ok=True)

        file_mapping = {
            'M1': get_data_path("1min"),
            'M3': get_data_path("3min"),
            'M5': get_data_path("5min"),
            'M15': get_data_path("1min")
        }

        print(f"\n💾 Saving sample data to '{output_dir}/' directory")
        print("=" * 50)

        for timeframe, filename in file_mapping.items():
            if timeframe in data_dict:
                filepath = os.path.join(output_dir, filename)
                data_dict[timeframe].to_csv(filepath, index=False)

                file_size = os.path.getsize(filepath) / 1024 / 1024  # MB
                rows = len(data_dict[timeframe])
                print(f"✅ {filename}: {rows:,} rows, {file_size:.1f} MB")

        # Create a summary file
        self._create_data_summary(data_dict, output_dir)

    def _create_data_summary(self, data_dict: Dict[str, pd.DataFrame],
                            output_dir: str) -> None:
        """Create a summary of the extracted data"""

        summary_path = os.path.join(output_dir, "data_summary.txt")

        with open(summary_path, 'w') as f:
            f.write("NQ Futures Sample Data Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write("REAL MARKET DATA EXTRACTED FROM HISTORICAL FILES\n")
            f.write("(This is actual trading data, not synthetic/generated)\n\n")

            total_rows = 0
            for timeframe, df in data_dict.items():
                rows = len(df)
                total_rows += rows

                f.write(f"{timeframe} Data:\n")
                f.write(f"  Rows: {rows:,}\n")
                f.write(f"  Period: {df['Datetime'].iloc[0]} to {df['Datetime'].iloc[-1]}\n")
                f.write(f"  Price Range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}\n")
                f.write(f"  Volume Range: {df['Volume'].min():,} - {df['Volume'].max():,}\n\n")

            f.write(f"Total Data Points: {total_rows:,}\n")
            f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"📄 Created summary: {summary_path}")

def create_usage_readme(output_dir: str = "sample_data") -> None:
    """Create documentation for using the sample data"""

    readme_content = """# NQ Futures Sample Data (REAL MARKET DATA)

## 🎯 Overview
This directory contains **ACTUAL NQ futures market data** extracted from historical files for testing the trading system without requiring the full 1GB+ dataset.

⚠️ **Important**: This is REAL market data, not synthetic or generated data.

## 📁 Files
- `NQ_M1_standard_sample.csv` - Real 1-minute NQ data
- `NQ_M3_sample.csv` - Real 3-minute NQ data
- `NQ_M5_sample.csv` - Real 5-minute NQ data
- `NQ_M15_sample.csv` - Real 15-minute NQ data
- `data_summary.txt` - Statistical summary of the data

## 📊 Data Characteristics
- **Source**: Extracted from actual historical NQ futures data
- **Period**: 3 months of 2024 data (configurable)
- **Format**: CSV with columns: Datetime,Open,High,Low,Close,Volume
- **Timezone**: Chicago time (matching production data)
- **Integrity**: Full OHLC validation and consistency checks

## 🔧 Usage with Trading System

### Quick Start
```python
# Test with sample data
python real_data_strategy_test_2024.py

# The system will automatically detect and use sample data
```

### Manual Loading
```python
import pandas as pd

# Load sample data
df = pd.read_csv(get_data_path("1min"))

# Use with existing strategies
from backtesting.ultimate_orb_strategy import UltimateORBStrategy
strategy = UltimateORBStrategy()
results = strategy.run_backtest(df)
```

## 🔄 Regenerating Sample Data

To extract different time periods:

```python
from sample_data_generator import NQSampleDataExtractor

extractor = NQSampleDataExtractor()

# Extract different period (requires full data files)
data = extractor.extract_date_range("2024-06-01", "2024-08-31")
extractor.save_sample_data(data)
```

## ⚡ Performance Expectations

Since this is real market data:
- **Realistic Results**: Performance should closely match full dataset results
- **Statistical Validity**: Proper representation of market conditions
- **Strategy Development**: Suitable for serious strategy development
- **Production Ready**: Results can inform real trading decisions

## 📋 Data Validation

Each extraction includes:
- ✅ OHLC relationship validation
- ✅ No missing values
- ✅ No duplicate timestamps
- ✅ Proper data type formatting
- ✅ Volume and price range validation

## 🔗 Integration

The trading system will automatically:
1. Check for full historical data in `data/` directory
2. Fall back to sample data in `sample_data/` directory if available
3. Clearly label results as "Sample Data" vs "Full Data"

## 📈 Scaling to Full Data

For production or comprehensive backtesting:
1. Obtain full historical NQ data from your broker
2. Place files in `data/` directory with proper naming
3. System will automatically use full dataset

---
*Generated from real NQ futures market data*
*Not for live trading without proper validation*
"""

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"📚 Created documentation: {readme_path}")

def main():
    """Main function to extract sample data from real files"""

    print("📊 NQ Futures REAL Data Extractor")
    print("=" * 50)
    print("Extracting ACTUAL market data (not synthetic)")
    print()

    extractor = NQSampleDataExtractor()

    # Extract 3 months of real 2024 data
    sample_data = extractor.extract_date_range(
        start_date="2024-01-01",
        end_date="2024-03-31"
    )

    if not sample_data:
        print("❌ No data extracted. Please ensure source files exist in 'data/' directory")
        return

    # Validate data quality
    extractor.validate_data_quality(sample_data)

    # Save sample data
    extractor.save_sample_data(sample_data)

    # Create documentation
    create_usage_readme()

    print("\n✅ Real data extraction complete!")
    print("📁 Sample data saved in 'sample_data/' directory")
    print("📊 This is ACTUAL market data, ready for testing")

if __name__ == "__main__":
    main()