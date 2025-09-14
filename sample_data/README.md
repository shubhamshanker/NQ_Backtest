# NQ Futures Sample Data (REAL MARKET DATA)

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
df = pd.read_csv('sample_data/NQ_M1_standard_sample.csv')

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
