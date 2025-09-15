# 🚀 NQ Futures Trading System V1

**Systematic Opening Range Breakout (ORB) trading system targeting 30+ points per day**

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Status](https://img.shields.io/badge/Status-Active%20Development-green)
![Performance](https://img.shields.io/badge/Current%20Performance-10--15%20pts/day-yellow)
![Target](https://img.shields.io/badge/Target-30+%20pts/day-brightgreen)

---

## 📖 Quick Start

### Prerequisites
- Python 3.11 or higher
- 8GB+ RAM (for data processing)
- 2GB+ disk space (for market data)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd bt_
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install pandas numpy fastapi uvicorn scipy pydantic
   ```

4. **Verify data files**
   ```bash
   ls -la data/
   # Should show NQ_M1_standard.csv and other timeframe files
   ```

---

## 🏃‍♂️ Running the System

### 1. Run Latest Strategy Optimization
```bash
python optimized_multi_strategy_system.py
```

### 2. Test Individual Strategies
```bash
python real_data_strategy_test_2024.py
```

### 3. Start Backend API
```bash
cd trading_dashboard/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Run Comprehensive Analysis
```bash
python comprehensive_strategy_test_2024.py
```

---

## 📊 Current Performance

### ✅ **Real Data Results (2024)**

| Strategy | Daily Points | Win Rate | Sharpe | Status |
|----------|-------------|----------|---------|--------|
| **ULTIMATE_50** | **10.18** | 25.9% | 1.692 | ✅ Best |
| **ULTIMATE_30** | **7.65** | 23.7% | 2.041 | ✅ Stable |
| **ULTIMATE_25** | **6.33** | 26.3% | 1.541 | ✅ Consistent |

### 🎯 **Key Metrics**
- **Current Achievement:** 10-15 points/day
- **Target Goal:** 30+ points/day
- **Max Risk:** 3 trades/day, 1 contract
- **Win Rate:** 25-30% (realistic for breakouts)
- **Risk-Reward:** 3:1 to 6:1 ratios

---

## 🗂️ Project Structure

```
bt_/
├── 📁 data/                    # Market Data (CRITICAL - 1GB+)
│   ├── NQ_M1_standard.csv     # Main 1-minute data
│   ├── NQ_M3.csv, NQ_M5.csv   # Multi-timeframe data
│   └── NQ_M15.csv             # 15-minute data
│
├── 📁 backtesting/             # Core Strategy Framework
│   ├── ultimate_orb_strategy.py      # Main ORB strategy
│   ├── parameter_optimization.py     # Strategy optimizer
│   ├── regime_optimized_orb.py      # Market regime detection
│   ├── advanced_trade_management.py  # Position management
│   └── portfolio.py                 # Risk management
│
├── 📁 trading_dashboard/backend/     # API Backend
│   ├── main.py               # FastAPI application
│   └── data_processor.py     # Data processing
│
├── 📄 optimized_multi_strategy_system.py  # Latest optimization
├── 📄 advanced_filtered_strategies.py     # Advanced filtering
├── 📄 real_data_strategy_test_2024.py    # Real data testing
├── 📄 PROJECT_SUMMARY_V1.md              # Complete overview
└── 📄 README.md                          # This file
```

---

## 🔧 Configuration

### Strategy Parameters

**ULTIMATE_50 (Best Performer):**
```python
risk_per_trade = 0.03          # 3% risk
or_minutes = 45                # 45-minute opening range
fixed_stop_points = 30.0       # 30-point stops
target_multiplier = 5.0        # 5:1 risk-reward (150pt targets)
max_trades_per_day = 2         # Maximum 2 trades
```

### Risk Management
- **Daily Stop:** 100 points maximum loss
- **Position Size:** 1 contract (NQ = $20/point)
- **Trading Hours:** 9:30 AM - 4:00 PM EST
- **Max Drawdown:** <600 points target

---

## 📈 Strategy Logic

### Opening Range Breakout (ORB)
1. **Define Opening Range:** First N minutes of trading (15-60 min)
2. **Wait for Breakout:** Price breaks above/below range
3. **Enter Position:** Buy breakout up, sell breakout down
4. **Set Stops:** Fixed stop loss based on strategy
5. **Target Profits:** 3:1 to 6:1 risk-reward ratios

### Advanced Filters
- **Volatility Filter:** Only trade high-volatility days
- **Volume Filter:** Require above-average volume
- **Time Filter:** Optimal trading windows
- **Trend Filter:** Market regime detection

---

## 🎯 Next Development Phase

### Immediate Goals (Next 4 Weeks)
- [ ] Complete multi-strategy portfolio system
- [ ] Implement enhanced market filtering
- [ ] Add real-time data integration
- [ ] Scale to 20+ points/day consistently

### Medium-term Goals (2-3 Months)
- [ ] Dynamic position sizing (2-3 contracts)
- [ ] Machine learning pattern recognition
- [ ] Live trading integration
- [ ] Achieve 30+ points/day target

---

## 🚨 Important Notes

### Data Files
- **CRITICAL:** Never delete files in `data/` folder
- Market data represents 6+ years of historical NQ futures
- Files are large (1GB+) but essential for backtesting

### Risk Warning
- Futures trading involves substantial risk
- Past performance doesn't guarantee future results
- Use proper risk management and position sizing
- Start with paper trading before live implementation

### System Requirements
- This system requires significant computational resources
- Backtesting can take 10-30 minutes for full analysis
- Ensure adequate disk space for data processing

---

## 📞 Support & Documentation

### Key Files for Understanding
1. **`PROJECT_SUMMARY_V1.md`** - Complete project overview
2. **`strategy_analysis_summary_2024.md`** - Performance analysis
3. **`backtesting/ultimate_orb_strategy.py`** - Core strategy code

### Performance Files
- **`optimized_trades.json`** - Generated trade data
- **`*_results_2024.json`** - Backtesting results
- **`strategy_analysis_summary_2024.md`** - Analysis summary

### Running Tests
```bash
# Test core strategy
python backtesting/ultimate_orb_strategy.py

# Test data loading
python backtesting/data_handler.py

# Run parameter optimization (long process)
python backtesting/parameter_optimization.py
```

---

## 🔄 Version History

### Version 1.0 (Current)
- ✅ Complete ORB strategy framework
- ✅ Real data backtesting (10-15 pts/day achieved)
- ✅ Advanced filtering and optimization
- ✅ Multi-strategy portfolio approach
- ✅ Comprehensive performance analysis

### Planned Version 2.0
- 🎯 Real-time data integration
- 🎯 Live trading capabilities
- 🎯 Machine learning enhancements
- 🎯 30+ points/day achievement

---

## 💡 Quick Tips

### Best Practices
- Always test on historical data first
- Use proper risk management (never risk more than 2-5% per trade)
- Monitor performance daily and adjust as needed
- Keep detailed trading logs for analysis

### Common Issues
- **Slow backtesting:** Large data files require patience
- **Memory usage:** Close other applications during testing
- **Data errors:** Verify data files are intact and properly formatted

### Performance Optimization
- Use SSD storage for faster data access
- Increase RAM for better performance
- Run tests during off-peak hours
- Consider cloud computing for intensive backtesting

---

**🎯 Current Status:** Active development targeting 30+ points/day
**📊 Next Milestone:** Complete multi-strategy system and achieve 20+ points/day
**🚀 Ultimate Goal:** Consistent 30-50+ points/day with robust risk management

---

*Last Updated: September 2024*
*Status: Ready for Version 1.0 Git Commit*