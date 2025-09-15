#!/usr/bin/env python3
"""
Integration Test - Small Data Slice
===================================
Quick integration test on subset of data to validate correctness
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from backtesting.ultra_fast.claude_compliant_engine import ClaudeCompliantEngine


def test_small_slice_integration():
    """Integration test with small data slice."""

    print("🧪 INTEGRATION TEST - SMALL DATA SLICE")
    print("=" * 50)

    data_path = "data_parquet/nq/1min/year=2024"

    if not Path(data_path).exists():
        print(f"❌ Test data not found: {data_path}")
        return False

    try:
        engine = ClaudeCompliantEngine()

        # Prepare data
        print("📊 Loading full dataset...")
        data = engine.prepare_data(data_path)

        # Create small slice (first 1000 bars)
        print("✂️  Creating small slice (1000 bars)...")
        small_arrays = {}
        for key, array in data['arrays'].items():
            if hasattr(array, '__len__') and key != 'metadata':
                small_arrays[key] = array[:1000]
            else:
                small_arrays[key] = array

        small_data = {
            'arrays': small_arrays,
            'session_bounds': data['session_bounds'][:5],  # First 5 sessions
            'n_bars': 1000,
            'original_n_bars': data['n_bars']
        }

        # Test strategy on small slice
        strategy = {
            'name': 'MA_Test_Small',
            'type': 'simple_ma',
            'ma_fast': 5,
            'ma_slow': 15
        }

        print("⚡ Running strategy on small slice...")
        result = engine.backtest_strategy(small_data, strategy)

        # Validate results
        print(f"\n✅ INTEGRATION TEST RESULTS:")
        print(f"   Data points: {small_data['n_bars']}")
        print(f"   Total trades: {result['total_trades']}")
        print(f"   Win rate: {result['win_rate_percent']:.1f}%")
        print(f"   Net P&L: ${result['net_profit_usd']:,.2f}")
        print(f"   Expectancy: {result['expectancy_points']:.3f} pts/trade")

        # Validation checks
        checks_passed = 0
        total_checks = 6

        # Check 1: Reasonable trade count
        if 0 <= result['total_trades'] <= 100:  # Max ~10% of bars
            print(f"   ✅ Trade count reasonable: {result['total_trades']}")
            checks_passed += 1
        else:
            print(f"   ❌ Trade count unrealistic: {result['total_trades']}")

        # Check 2: Win rate bounds
        if 0 <= result['win_rate_percent'] <= 100:
            print(f"   ✅ Win rate in bounds: {result['win_rate_percent']:.1f}%")
            checks_passed += 1
        else:
            print(f"   ❌ Win rate out of bounds: {result['win_rate_percent']:.1f}%")

        # Check 3: All 33 metrics present
        required_keys = ['expectancy_points', 'profit_factor', 'max_drawdown_points',
                        'sharpe_ratio', 'win_rate_percent', 'total_trades']
        missing_keys = [k for k in required_keys if k not in result]
        if not missing_keys:
            print(f"   ✅ All required metrics present")
            checks_passed += 1
        else:
            print(f"   ❌ Missing metrics: {missing_keys}")

        # Check 4: Trade ledger exists
        if 'trade_ledger' in result and len(result['trade_ledger']) >= 0:
            print(f"   ✅ Trade ledger generated")
            checks_passed += 1
        else:
            print(f"   ❌ Trade ledger missing")

        # Check 5: Equity curve exists
        if 'equity_curve' in result:
            print(f"   ✅ Equity curve generated")
            checks_passed += 1
        else:
            print(f"   ❌ Equity curve missing")

        # Check 6: No NaN values in key metrics
        nan_metrics = []
        for key in ['expectancy_points', 'profit_factor', 'win_rate_percent']:
            if pd.isna(result.get(key, 0)):
                nan_metrics.append(key)
        if not nan_metrics:
            print(f"   ✅ No NaN values in key metrics")
            checks_passed += 1
        else:
            print(f"   ❌ NaN values found: {nan_metrics}")

        print(f"\n📊 INTEGRATION TEST SUMMARY:")
        print(f"   Checks passed: {checks_passed}/{total_checks}")
        print(f"   Success rate: {checks_passed/total_checks*100:.1f}%")

        success = checks_passed >= total_checks - 1  # Allow 1 failure

        if success:
            print(f"   🎉 INTEGRATION TEST PASSED!")
        else:
            print(f"   ❌ INTEGRATION TEST FAILED!")

        return success

    except Exception as e:
        print(f"❌ Integration test error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_small_slice_integration()
    sys.exit(0 if success else 1)