#!/usr/bin/env python3
"""
Migration Validation Script
===========================
Comprehensive validation of CSV → Parquet migration:
- Data integrity comparison
- Performance benchmarking
- Backtest result validation
- Timezone accuracy checks
"""

import pandas as pd
import numpy as np
import time
import sys
import json
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Any, Tuple
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.data_config import get_data_config, get_data_path
from backtesting.data_handler import DataHandler

# Try to import Parquet handler
try:
    from backtesting.parquet_data_handler import ParquetDataHandler
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MigrationValidator:
    """Validates CSV to Parquet migration integrity and performance."""

    def __init__(self):
        self.config = get_data_config()
        self.validation_report = {
            'timestamp': datetime.now().isoformat(),
            'parquet_available': PARQUET_AVAILABLE,
            'tests': {}
        }

    def test_data_availability(self) -> Dict[str, Any]:
        """Test that both CSV and Parquet data sources are available."""
        logger.info("🔍 Testing data source availability...")

        test_result = {
            'csv_available': False,
            'parquet_available': False,
            'csv_files': {},
            'parquet_timeframes': [],
            'issues': []
        }

        # Test CSV availability
        try:
            csv_root = Path(self.config.get_data_path('csv'))
            test_result['csv_available'] = csv_root.exists()

            for timeframe in ['1min', '3min', '5min', '15min']:
                csv_file = Path(self.config.get_csv_file_path(timeframe))
                test_result['csv_files'][timeframe] = csv_file.exists()
                if not csv_file.exists():
                    test_result['issues'].append(f"Missing CSV file: {csv_file}")

        except Exception as e:
            test_result['issues'].append(f"CSV availability check failed: {e}")

        # Test Parquet availability
        try:
            if PARQUET_AVAILABLE:
                parquet_root = Path(self.config.get_data_path('parquet'))
                test_result['parquet_available'] = parquet_root.exists()

                if parquet_root.exists():
                    nq_dir = parquet_root / 'nq'
                    if nq_dir.exists():
                        test_result['parquet_timeframes'] = [
                            d.name for d in nq_dir.iterdir() if d.is_dir()
                        ]
                    else:
                        test_result['issues'].append("Parquet NQ directory not found")
                else:
                    test_result['issues'].append(f"Parquet root not found: {parquet_root}")
            else:
                test_result['issues'].append("Parquet support not available (missing dependencies)")

        except Exception as e:
            test_result['issues'].append(f"Parquet availability check failed: {e}")

        test_result['passed'] = (
            test_result['csv_available'] and
            test_result['parquet_available'] and
            len(test_result['issues']) == 0
        )

        logger.info(f"✅ Data availability test: {'PASSED' if test_result['passed'] else 'FAILED'}")
        return test_result

    def test_data_integrity(self, timeframe: str = "1min", sample_size: int = 10000) -> Dict[str, Any]:
        """Compare data integrity between CSV and Parquet sources."""
        logger.info(f"🔍 Testing data integrity for {timeframe}...")

        test_result = {
            'timeframe': timeframe,
            'csv_records': 0,
            'parquet_records': 0,
            'sample_comparison': {},
            'timezone_check': {},
            'issues': []
        }

        try:
            # Load CSV data
            csv_handler = DataHandler(
                data_path=self.config.get_csv_file_path(timeframe),
                timeframe=timeframe,
                use_parquet=False
            )
            csv_handler.load_data()
            csv_data = csv_handler.data.copy()
            test_result['csv_records'] = len(csv_data)

            # Load Parquet data
            if PARQUET_AVAILABLE:
                parquet_handler = ParquetDataHandler(self.config.get_data_path('parquet'))
                parquet_data = parquet_handler.load_data("NQ", timeframe)
                test_result['parquet_records'] = len(parquet_data)

                # Compare sample data
                if len(csv_data) > 0 and len(parquet_data) > 0:
                    # Take common date range for comparison
                    csv_start, csv_end = csv_data.index.min(), csv_data.index.max()
                    parquet_start, parquet_end = parquet_data.index.min(), parquet_data.index.max()

                    common_start = max(csv_start, parquet_start)
                    common_end = min(csv_end, parquet_end)

                    if common_start < common_end:
                        # Sample from common range
                        csv_sample = csv_data[common_start:common_end].head(sample_size)
                        parquet_sample = parquet_data[common_start:common_end].head(sample_size)

                        # Compare OHLCV values
                        test_result['sample_comparison'] = self._compare_ohlcv_data(
                            csv_sample, parquet_sample
                        )

                        # Timezone checks
                        test_result['timezone_check'] = self._validate_timezone_conversion(
                            csv_sample, parquet_sample
                        )
                    else:
                        test_result['issues'].append("No overlapping date range found")
                else:
                    test_result['issues'].append("Empty datasets")

            else:
                test_result['issues'].append("Parquet handler not available")

        except Exception as e:
            test_result['issues'].append(f"Data integrity test failed: {e}")

        test_result['passed'] = len(test_result['issues']) == 0
        logger.info(f"✅ Data integrity test: {'PASSED' if test_result['passed'] else 'FAILED'}")
        return test_result

    def _compare_ohlcv_data(self, csv_data: pd.DataFrame, parquet_data: pd.DataFrame) -> Dict[str, Any]:
        """Compare OHLCV data between CSV and Parquet."""
        comparison = {
            'price_correlation': {},
            'volume_correlation': 0.0,
            'timestamp_alignment': 0.0,
            'differences': []
        }

        try:
            # Align data by timestamp
            aligned_csv = csv_data.sort_index()
            aligned_parquet = parquet_data.sort_index()

            # Find common timestamps
            common_timestamps = aligned_csv.index.intersection(aligned_parquet.index)

            if len(common_timestamps) > 100:  # Need sufficient data
                csv_common = aligned_csv.loc[common_timestamps]
                parquet_common = aligned_parquet.loc[common_timestamps]

                # Compare price columns
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col in csv_common.columns and col in parquet_common.columns:
                        correlation = csv_common[col].corr(parquet_common[col])
                        comparison['price_correlation'][col] = correlation

                        # Check for significant differences
                        diff = abs(csv_common[col] - parquet_common[col])
                        max_diff = diff.max()
                        if max_diff > 0.5:  # More than 0.5 point difference
                            comparison['differences'].append(f"{col}: max diff {max_diff:.2f}")

                # Compare volume
                if 'Volume' in csv_common.columns and 'Volume' in parquet_common.columns:
                    comparison['volume_correlation'] = csv_common['Volume'].corr(parquet_common['Volume'])

                # Timestamp alignment score
                comparison['timestamp_alignment'] = len(common_timestamps) / len(aligned_csv)

            else:
                comparison['differences'].append("Insufficient common timestamps for comparison")

        except Exception as e:
            comparison['differences'].append(f"Comparison failed: {e}")

        return comparison

    def _validate_timezone_conversion(self, csv_data: pd.DataFrame, parquet_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate timezone conversion accuracy."""
        validation = {
            'csv_timezone': str(csv_data.index.tz) if hasattr(csv_data.index, 'tz') else 'None',
            'parquet_timezone': str(parquet_data.index.tz) if hasattr(parquet_data.index, 'tz') else 'None',
            'session_hours_check': {},
            'issues': []
        }

        try:
            # Check session hours filtering
            if len(parquet_data) > 0:
                # All Parquet data should be within NY session hours
                times = parquet_data.index.time
                before_session = sum(1 for t in times if t < time(9, 30))
                after_session = sum(1 for t in times if t > time(16, 0))

                validation['session_hours_check'] = {
                    'total_records': len(parquet_data),
                    'before_session': before_session,
                    'after_session': after_session,
                    'session_compliance': (before_session + after_session) == 0
                }

                if before_session > 0 or after_session > 0:
                    validation['issues'].append(f"Found {before_session + after_session} records outside session hours")

        except Exception as e:
            validation['issues'].append(f"Timezone validation failed: {e}")

        return validation

    def test_performance_comparison(self, timeframe: str = "1min") -> Dict[str, Any]:
        """Compare loading performance between CSV and Parquet."""
        logger.info(f"🏃 Testing performance for {timeframe}...")

        test_result = {
            'timeframe': timeframe,
            'csv_performance': {},
            'parquet_performance': {},
            'speedup_ratio': 0.0,
            'issues': []
        }

        try:
            # Test CSV loading performance
            start_time = time.time()
            csv_handler = DataHandler(
                data_path=self.config.get_csv_file_path(timeframe),
                timeframe=timeframe,
                use_parquet=False
            )
            csv_handler.load_data()
            csv_load_time = time.time() - start_time

            test_result['csv_performance'] = {
                'load_time_seconds': csv_load_time,
                'records_loaded': len(csv_handler.data) if csv_handler.data is not None else 0,
                'records_per_second': len(csv_handler.data) / csv_load_time if csv_load_time > 0 else 0
            }

            # Test Parquet loading performance
            if PARQUET_AVAILABLE:
                start_time = time.time()
                parquet_handler = ParquetDataHandler(self.config.get_data_path('parquet'))
                parquet_data = parquet_handler.load_data("NQ", timeframe)
                parquet_load_time = time.time() - start_time

                test_result['parquet_performance'] = {
                    'load_time_seconds': parquet_load_time,
                    'records_loaded': len(parquet_data),
                    'records_per_second': len(parquet_data) / parquet_load_time if parquet_load_time > 0 else 0
                }

                # Calculate speedup
                if csv_load_time > 0 and parquet_load_time > 0:
                    test_result['speedup_ratio'] = csv_load_time / parquet_load_time

            else:
                test_result['issues'].append("Parquet handler not available for performance test")

        except Exception as e:
            test_result['issues'].append(f"Performance test failed: {e}")

        test_result['passed'] = len(test_result['issues']) == 0
        logger.info(f"✅ Performance test: {'PASSED' if test_result['passed'] else 'FAILED'}")
        return test_result

    def test_backtest_consistency(self, timeframe: str = "1min") -> Dict[str, Any]:
        """Test that backtests produce consistent results with both data sources."""
        logger.info(f"🧪 Testing backtest consistency for {timeframe}...")

        test_result = {
            'timeframe': timeframe,
            'csv_backtest': {},
            'parquet_backtest': {},
            'consistency_check': {},
            'issues': []
        }

        try:
            # Simple test strategy for validation
            from backtesting.ultimate_orb_strategy import UltimateORBStrategy
            from backtesting.portfolio import Portfolio

            # Test with CSV data
            csv_handler = DataHandler(
                data_path=self.config.get_csv_file_path(timeframe),
                timeframe=timeframe,
                start_date="2024-01-01",
                end_date="2024-01-31",
                use_parquet=False
            )
            csv_handler.load_data()

            if csv_handler.data is not None and len(csv_handler.data) > 1000:
                csv_strategy = UltimateORBStrategy()
                csv_portfolio = Portfolio()
                csv_signals = self._run_simple_backtest(csv_handler, csv_strategy, csv_portfolio)

                test_result['csv_backtest'] = {
                    'total_signals': len(csv_signals),
                    'buy_signals': sum(1 for s in csv_signals if s.get('signal') == 'BUY'),
                    'sell_signals': sum(1 for s in csv_signals if s.get('signal') == 'SELL'),
                    'data_bars': len(csv_handler.data)
                }

            # Test with Parquet data (if available)
            if PARQUET_AVAILABLE:
                parquet_handler = DataHandler(
                    data_path=self.config.get_data_path('parquet'),
                    timeframe=timeframe,
                    start_date="2024-01-01",
                    end_date="2024-01-31",
                    use_parquet=True,
                    symbol="NQ"
                )
                parquet_handler.load_data()

                if parquet_handler.data is not None and len(parquet_handler.data) > 1000:
                    parquet_strategy = UltimateORBStrategy()
                    parquet_portfolio = Portfolio()
                    parquet_signals = self._run_simple_backtest(parquet_handler, parquet_strategy, parquet_portfolio)

                    test_result['parquet_backtest'] = {
                        'total_signals': len(parquet_signals),
                        'buy_signals': sum(1 for s in parquet_signals if s.get('signal') == 'BUY'),
                        'sell_signals': sum(1 for s in parquet_signals if s.get('signal') == 'SELL'),
                        'data_bars': len(parquet_handler.data)
                    }

                    # Compare results
                    csv_total = test_result['csv_backtest']['total_signals']
                    parquet_total = test_result['parquet_backtest']['total_signals']

                    if csv_total > 0 and parquet_total > 0:
                        signal_ratio = parquet_total / csv_total
                        test_result['consistency_check'] = {
                            'signal_ratio': signal_ratio,
                            'consistent': 0.8 <= signal_ratio <= 1.2,  # Within 20%
                            'csv_signals': csv_total,
                            'parquet_signals': parquet_total
                        }

                        if not test_result['consistency_check']['consistent']:
                            test_result['issues'].append(f"Signal count difference: CSV {csv_total} vs Parquet {parquet_total}")

            else:
                test_result['issues'].append("Parquet not available for backtest consistency test")

        except Exception as e:
            test_result['issues'].append(f"Backtest consistency test failed: {e}")

        test_result['passed'] = len(test_result['issues']) == 0
        logger.info(f"✅ Backtest consistency test: {'PASSED' if test_result['passed'] else 'FAILED'}")
        return test_result

    def _run_simple_backtest(self, data_handler: DataHandler, strategy, portfolio) -> List[Dict]:
        """Run a simple backtest for validation purposes."""
        signals = []
        data_handler.reset()

        # Process first 1000 bars for quick validation
        for i, bar in enumerate(data_handler):
            if i >= 1000:  # Limit for validation
                break

            signal = strategy.generate_signal(bar, portfolio)
            if signal.get('signal') in ['BUY', 'SELL']:
                signals.append(signal)

        return signals

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests."""
        logger.info("🚀 Starting comprehensive migration validation...")

        self.validation_report['tests']['availability'] = self.test_data_availability()

        if self.validation_report['tests']['availability']['passed']:
            # Only run other tests if data is available
            self.validation_report['tests']['integrity_1min'] = self.test_data_integrity('1min')
            self.validation_report['tests']['performance_1min'] = self.test_performance_comparison('1min')
            self.validation_report['tests']['backtest_consistency'] = self.test_backtest_consistency('1min')

            # Test additional timeframes if available
            for timeframe in ['5min', '15min']:
                if timeframe in self.validation_report['tests']['availability']['csv_files']:
                    self.validation_report['tests'][f'integrity_{timeframe}'] = self.test_data_integrity(timeframe)

        # Calculate overall status
        all_tests = [test for test in self.validation_report['tests'].values()]
        passed_tests = [test for test in all_tests if test.get('passed', False)]

        self.validation_report['summary'] = {
            'total_tests': len(all_tests),
            'passed_tests': len(passed_tests),
            'success_rate': len(passed_tests) / len(all_tests) if all_tests else 0,
            'overall_status': 'PASSED' if len(passed_tests) == len(all_tests) else 'FAILED'
        }

        return self.validation_report

    def save_report(self, output_file: str) -> None:
        """Save validation report to file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.validation_report, f, indent=2, default=str)
            logger.info(f"📋 Validation report saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

def main():
    """Main validation function."""
    print("🧪 Migration Validation Suite")
    print("=" * 50)

    # Initialize validator
    validator = MigrationValidator()

    # Run all tests
    report = validator.run_all_tests()

    # Print summary
    summary = report['summary']
    print(f"\n📊 Validation Summary:")
    print(f"Total tests: {summary['total_tests']}")
    print(f"Passed tests: {summary['passed_tests']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Overall status: {summary['overall_status']}")

    # Show test details
    print(f"\n📋 Test Results:")
    for test_name, test_result in report['tests'].items():
        status = "✅ PASSED" if test_result.get('passed', False) else "❌ FAILED"
        print(f"  {test_name}: {status}")

        if test_result.get('issues'):
            for issue in test_result['issues'][:3]:  # Show first 3 issues
                print(f"    - {issue}")

    # Performance results
    if 'performance_1min' in report['tests']:
        perf = report['tests']['performance_1min']
        if 'speedup_ratio' in perf and perf['speedup_ratio'] > 0:
            print(f"\n⚡ Performance: Parquet is {perf['speedup_ratio']:.1f}x faster than CSV")

    # Save report
    output_file = "/Users/shubhamshanker/bt_/validation_report.json"
    validator.save_report(output_file)

if __name__ == "__main__":
    main()