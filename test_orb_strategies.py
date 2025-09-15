#!/usr/bin/env python3
"""
ORB (Opening Range Breakout) Strategy Testing Suite
==================================================
Comprehensive testing of multiple ORB strategy variations with proper constraints
"""

import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from backtesting.ultra_fast.claude_compliant_engine import ClaudeCompliantEngine


def create_orb_strategy_suite():
    """Create comprehensive ORB strategy variations."""

    return [
        # === TIME-BASED VARIATIONS ===
        {
            'name': 'ORB_15min_Quick',
            'type': 'orb_breakout',
            'orb_minutes': 15,
            'stop_loss': 8,
            'profit_target': 16,
            'description': '15-minute range, 2:1 R/R'
        },
        {
            'name': 'ORB_30min_Standard',
            'type': 'orb_breakout',
            'orb_minutes': 30,
            'stop_loss': 10,
            'profit_target': 20,
            'description': '30-minute range, 2:1 R/R'
        },
        {
            'name': 'ORB_45min_Extended',
            'type': 'orb_breakout',
            'orb_minutes': 45,
            'stop_loss': 12,
            'profit_target': 24,
            'description': '45-minute range, 2:1 R/R'
        },
        {
            'name': 'ORB_60min_Full',
            'type': 'orb_breakout',
            'orb_minutes': 60,
            'stop_loss': 15,
            'profit_target': 30,
            'description': '60-minute range, 2:1 R/R'
        },

        # === RISK/REWARD VARIATIONS ===
        {
            'name': 'ORB_30min_Aggressive_1to1',
            'type': 'orb_breakout',
            'orb_minutes': 30,
            'stop_loss': 15,
            'profit_target': 15,
            'description': '30min range, 1:1 R/R (aggressive)'
        },
        {
            'name': 'ORB_30min_Conservative_3to1',
            'type': 'orb_breakout',
            'orb_minutes': 30,
            'stop_loss': 8,
            'profit_target': 24,
            'description': '30min range, 3:1 R/R (conservative)'
        },
        {
            'name': 'ORB_30min_Ultra_Conservative_4to1',
            'type': 'orb_breakout',
            'orb_minutes': 30,
            'stop_loss': 6,
            'profit_target': 24,
            'description': '30min range, 4:1 R/R (ultra conservative)'
        },

        # === TIGHT STOP VARIATIONS ===
        {
            'name': 'ORB_30min_TightStop_5pt',
            'type': 'orb_breakout',
            'orb_minutes': 30,
            'stop_loss': 5,
            'profit_target': 15,
            'description': '30min range, very tight 5pt stop'
        },
        {
            'name': 'ORB_45min_TightStop_7pt',
            'type': 'orb_breakout',
            'orb_minutes': 45,
            'stop_loss': 7,
            'profit_target': 21,
            'description': '45min range, tight 7pt stop'
        },

        # === WIDE STOP VARIATIONS ===
        {
            'name': 'ORB_30min_WideStop_20pt',
            'type': 'orb_breakout',
            'orb_minutes': 30,
            'stop_loss': 20,
            'profit_target': 40,
            'description': '30min range, wide 20pt stop'
        },
        {
            'name': 'ORB_45min_WideStop_25pt',
            'type': 'orb_breakout',
            'orb_minutes': 45,
            'stop_loss': 25,
            'profit_target': 50,
            'description': '45min range, wide 25pt stop'
        },

        # === ASYMMETRIC VARIATIONS ===
        {
            'name': 'ORB_30min_Asymmetric_Small_Big',
            'type': 'orb_breakout',
            'orb_minutes': 30,
            'stop_loss': 5,
            'profit_target': 25,
            'description': '30min range, 5:1 asymmetric (small risk, big reward)'
        },
        {
            'name': 'ORB_45min_Asymmetric_Med_Large',
            'type': 'orb_breakout',
            'orb_minutes': 45,
            'stop_loss': 10,
            'profit_target': 35,
            'description': '45min range, 3.5:1 asymmetric'
        }
    ]


def analyze_orb_results(results):
    """Analyze and categorize ORB strategy results."""

    categories = {
        'best_expectancy': None,
        'best_profit_factor': None,
        'best_win_rate': None,
        'lowest_drawdown': None,
        'most_trades': None,
        'best_sharpe': None
    }

    best_values = {
        'expectancy': -float('inf'),
        'profit_factor': 0,
        'win_rate': 0,
        'drawdown': float('inf'),
        'trades': 0,
        'sharpe': -float('inf')
    }

    for name, result in results.items():
        if result['expectancy_points'] > best_values['expectancy']:
            best_values['expectancy'] = result['expectancy_points']
            categories['best_expectancy'] = name

        if result['profit_factor'] > best_values['profit_factor']:
            best_values['profit_factor'] = result['profit_factor']
            categories['best_profit_factor'] = name

        if result['win_rate_percent'] > best_values['win_rate']:
            best_values['win_rate'] = result['win_rate_percent']
            categories['best_win_rate'] = name

        if result['max_drawdown_points'] < best_values['drawdown']:
            best_values['drawdown'] = result['max_drawdown_points']
            categories['lowest_drawdown'] = name

        if result['total_trades'] > best_values['trades']:
            best_values['trades'] = result['total_trades']
            categories['most_trades'] = name

        if result['sharpe_ratio'] > best_values['sharpe']:
            best_values['sharpe'] = result['sharpe_ratio']
            categories['best_sharpe'] = name

    return categories, best_values


def test_orb_strategies():
    """Test comprehensive ORB strategy suite."""

    print("🎯 ORB STRATEGY TESTING SUITE")
    print("Testing multiple Opening Range Breakout variations")
    print("=" * 60)

    # Data path
    data_path = "data_parquet/nq/1min/year=2024"

    if not Path(data_path).exists():
        print(f"❌ Data not found: {data_path}")
        return False

    try:
        # Initialize engine
        engine = ClaudeCompliantEngine()

        # Prepare data
        print("\n📊 Preparing data...")
        start_time = time.time()
        data = engine.prepare_data(data_path)
        prep_time = time.time() - start_time

        # Get ORB strategies
        orb_strategies = create_orb_strategy_suite()

        print(f"\n⚡ Testing {len(orb_strategies)} ORB strategy variations...")
        print("   All strategies use constraints: max 3 trades/day, one active trade")

        # Run all ORB strategies
        exec_start = time.time()
        results = {}

        for i, strategy in enumerate(orb_strategies, 1):
            print(f"\n  📈 [{i}/{len(orb_strategies)}] {strategy['name']}...")
            print(f"      {strategy['description']}")

            result = engine.backtest_strategy(data, strategy)
            results[strategy['name']] = result

            # Quick summary
            print(f"      Trades: {result['total_trades']}, "
                  f"Win Rate: {result['win_rate_percent']:.1f}%, "
                  f"P&L: ${result['net_profit_usd']:,.0f}")

        exec_time = time.time() - exec_start
        total_time = prep_time + exec_time

        # Analyze results
        print(f"\n🏆 ORB STRATEGY ANALYSIS")
        print("=" * 50)

        categories, best_values = analyze_orb_results(results)

        print(f"🥇 BEST PERFORMERS:")
        print(f"   Best Expectancy: {categories['best_expectancy']} ({best_values['expectancy']:.2f} pts/trade)")
        print(f"   Best Profit Factor: {categories['best_profit_factor']} ({best_values['profit_factor']:.2f})")
        print(f"   Best Win Rate: {categories['best_win_rate']} ({best_values['win_rate']:.1f}%)")
        print(f"   Lowest Drawdown: {categories['lowest_drawdown']} ({best_values['drawdown']:.1f} pts)")
        print(f"   Best Sharpe Ratio: {categories['best_sharpe']} ({best_values['sharpe']:.2f})")
        print(f"   Most Active: {categories['most_trades']} ({best_values['trades']} trades)")

        # Detailed results table
        print(f"\n📊 DETAILED ORB RESULTS:")
        print(f"{'Strategy':<30} {'Trades':<7} {'Win%':<6} {'Expect':<8} {'PF':<6} {'P&L ($)':<10} {'DD (pts)':<8}")
        print("-" * 85)

        # Sort by expectancy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['expectancy_points'], reverse=True)

        for name, result in sorted_results:
            print(f"{name:<30} {result['total_trades']:<7} "
                  f"{result['win_rate_percent']:<6.1f} "
                  f"{result['expectancy_points']:<8.2f} "
                  f"{result['profit_factor']:<6.2f} "
                  f"{result['net_profit_usd']:<10,.0f} "
                  f"{result['max_drawdown_points']:<8.1f}")

        # Performance summary
        print(f"\n⚡ PERFORMANCE SUMMARY:")
        print(f"   Data preparation: {prep_time:.3f}s")
        print(f"   Strategy execution: {exec_time:.3f}s")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Strategies tested: {len(orb_strategies)}")
        print(f"   Avg trades per strategy: {sum(r['total_trades'] for r in results.values()) / len(results):.0f}")

        # Save detailed results
        detailed_results = {
            'test_info': {
                'test_type': 'ORB_Strategy_Suite',
                'timestamp': time.time(),
                'strategies_tested': len(orb_strategies),
                'data_source': data_path
            },
            'performance': {
                'prep_time': prep_time,
                'exec_time': exec_time,
                'total_time': total_time
            },
            'best_performers': categories,
            'best_values': best_values,
            'strategy_configs': orb_strategies,
            'detailed_results': results
        }

        results_file = Path("orb_strategy_test_results.json")
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)

        print(f"\n📄 Detailed results saved to: {results_file}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_orb_strategies()
    print(f"\n{'🎉 ORB TESTING COMPLETE!' if success else '❌ ORB TESTING FAILED'}")
    sys.exit(0 if success else 1)