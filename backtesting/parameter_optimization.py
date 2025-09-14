"""
Parameter Optimization Matrix - Target 15-30+ Points/Day
=======================================================
Comprehensive parameter optimization for ORB strategies to achieve
15-30+ points per day with maximum 3 trades and 1 contract.

Tests all combinations of:
- ORB periods: 5, 15, 30, 45, 60, 90 minutes
- Stop losses: 10, 15, 20, 25, 30, 40, 50 points
- R:R ratios: 2.0, 3.0, 4.0, 5.0, 6.0, 7.0
- Max trades: 1, 2, 3 per day
- Target points: 15, 20, 25, 30+ per day
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent))

from main import BacktestEngine
from regime_optimized_orb import RegimeOptimizedORB, MarketRegime

@dataclass
class OptimizationResult:
    """Results from parameter optimization"""
    params: Dict[str, Any]
    total_days: int
    profitable_days: int
    total_pnl_points: float
    avg_daily_pnl: float
    max_daily_pnl: float
    min_daily_pnl: float
    daily_target_achievement: float
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    regime_performance: Dict[str, float]
    daily_pnl_series: List[float]

class ParameterOptimizer:
    """Parameter optimization engine for ORB strategies"""

    def __init__(self, data_path: str, optimization_target: str = "daily_points"):
        """
        Initialize optimizer

        Args:
            data_path: Path to market data file
            optimization_target: Target metric ('daily_points', 'total_return', 'sharpe')
        """
        self.data_path = data_path
        self.optimization_target = optimization_target
        self.results = []

        # Define parameter space
        self.parameter_space = {
            'or_minutes': [5, 15, 30, 45, 60, 90],
            'stop_points': [10, 15, 20, 25, 30, 40, 50],
            'rr_ratio': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'max_trades_per_day': [1, 2, 3],
            'target_points_per_day': [15, 20, 25, 30, 35]
        }

        # Advanced parameter combinations
        self.regime_specific_params = {
            'trending': {'or_minutes': [15, 30], 'stop_points': [20, 25, 30], 'rr_ratio': [5.0, 6.0, 7.0]},
            'ranging': {'or_minutes': [30, 45, 60], 'stop_points': [15, 20], 'rr_ratio': [3.0, 4.0]},
            'high_vol': {'or_minutes': [30, 45], 'stop_points': [30, 40, 50], 'rr_ratio': [6.0, 7.0]},
            'low_vol': {'or_minutes': [45, 60, 90], 'stop_points': [10, 15], 'rr_ratio': [4.0, 5.0]}
        }

    def optimize_parameters(self, years: List[int] = [2020, 2021, 2022, 2023, 2024, 2025]) -> List[OptimizationResult]:
        """
        Run comprehensive parameter optimization

        Args:
            years: Years to test on

        Returns:
            List of optimization results sorted by performance
        """
        print("🔄 STARTING PARAMETER OPTIMIZATION FOR 15-30+ POINTS/DAY")
        print("=" * 70)
        print(f"📊 Testing {len(list(self._generate_parameter_combinations()))} parameter combinations")
        print(f"📅 Years: {', '.join(map(str, years))}")
        print(f"🎯 Target: 15-30+ points per day with max 3 trades")
        print()

        total_combinations = len(list(self._generate_parameter_combinations()))
        current_combination = 0

        best_results = []

        for params in self._generate_parameter_combinations():
            current_combination += 1

            if current_combination % 50 == 0 or current_combination <= 10:
                progress = (current_combination / total_combinations) * 100
                print(f"🔍 Testing combination {current_combination}/{total_combinations} ({progress:.1f}%)")
                print(f"   OR: {params['or_minutes']}min, Stop: {params['stop_points']}pts, RR: {params['rr_ratio']}, MaxTrades: {params['max_trades_per_day']}, Target: {params['target_points_per_day']}")

            try:
                result = self._test_parameter_combination(params, years)
                if result:
                    self.results.append(result)

                    # Track best results for each target level
                    if result.avg_daily_pnl >= 15:  # Minimum target
                        best_results.append(result)

                        if result.avg_daily_pnl >= 20:
                            print(f"   ✅ PROMISING: {result.avg_daily_pnl:.1f} pts/day, Target Achievement: {result.daily_target_achievement:.1f}%")

            except Exception as e:
                if current_combination <= 10:  # Show errors for first few combinations
                    print(f"   ❌ Error: {str(e)}")
                continue

        # Sort results by performance
        self.results.sort(key=lambda x: self._calculate_fitness_score(x), reverse=True)

        print(f"\n✅ OPTIMIZATION COMPLETE")
        print(f"📊 Total combinations tested: {len(self.results)}")
        print(f"🏆 Results achieving 15+ pts/day: {len(best_results)}")

        # Show top 10 results
        print(f"\n🥇 TOP 10 PARAMETER COMBINATIONS:")
        print("-" * 100)
        for i, result in enumerate(self.results[:10]):
            params = result.params
            fitness = self._calculate_fitness_score(result)
            print(f"{i+1:2d}. OR:{params['or_minutes']:2d}min Stop:{params['stop_points']:2.0f} RR:{params['rr_ratio']:.1f} MaxTrades:{params['max_trades_per_day']} "
                  f"→ {result.avg_daily_pnl:6.1f} pts/day ({result.daily_target_achievement:5.1f}% target) "
                  f"Fitness:{fitness:6.1f} WR:{result.win_rate:5.1f}%")

        return self.results

    def _generate_parameter_combinations(self):
        """Generate all parameter combinations to test"""
        # Standard combinations
        for or_min in self.parameter_space['or_minutes']:
            for stop in self.parameter_space['stop_points']:
                for rr in self.parameter_space['rr_ratio']:
                    for max_trades in self.parameter_space['max_trades_per_day']:
                        for target in self.parameter_space['target_points_per_day']:
                            # Skip unrealistic combinations
                            if self._is_valid_combination(or_min, stop, rr, max_trades, target):
                                yield {
                                    'or_minutes': or_min,
                                    'stop_points': stop,
                                    'rr_ratio': rr,
                                    'max_trades_per_day': max_trades,
                                    'target_points_per_day': target
                                }

    def _is_valid_combination(self, or_min: int, stop: float, rr: float, max_trades: int, target: float) -> bool:
        """Filter out unrealistic parameter combinations"""
        # Target per trade needed
        target_per_trade = target / max_trades
        max_win_per_trade = stop * rr

        # Must be achievable with the R:R ratio
        if max_win_per_trade < target_per_trade * 0.5:  # Need at least 50% efficiency
            return False

        # Very wide stops with short OR periods don't make sense
        if or_min <= 15 and stop >= 40:
            return False

        # Very tight stops with long OR periods are suboptimal
        if or_min >= 60 and stop <= 10:
            return False

        # High targets need adequate R:R ratios
        if target >= 25 and rr < 4.0:
            return False

        # Single trade per day needs very high R:R for high targets
        if max_trades == 1 and target >= 20 and rr < 5.0:
            return False

        return True

    def _test_parameter_combination(self, params: Dict[str, Any], years: List[int]) -> OptimizationResult:
        """Test a specific parameter combination"""

        # Create strategy with these parameters
        strategy = RegimeOptimizedORB(
            target_points_per_day=params['target_points_per_day'],
            max_trades_per_day=params['max_trades_per_day']
        )

        # Override regime parameters for testing
        for regime in strategy.regime_params:
            strategy.regime_params[regime].or_minutes = params['or_minutes']
            strategy.regime_params[regime].stop_points = params['stop_points']
            strategy.regime_params[regime].rr_ratio = params['rr_ratio']
            strategy.regime_params[regime].max_trades = params['max_trades_per_day']

        all_trades = []
        daily_pnl = {}
        regime_performance = {}

        for year in years:
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"

            engine = BacktestEngine(
                data_path=self.data_path,
                strategy=strategy,
                initial_capital=100000.0,
                point_value=20.0,
                commission_per_trade=2.50,
                timeframe="1min",
                start_date=start_date,
                end_date=end_date
            )

            # Run backtest
            results = engine.run_backtest(verbose=False)

            if 'error' in results:
                continue

            # Get trades
            trades_df = engine.get_trades_dataframe()
            if not trades_df.empty:
                all_trades.extend(trades_df.to_dict('records'))

        if not all_trades:
            return None

        # Calculate performance metrics
        trades_df = pd.DataFrame(all_trades)

        # Calculate daily P&L
        trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date
        daily_trades = trades_df.groupby('date')['pnl_points'].sum().reset_index()
        daily_pnl_values = daily_trades['pnl_points'].tolist()

        total_days = len(daily_pnl_values)
        profitable_days = len([pnl for pnl in daily_pnl_values if pnl > 0])
        total_pnl_points = sum(daily_pnl_values)
        avg_daily_pnl = total_pnl_points / total_days if total_days > 0 else 0

        # Target achievement
        target_achievement = len([pnl for pnl in daily_pnl_values if pnl >= params['target_points_per_day']]) / total_days * 100 if total_days > 0 else 0

        # Trading metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_points'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Profit factor
        gross_profit = trades_df[trades_df['pnl_points'] > 0]['pnl_points'].sum()
        gross_loss = abs(trades_df[trades_df['pnl_points'] < 0]['pnl_points'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Max drawdown
        cumulative_pnl = trades_df['pnl_points'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Sharpe ratio (simplified)
        if len(daily_pnl_values) > 1:
            daily_returns = np.array(daily_pnl_values)
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        else:
            sharpe_ratio = 0

        return OptimizationResult(
            params=params,
            total_days=total_days,
            profitable_days=profitable_days,
            total_pnl_points=total_pnl_points,
            avg_daily_pnl=avg_daily_pnl,
            max_daily_pnl=max(daily_pnl_values) if daily_pnl_values else 0,
            min_daily_pnl=min(daily_pnl_values) if daily_pnl_values else 0,
            daily_target_achievement=target_achievement,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            regime_performance={},
            daily_pnl_series=daily_pnl_values
        )

    def _calculate_fitness_score(self, result: OptimizationResult) -> float:
        """Calculate fitness score for ranking results"""
        # Multi-objective fitness function
        daily_pnl_score = min(result.avg_daily_pnl / 30.0, 1.0) * 40  # 40% weight, capped at 30 pts/day
        target_achievement_score = result.daily_target_achievement / 100.0 * 25  # 25% weight
        win_rate_score = min(result.win_rate / 60.0, 1.0) * 15  # 15% weight, target 60% win rate
        profit_factor_score = min(result.profit_factor / 3.0, 1.0) * 10  # 10% weight, target 3.0 PF
        sharpe_score = min(result.sharpe_ratio / 2.0, 1.0) * 10  # 10% weight, target 2.0 Sharpe

        # Penalty for high drawdown
        drawdown_penalty = min(result.max_drawdown / 100.0, 1.0) * 10  # Max 10 point penalty

        total_score = (daily_pnl_score + target_achievement_score + win_rate_score +
                      profit_factor_score + sharpe_score - drawdown_penalty)

        return max(0, total_score)

    def get_best_parameters_for_target(self, target_points: float) -> List[OptimizationResult]:
        """Get best parameters for specific daily target"""
        filtered_results = [r for r in self.results if r.avg_daily_pnl >= target_points * 0.8]  # Within 80% of target
        return sorted(filtered_results, key=lambda x: self._calculate_fitness_score(x), reverse=True)[:5]

    def save_optimization_results(self, filename: str = "optimization_results.json"):
        """Save optimization results to file"""
        results_data = []

        for result in self.results[:50]:  # Save top 50 results
            result_dict = {
                'params': result.params,
                'metrics': {
                    'avg_daily_pnl': result.avg_daily_pnl,
                    'daily_target_achievement': result.daily_target_achievement,
                    'total_trades': result.total_trades,
                    'win_rate': result.win_rate,
                    'profit_factor': result.profit_factor,
                    'max_drawdown': result.max_drawdown,
                    'sharpe_ratio': result.sharpe_ratio,
                    'fitness_score': self._calculate_fitness_score(result)
                }
            }
            results_data.append(result_dict)

        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"💾 Optimization results saved to {filename}")

    def create_optimized_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Create optimized strategy configurations"""
        strategies = {}

        # Best overall strategy
        if self.results:
            best_overall = self.results[0]
            strategies['Ultimate_Champion'] = {
                'class': 'RegimeOptimizedORB',
                'params': best_overall.params,
                'expected_daily_pnl': best_overall.avg_daily_pnl,
                'target_achievement': best_overall.daily_target_achievement,
                'description': f'Optimized for {best_overall.avg_daily_pnl:.1f} pts/day with {best_overall.daily_target_achievement:.1f}% target achievement'
            }

        # Strategy for different targets
        for target in [15, 20, 25, 30]:
            best_for_target = self.get_best_parameters_for_target(target)
            if best_for_target:
                result = best_for_target[0]
                strategies[f'Target_{target}_Points'] = {
                    'class': 'RegimeOptimizedORB',
                    'params': result.params,
                    'expected_daily_pnl': result.avg_daily_pnl,
                    'target_achievement': result.daily_target_achievement,
                    'description': f'Optimized for {target}+ pts/day target'
                }

        return strategies

def run_parameter_optimization():
    """Main optimization runner"""
    data_path = "/Users/shubhamshanker/bt_/data/NQ_M1_standard.csv"

    optimizer = ParameterOptimizer(data_path)
    results = optimizer.optimize_parameters()

    # Save results
    optimizer.save_optimization_results("orb_optimization_results.json")

    # Create optimized strategies
    optimized_strategies = optimizer.create_optimized_strategies()

    with open("optimized_strategies.json", 'w') as f:
        json.dump(optimized_strategies, f, indent=2)

    print(f"\n🎉 OPTIMIZATION COMPLETE!")
    print(f"📊 Created {len(optimized_strategies)} optimized strategy configurations")
    print("📁 Results saved to orb_optimization_results.json and optimized_strategies.json")

    return results, optimized_strategies

if __name__ == "__main__":
    results, strategies = run_parameter_optimization()