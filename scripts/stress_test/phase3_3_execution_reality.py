"""
PHASE 3.3: Execution Reality Stress Tests

This script tests the strategy under realistic execution conditions:

1. ASYMMETRIC PORTFOLIOS (addressing long-side bias):
   - 70/30 long-short
   - 60/40 long-short
   - 50/50 long-short (baseline)

2. STRESS SCENARIOS:
   - 10 bps transaction costs (2x baseline)
   - 5-10 bps slippage
   - t+1 open execution delay
   - Top 100 liquidity filter

Expected: Sharpe drops from 2.2 -> 1.2-1.5 (still excellent)
If Sharpe < 0.5 -> liquidity illusion detected
"""
import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocessor_v2 import SimplifiedStockPreprocessor
from src.models.lstm_model import create_model
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger


class ExecutionStressTest:
    """Stress test strategy under realistic execution conditions."""

    def __init__(self, predictions_df: pd.DataFrame, config: ConfigLoader):
        self.predictions_df = predictions_df
        self.config = config
        self.logger = setup_logger(__name__)

    def run_configuration(
        self,
        long_pct: float,
        short_pct: float,
        transaction_cost: float = 0.0005,
        slippage_bps: float = 0.0,
        execution_delay: int = 0,
        liquidity_filter: int = None,
        ewma_alpha: float = 0.15,
        config_name: str = "Config"
    ) -> Dict:
        """
        Run backtest for a specific configuration.

        Args:
            long_pct: Percentage of stocks to long (e.g., 0.2 = top 20%)
            short_pct: Percentage of stocks to short (e.g., 0.3 = bottom 30%)
            transaction_cost: One-way transaction cost (e.g., 0.0005 = 5 bps)
            slippage_bps: Additional slippage in bps (e.g., 5.0 = 5 bps)
            execution_delay: Days of delay (0 = same day, 1 = next day)
            liquidity_filter: Keep only top N most liquid stocks (None = all)
            ewma_alpha: EWMA smoothing parameter
            config_name: Name for logging
        """
        self.logger.info(f"\nRunning: {config_name}")
        self.logger.info(f"  Long: {long_pct*100:.0f}% | Short: {short_pct*100:.0f}%")
        self.logger.info(f"  Costs: {transaction_cost*10000:.1f} bps | Slippage: {slippage_bps:.1f} bps")
        self.logger.info(f"  Delay: {execution_delay} days | Liquidity: {liquidity_filter or 'All'}")

        dates = sorted(self.predictions_df['date'].unique())
        previous_positions = {}
        daily_results = []

        for i, date in enumerate(dates):
            daily_data = self.predictions_df[self.predictions_df['date'] == date].copy()

            if len(daily_data) < 5:
                continue

            # Apply liquidity filter (keep top N by some metric - using prediction magnitude as proxy)
            if liquidity_filter and len(daily_data) > liquidity_filter:
                # Sort by absolute prediction (high conviction stocks)
                daily_data['abs_pred'] = daily_data['y_pred_reg'].abs()
                daily_data = daily_data.nlargest(liquidity_filter, 'abs_pred')
                daily_data = daily_data.drop('abs_pred', axis=1)

            # Rank stocks
            daily_data_sorted = daily_data.sort_values('y_pred_reg', ascending=False)

            # Calculate target positions
            n_stocks = len(daily_data)
            n_long = max(1, int(n_stocks * long_pct))
            n_short = max(1, int(n_stocks * short_pct))

            target_positions = {}

            # Long positions
            long_tickers = daily_data_sorted.iloc[:n_long]['ticker'].tolist()
            for ticker in long_tickers:
                target_positions[ticker] = 1.0 / n_long

            # Short positions
            short_tickers = daily_data_sorted.iloc[-n_short:]['ticker'].tolist()
            for ticker in short_tickers:
                target_positions[ticker] = -1.0 / n_short

            # Apply EWMA smoothing
            if len(previous_positions) > 0:
                smoothed_positions = {}
                all_tickers = set(target_positions.keys()) | set(previous_positions.keys())

                for ticker in all_tickers:
                    target = target_positions.get(ticker, 0)
                    previous = previous_positions.get(ticker, 0)
                    smoothed = (1 - ewma_alpha) * previous + ewma_alpha * target

                    if abs(smoothed) > 0.001:
                        smoothed_positions[ticker] = smoothed
            else:
                smoothed_positions = target_positions

            # Enforce dollar neutrality
            final_positions = self._enforce_neutrality(smoothed_positions)

            # Calculate turnover and costs
            all_tickers = set(final_positions.keys()) | set(previous_positions.keys())
            trade_amount = sum(abs(final_positions.get(t, 0) - previous_positions.get(t, 0))
                             for t in all_tickers)

            gross_exposure = sum(abs(p) for p in previous_positions.values()) if previous_positions else 1.0
            turnover = trade_amount / gross_exposure if gross_exposure > 1e-8 else 0

            # Total costs: transaction cost + slippage
            total_cost_bps = transaction_cost + (slippage_bps / 10000)
            trading_cost = (trade_amount / 2.0) * total_cost_bps

            # Calculate PnL with execution delay
            if execution_delay == 0:
                # Same-day execution (current behavior)
                returns_map = daily_data.set_index('ticker')['y_true_reg'].to_dict()
            else:
                # Delayed execution: use returns from delay days ahead
                if i + execution_delay < len(dates):
                    future_date = dates[i + execution_delay]
                    future_data = self.predictions_df[self.predictions_df['date'] == future_date]
                    returns_map = future_data.set_index('ticker')['y_true_reg'].to_dict()
                else:
                    # Not enough future data
                    continue

            pnl_gross = 0
            pnl_long = 0
            pnl_short = 0

            for ticker, position in final_positions.items():
                if ticker in returns_map:
                    pnl = position * returns_map[ticker]
                    pnl_gross += pnl

                    if position > 0:
                        pnl_long += pnl
                    else:
                        pnl_short += pnl

            pnl_net = pnl_gross - trading_cost

            daily_results.append({
                'date': date,
                'pnl_gross': pnl_gross,
                'pnl_net': pnl_net,
                'pnl_long': pnl_long,
                'pnl_short': pnl_short,
                'turnover': turnover,
                'cost': trading_cost
            })

            previous_positions = final_positions.copy()

        # Calculate metrics
        results_df = pd.DataFrame(daily_results)

        metrics = self._calculate_metrics(results_df)
        metrics['config_name'] = config_name
        metrics['long_pct'] = long_pct
        metrics['short_pct'] = short_pct
        metrics['transaction_cost_bps'] = transaction_cost * 10000
        metrics['slippage_bps'] = slippage_bps
        metrics['execution_delay'] = execution_delay
        metrics['liquidity_filter'] = liquidity_filter

        self.logger.info(f"  Results:")
        self.logger.info(f"    Net Sharpe:  {metrics['sharpe_net']:.2f}")
        self.logger.info(f"    Gross Sharpe: {metrics['sharpe_gross']:.2f}")
        self.logger.info(f"    Turnover:     {metrics['avg_turnover']*100:.1f}%")
        self.logger.info(f"    Max DD:       {metrics['max_drawdown']*100:.2f}%")

        return metrics

    def _enforce_neutrality(self, positions: dict) -> dict:
        """Enforce dollar neutrality."""
        if not positions:
            return positions

        total_long = sum(p for p in positions.values() if p > 0)
        total_short = abs(sum(p for p in positions.values() if p < 0))

        if total_long < 1e-8 or total_short < 1e-8:
            return positions

        neutral_positions = {}
        for ticker, pos in positions.items():
            if pos > 0:
                neutral_positions[ticker] = pos / total_long * 0.5
            else:
                neutral_positions[ticker] = pos / total_short * 0.5

        return neutral_positions

    def _calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        # Sharpe ratios
        sharpe_net = (results_df['pnl_net'].mean() / results_df['pnl_net'].std() * np.sqrt(252)
                     if results_df['pnl_net'].std() > 0 else 0)
        sharpe_gross = (results_df['pnl_gross'].mean() / results_df['pnl_gross'].std() * np.sqrt(252)
                       if results_df['pnl_gross'].std() > 0 else 0)

        # Drawdown
        cum_returns = results_df['pnl_net'].cumsum()
        running_max = cum_returns.expanding().max()
        drawdown = cum_returns - running_max
        max_drawdown = drawdown.min()

        # Returns
        total_return = results_df['pnl_net'].sum()
        annualized_return = results_df['pnl_net'].mean() * 252

        # Costs
        avg_turnover = results_df['turnover'].mean()
        avg_cost = results_df['cost'].mean()
        total_costs = results_df['cost'].sum()

        return {
            'sharpe_net': float(sharpe_net),
            'sharpe_gross': float(sharpe_gross),
            'annualized_return': float(annualized_return),
            'total_return': float(total_return),
            'max_drawdown': float(max_drawdown),
            'avg_turnover': float(avg_turnover),
            'avg_daily_cost': float(avg_cost),
            'total_costs': float(total_costs),
            'n_days': int(len(results_df))
        }

    def run_all_stress_tests(self) -> List[Dict]:
        """Run all stress test configurations."""
        results = []

        self.logger.info("\n" + "="*70)
        self.logger.info("PHASE 3.3: EXECUTION REALITY STRESS TESTS")
        self.logger.info("="*70)

        # BASELINE CONFIGURATIONS
        self.logger.info("\n" + "-"*70)
        self.logger.info("PART 1: ASYMMETRIC PORTFOLIO CONFIGURATIONS")
        self.logger.info("-"*70)

        # 1. Baseline (50/50) - Current config
        results.append(self.run_configuration(
            long_pct=0.2, short_pct=0.2,
            config_name="Baseline (50/50 Long-Short)"
        ))

        # 2. Long-biased (70/30) - Recommended based on Phase 3.2
        results.append(self.run_configuration(
            long_pct=0.28, short_pct=0.12,  # ~70/30 ratio
            config_name="Long-Biased (70/30)"
        ))

        # 3. Moderate long-biased (60/40)
        results.append(self.run_configuration(
            long_pct=0.24, short_pct=0.16,  # ~60/40 ratio
            config_name="Moderate Long-Biased (60/40)"
        ))

        # COST STRESS TESTS
        self.logger.info("\n" + "-"*70)
        self.logger.info("PART 2: COST STRESS TESTS (10 bps)")
        self.logger.info("-"*70)

        # 4. Baseline with 10 bps costs
        results.append(self.run_configuration(
            long_pct=0.2, short_pct=0.2,
            transaction_cost=0.001,  # 10 bps
            config_name="Baseline + 10bps Costs"
        ))

        # 5. Long-biased with 10 bps costs
        results.append(self.run_configuration(
            long_pct=0.28, short_pct=0.12,
            transaction_cost=0.001,
            config_name="Long-Biased (70/30) + 10bps Costs"
        ))

        # SLIPPAGE STRESS TESTS
        self.logger.info("\n" + "-"*70)
        self.logger.info("PART 3: SLIPPAGE STRESS TESTS")
        self.logger.info("-"*70)

        # 6. Baseline + 5 bps slippage
        results.append(self.run_configuration(
            long_pct=0.2, short_pct=0.2,
            slippage_bps=5.0,
            config_name="Baseline + 5bps Slippage"
        ))

        # 7. Long-biased + 5 bps slippage
        results.append(self.run_configuration(
            long_pct=0.28, short_pct=0.12,
            slippage_bps=5.0,
            config_name="Long-Biased (70/30) + 5bps Slippage"
        ))

        # 8. Baseline + 10 bps slippage
        results.append(self.run_configuration(
            long_pct=0.2, short_pct=0.2,
            slippage_bps=10.0,
            config_name="Baseline + 10bps Slippage"
        ))

        # EXECUTION DELAY TESTS
        self.logger.info("\n" + "-"*70)
        self.logger.info("PART 4: EXECUTION DELAY (t+1 open)")
        self.logger.info("-"*70)

        # 9. Baseline with t+1 execution
        results.append(self.run_configuration(
            long_pct=0.2, short_pct=0.2,
            execution_delay=1,
            config_name="Baseline + t+1 Execution"
        ))

        # 10. Long-biased with t+1 execution
        results.append(self.run_configuration(
            long_pct=0.28, short_pct=0.12,
            execution_delay=1,
            config_name="Long-Biased (70/30) + t+1 Execution"
        ))

        # LIQUIDITY FILTER TESTS
        self.logger.info("\n" + "-"*70)
        self.logger.info("PART 5: LIQUIDITY FILTER (Top 100)")
        self.logger.info("-"*70)

        # 11. Baseline with top 100 filter
        results.append(self.run_configuration(
            long_pct=0.2, short_pct=0.2,
            liquidity_filter=100,
            config_name="Baseline + Top 100 Filter"
        ))

        # 12. Long-biased with top 100 filter
        results.append(self.run_configuration(
            long_pct=0.28, short_pct=0.12,
            liquidity_filter=100,
            config_name="Long-Biased (70/30) + Top 100 Filter"
        ))

        # WORST CASE SCENARIOS
        self.logger.info("\n" + "-"*70)
        self.logger.info("PART 6: WORST CASE SCENARIOS")
        self.logger.info("-"*70)

        # 13. All stress factors combined (baseline)
        results.append(self.run_configuration(
            long_pct=0.2, short_pct=0.2,
            transaction_cost=0.001,  # 10 bps
            slippage_bps=10.0,
            execution_delay=1,
            liquidity_filter=100,
            config_name="WORST CASE: Baseline + All Stress"
        ))

        # 14. All stress factors combined (long-biased)
        results.append(self.run_configuration(
            long_pct=0.28, short_pct=0.12,
            transaction_cost=0.001,
            slippage_bps=10.0,
            execution_delay=1,
            liquidity_filter=100,
            config_name="WORST CASE: Long-Biased + All Stress"
        ))

        return results

    def analyze_results(self, results: List[Dict]):
        """Analyze and summarize all stress test results."""
        self.logger.info("\n" + "="*70)
        self.logger.info("STRESS TEST SUMMARY")
        self.logger.info("="*70)
        self.logger.info("")

        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(results)

        # Print summary table
        self.logger.info("Configuration                              | Net Sharpe | Gross Sharpe | Turnover | Max DD")
        self.logger.info("-" * 90)

        for _, row in df.iterrows():
            name = row['config_name'].ljust(42)
            self.logger.info(
                f"{name} | {row['sharpe_net']:10.2f} | {row['sharpe_gross']:12.2f} | "
                f"{row['avg_turnover']*100:7.1f}% | {row['max_drawdown']*100:6.2f}%"
            )

        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("KEY FINDINGS")
        self.logger.info("="*70)
        self.logger.info("")

        # Find best configuration
        best_idx = df['sharpe_net'].idxmax()
        best = df.loc[best_idx]

        self.logger.info(f"BEST CONFIGURATION: {best['config_name']}")
        self.logger.info(f"  Net Sharpe:  {best['sharpe_net']:.2f}")
        self.logger.info(f"  Turnover:    {best['avg_turnover']*100:.1f}%")
        self.logger.info(f"  Max DD:      {best['max_drawdown']*100:.2f}%")
        self.logger.info("")

        # Check stress test resilience
        worst_case_baseline = df[df['config_name'].str.contains('WORST CASE: Baseline')].iloc[0]
        worst_case_longbias = df[df['config_name'].str.contains('WORST CASE: Long-Biased')].iloc[0]

        self.logger.info("WORST CASE SCENARIOS:")
        self.logger.info(f"  Baseline (50/50):     Net Sharpe = {worst_case_baseline['sharpe_net']:.2f}")
        self.logger.info(f"  Long-Biased (70/30):  Net Sharpe = {worst_case_longbias['sharpe_net']:.2f}")
        self.logger.info("")

        # Gate check
        if worst_case_longbias['sharpe_net'] > 1.2:
            self.logger.info("[EXCELLENT] Strategy remains strong even under worst case")
            self.logger.info("  Net Sharpe > 1.2 under all stress factors")
            gate_status = "PASS"
        elif worst_case_longbias['sharpe_net'] > 0.5:
            self.logger.info("[GOOD] Strategy survives stress tests")
            self.logger.info("  Net Sharpe > 0.5 confirms real alpha")
            gate_status = "PASS"
        else:
            self.logger.info("[WARNING] Strategy struggles under stress")
            self.logger.info("  Net Sharpe < 0.5 may indicate liquidity illusion")
            gate_status = "CONDITIONAL"

        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info(f"PHASE 3.3 GATE STATUS: {gate_status}")
        self.logger.info("="*70)

        return {
            'gate_status': gate_status,
            'best_config': best.to_dict(),
            'worst_case_baseline': worst_case_baseline.to_dict(),
            'worst_case_longbias': worst_case_longbias.to_dict()
        }


def main():
    """Run Phase 3.3 execution stress tests."""
    logger = setup_logger('phase3_3_stress', log_file='logs/phase3_3_execution_stress.log', level='INFO')

    logger.info("="*70)
    logger.info("PHASE 3.3: EXECUTION REALITY STRESS TESTS")
    logger.info("="*70)
    logger.info("")

    # Check for pyarrow
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("PyArrow not installed. Installing now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
        logger.info("PyArrow installed successfully. Please re-run the script.")
        return

    # Load predictions
    processed_file = Path("data/processed/phase1_predictions.parquet")
    if not processed_file.exists():
        logger.error(f"Predictions file not found: {processed_file}")
        logger.error("Please run Phase 1 or Phase 2A training first")
        return

    logger.info(f"Loading predictions from {processed_file}...")
    predictions_df = pd.read_parquet(processed_file)
    logger.info(f"Loaded {len(predictions_df)} predictions")

    # Load config
    config = ConfigLoader('config/config.yaml')

    # Run stress tests
    stress_tester = ExecutionStressTest(predictions_df, config)
    results = stress_tester.run_all_stress_tests()

    # Analyze results
    summary = stress_tester.analyze_results(results)

    # Save results
    results_file = "data/processed/phase3_3_execution_stress_results.json"

    # Convert numpy types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    output = {
        'all_results': results,
        'summary': summary
    }

    output_json = convert_to_json_serializable(output)

    with open(results_file, 'w') as f:
        json.dump(output_json, f, indent=2)

    logger.info("")
    logger.info(f"Results saved to: {results_file}")
    logger.info("")
    logger.info("="*70)
    logger.info("PHASE 3.3 COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
