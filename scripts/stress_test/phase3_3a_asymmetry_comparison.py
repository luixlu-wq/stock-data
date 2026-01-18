"""
PHASE 3.3A: Long/Short Asymmetry Comparison

CRITICAL QUESTION: How much Sharpe are we leaving on the table with balanced L/S?

This script compares portfolio structures:
1. Long-only (100% long, 0% short) + vol targeting
2. Long-heavy (70% long, 30% short)
3. Moderate (60% long, 40% short)
4. Balanced (50% long, 50% short) - baseline
5. 130/30 (130% long, 30% short) - classic structure

Same signals, same costs, same smoothing - only portfolio structure changes.

Based on Phase 3.2 finding:
  Long Sharpe: +1.28
  Short Sharpe: -0.45
  → Shorts are dragging performance

Expected result: Long-heavy configs should win decisively.
"""
import sys
from pathlib import Path
import json
from typing import Dict, List

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger


class AsymmetryComparison:
    """Compare different long/short asymmetry configurations."""

    def __init__(self, predictions_df: pd.DataFrame, config: ConfigLoader, logger=None):
        self.predictions_df = predictions_df
        self.config = config
        self.logger = logger or setup_logger(__name__)

        # Fixed parameters (from Phase 2B best config)
        self.ewma_alpha = 0.15
        self.transaction_cost = 0.0005  # 5 bps
        self.target_vol = 0.08  # 8% annual for long-only

    def run_portfolio_config(
        self,
        long_weight: float,
        short_weight: float,
        apply_vol_targeting: bool = False,
        config_name: str = "Config"
    ) -> Dict:
        """
        Run backtest for a specific long/short allocation.

        Args:
            long_weight: Gross long exposure (e.g., 1.0 = 100%, 1.3 = 130%)
            short_weight: Gross short exposure (e.g., 0.3 = 30%, 0.5 = 50%)
            apply_vol_targeting: Whether to apply volatility targeting
            config_name: Name for logging

        Returns:
            metrics dict
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"TESTING: {config_name}")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"  Long Exposure:  {long_weight*100:.0f}%")
        self.logger.info(f"  Short Exposure: {short_weight*100:.0f}%")
        self.logger.info(f"  Gross Exposure: {(long_weight+short_weight)*100:.0f}%")
        self.logger.info(f"  Net Exposure:   {(long_weight-short_weight)*100:.0f}%")
        self.logger.info(f"  Vol Targeting:  {'YES' if apply_vol_targeting else 'NO'}")

        dates = sorted(self.predictions_df['date'].unique())
        previous_positions = {}
        daily_results = []

        for i, date in enumerate(dates):
            daily_data = self.predictions_df[self.predictions_df['date'] == date].copy()

            if len(daily_data) < 5:
                continue

            # Rank stocks by prediction
            daily_data_sorted = daily_data.sort_values('y_pred_reg', ascending=False)
            n_stocks = len(daily_data)

            # For long-only: just take top 20%
            # For long/short: take top 20% long, bottom 20% short
            n_long = max(1, int(n_stocks * 0.2))
            n_short = max(1, int(n_stocks * 0.2)) if short_weight > 0 else 0

            # Create target positions
            target_positions = {}

            # Long positions (weighted by long_weight)
            long_tickers = daily_data_sorted.iloc[:n_long]['ticker'].tolist()
            for ticker in long_tickers:
                target_positions[ticker] = (1.0 / n_long) * long_weight

            # Short positions (weighted by short_weight)
            if short_weight > 0:
                short_tickers = daily_data_sorted.iloc[-n_short:]['ticker'].tolist()
                for ticker in short_tickers:
                    target_positions[ticker] = -(1.0 / n_short) * short_weight

            # Apply EWMA smoothing
            if len(previous_positions) > 0:
                smoothed_positions = {}
                all_tickers = set(target_positions.keys()) | set(previous_positions.keys())

                for ticker in all_tickers:
                    target = target_positions.get(ticker, 0)
                    previous = previous_positions.get(ticker, 0)
                    smoothed = (1 - self.ewma_alpha) * previous + self.ewma_alpha * target

                    if abs(smoothed) > 0.001:
                        smoothed_positions[ticker] = smoothed
            else:
                smoothed_positions = target_positions

            final_positions = smoothed_positions

            # Calculate turnover and costs
            all_tickers = set(final_positions.keys()) | set(previous_positions.keys())
            trade_amount = sum(abs(final_positions.get(t, 0) - previous_positions.get(t, 0))
                             for t in all_tickers)

            gross_exposure = sum(abs(p) for p in previous_positions.values()) if previous_positions else (long_weight + short_weight)
            turnover = trade_amount / gross_exposure if gross_exposure > 1e-8 else 0

            trading_cost = (trade_amount / 2.0) * self.transaction_cost

            # Calculate PnL
            returns_map = daily_data.set_index('ticker')['y_true_reg'].to_dict()

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

        # Convert to DataFrame
        results_df = pd.DataFrame(daily_results)

        # Apply volatility targeting if requested (for long-only)
        if apply_vol_targeting:
            results_df = self._apply_vol_targeting(results_df)

        # Calculate metrics
        metrics = self._calculate_metrics(results_df)
        metrics['config_name'] = config_name
        metrics['long_weight'] = long_weight
        metrics['short_weight'] = short_weight
        metrics['gross_exposure'] = long_weight + short_weight
        metrics['net_exposure'] = long_weight - short_weight
        metrics['vol_targeted'] = apply_vol_targeting

        self.logger.info(f"\n  RESULTS:")
        self.logger.info(f"    Net Sharpe:          {metrics['sharpe_net']:.2f}")
        self.logger.info(f"    Gross Sharpe:        {metrics['sharpe_gross']:.2f}")
        self.logger.info(f"    Annualized Return:   {metrics['annualized_return']*100:.2f}%")
        self.logger.info(f"    Realized Vol:        {metrics['realized_vol']*100:.2f}%")
        self.logger.info(f"    Turnover:            {metrics['avg_turnover']*100:.1f}%")
        self.logger.info(f"    Max Drawdown:        {metrics['max_drawdown']*100:.2f}%")
        self.logger.info(f"    Long Sharpe:         {metrics['sharpe_long']:.2f}")
        self.logger.info(f"    Short Sharpe:        {metrics['sharpe_short']:.2f}")

        return metrics

    def _apply_vol_targeting(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Apply volatility targeting to scale positions."""
        vol_lookback = 60
        scaled_results = []

        for i, row in results_df.iterrows():
            if i >= vol_lookback:
                returns_history = results_df.iloc[i-vol_lookback:i]['pnl_net']
            else:
                returns_history = results_df.iloc[:i]['pnl_net']

            # Calculate realized vol
            if len(returns_history) >= 20:
                realized_vol = returns_history.std() * np.sqrt(252)
                scalar = self.target_vol / realized_vol if realized_vol > 1e-6 else 1.0
                scalar = np.clip(scalar, 0.5, 3.0)  # Cap at reasonable bounds
            else:
                scalar = 1.0

            # Scale PnL
            scaled_results.append({
                'date': row['date'],
                'pnl_gross': row['pnl_gross'] * scalar,
                'pnl_net': row['pnl_net'] * scalar,
                'pnl_long': row['pnl_long'] * scalar,
                'pnl_short': row['pnl_short'] * scalar,
                'turnover': row['turnover'],
                'cost': row['cost'] * scalar,
                'vol_scalar': scalar
            })

        return pd.DataFrame(scaled_results)

    def _calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        # Sharpe ratios
        sharpe_net = (results_df['pnl_net'].mean() / results_df['pnl_net'].std() * np.sqrt(252)
                     if results_df['pnl_net'].std() > 0 else 0)
        sharpe_gross = (results_df['pnl_gross'].mean() / results_df['pnl_gross'].std() * np.sqrt(252)
                       if results_df['pnl_gross'].std() > 0 else 0)

        # Long and short Sharpe
        sharpe_long = (results_df['pnl_long'].mean() / results_df['pnl_long'].std() * np.sqrt(252)
                      if results_df['pnl_long'].std() > 0 else 0)
        sharpe_short = (results_df['pnl_short'].mean() / results_df['pnl_short'].std() * np.sqrt(252)
                       if results_df['pnl_short'].std() > 0 and len(results_df[results_df['pnl_short'] != 0]) > 0 else 0)

        # Drawdown
        cum_returns = results_df['pnl_net'].cumsum()
        running_max = cum_returns.expanding().max()
        drawdown = cum_returns - running_max
        max_drawdown = drawdown.min()

        # Returns and volatility
        annualized_return = results_df['pnl_net'].mean() * 252
        realized_vol = results_df['pnl_net'].std() * np.sqrt(252)

        # Costs
        avg_turnover = results_df['turnover'].mean()
        total_costs = results_df['cost'].sum()

        return {
            'sharpe_net': float(sharpe_net),
            'sharpe_gross': float(sharpe_gross),
            'sharpe_long': float(sharpe_long),
            'sharpe_short': float(sharpe_short),
            'annualized_return': float(annualized_return),
            'realized_vol': float(realized_vol),
            'max_drawdown': float(max_drawdown),
            'avg_turnover': float(avg_turnover),
            'total_costs': float(total_costs),
            'n_days': int(len(results_df))
        }

    def run_all_comparisons(self) -> List[Dict]:
        """Run all asymmetry comparisons."""
        results = []

        self.logger.info("\n" + "="*70)
        self.logger.info("PHASE 3.3A: LONG/SHORT ASYMMETRY COMPARISON")
        self.logger.info("="*70)
        self.logger.info("")
        self.logger.info("Testing different portfolio structures with same signals...")
        self.logger.info("")

        # 1. Baseline: Balanced 50/50
        results.append(self.run_portfolio_config(
            long_weight=0.5,
            short_weight=0.5,
            apply_vol_targeting=False,
            config_name="Baseline (50/50 Balanced)"
        ))

        # 2. Long-heavy: 60/40
        results.append(self.run_portfolio_config(
            long_weight=0.6,
            short_weight=0.4,
            apply_vol_targeting=False,
            config_name="Long-Heavy (60/40)"
        ))

        # 3. Long-biased: 70/30
        results.append(self.run_portfolio_config(
            long_weight=0.7,
            short_weight=0.3,
            apply_vol_targeting=False,
            config_name="Long-Biased (70/30)"
        ))

        # 4. Very long-biased: 80/20
        results.append(self.run_portfolio_config(
            long_weight=0.8,
            short_weight=0.2,
            apply_vol_targeting=False,
            config_name="Very Long-Biased (80/20)"
        ))

        # 5. Long-only with vol targeting
        results.append(self.run_portfolio_config(
            long_weight=1.0,
            short_weight=0.0,
            apply_vol_targeting=True,
            config_name="Long-Only + Vol Targeting"
        ))

        # 6. Classic 130/30
        results.append(self.run_portfolio_config(
            long_weight=1.3,
            short_weight=0.3,
            apply_vol_targeting=False,
            config_name="Classic 130/30"
        ))

        return results

    def analyze_results(self, results: List[Dict]):
        """Analyze and summarize asymmetry comparison."""
        self.logger.info("\n" + "="*70)
        self.logger.info("ASYMMETRY COMPARISON SUMMARY")
        self.logger.info("="*70)
        self.logger.info("")

        # Create comparison table
        df = pd.DataFrame(results)

        self.logger.info("Configuration                  | Net Sharpe | Ann Return | Realized Vol | Turnover | Max DD")
        self.logger.info("-" * 95)

        for _, row in df.iterrows():
            name = row['config_name'].ljust(30)
            self.logger.info(
                f"{name} | {row['sharpe_net']:10.2f} | {row['annualized_return']*100:9.2f}% | "
                f"{row['realized_vol']*100:11.2f}% | {row['avg_turnover']*100:7.1f}% | {row['max_drawdown']*100:6.2f}%"
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
        self.logger.info(f"  Net Sharpe:          {best['sharpe_net']:.2f}")
        self.logger.info(f"  Annualized Return:   {best['annualized_return']*100:.2f}%")
        self.logger.info(f"  Realized Volatility: {best['realized_vol']*100:.2f}%")
        self.logger.info(f"  Turnover:            {best['avg_turnover']*100:.1f}%")
        self.logger.info(f"  Max Drawdown:        {best['max_drawdown']*100:.2f}%")
        self.logger.info("")

        # Compare to baseline
        baseline = df[df['config_name'].str.contains('Baseline')].iloc[0]
        sharpe_improvement = ((best['sharpe_net'] - baseline['sharpe_net']) / baseline['sharpe_net'] * 100)

        self.logger.info("IMPROVEMENT VS BASELINE:")
        self.logger.info(f"  Baseline (50/50):  Net Sharpe = {baseline['sharpe_net']:.2f}")
        self.logger.info(f"  Best Config:       Net Sharpe = {best['sharpe_net']:.2f}")
        self.logger.info(f"  Improvement:       {sharpe_improvement:+.1f}%")
        self.logger.info("")

        # Interpretation
        self.logger.info("INTERPRETATION:")
        self.logger.info("-" * 70)

        if sharpe_improvement > 20:
            self.logger.info("  [CRITICAL] Asymmetry matters significantly!")
            self.logger.info(f"  Switching to {best['config_name']} would boost Sharpe by {sharpe_improvement:.0f}%")
            self.logger.info("  This confirms shorts are dragging performance.")
        elif sharpe_improvement > 10:
            self.logger.info("  [IMPORTANT] Moderate improvement with asymmetry")
            self.logger.info(f"  Consider switching to {best['config_name']}")
        else:
            self.logger.info("  [SURPRISING] Balanced portfolio performs well")
            self.logger.info("  Short book may be more valuable than Phase 3.2 suggested")

        self.logger.info("")

        # Long-only analysis
        long_only = df[df['config_name'].str.contains('Long-Only')].iloc[0]
        self.logger.info("LONG-ONLY ANALYSIS:")
        self.logger.info(f"  Net Sharpe:  {long_only['sharpe_net']:.2f}")
        self.logger.info(f"  Return:      {long_only['annualized_return']*100:.2f}%")
        self.logger.info(f"  Volatility:  {long_only['realized_vol']*100:.2f}% (targeted at 8%)")
        self.logger.info("")

        if long_only['sharpe_net'] > baseline['sharpe_net'] * 0.9:
            self.logger.info("  [CONSIDER] Long-only is competitive")
            self.logger.info("  Simpler execution, no shorting costs, easier to scale")
        else:
            self.logger.info("  [NOTE] Long/short still superior despite short drag")

        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("RECOMMENDATION")
        self.logger.info("="*70)
        self.logger.info("")
        self.logger.info(f"Deploy with: {best['config_name']}")
        self.logger.info(f"Expected Net Sharpe: {best['sharpe_net']:.2f}")
        self.logger.info(f"Expected Return: {best['annualized_return']*100:.2f}%")
        self.logger.info("")

        return {
            'best_config': best.to_dict(),
            'baseline': baseline.to_dict(),
            'long_only': long_only.to_dict(),
            'improvement_pct': float(sharpe_improvement)
        }


def main():
    """Run Phase 3.3A asymmetry comparison."""
    logger = setup_logger('phase3_3a_asymmetry', log_file='logs/phase3_3a_asymmetry_comparison.log', level='INFO')

    logger.info("="*70)
    logger.info("PHASE 3.3A: LONG/SHORT ASYMMETRY COMPARISON")
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
    logger.info("")

    # Load config
    config = ConfigLoader('config/config.yaml')

    # Run comparison
    comparator = AsymmetryComparison(predictions_df, config, logger)
    results = comparator.run_all_comparisons()

    # Analyze results
    summary = comparator.analyze_results(results)

    # Save results
    results_file = "data/processed/phase3_3a_asymmetry_results.json"

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
    logger.info("PHASE 3.3A COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
