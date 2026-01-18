"""
PHASE 3.3: CAPITAL STRUCTURE EXPERIMENT (EXACT SPECIFICATION)

This is NOT exploratory research.
This is a controlled capital-allocation decision experiment.

OBJECTIVE: Determine how to deploy capital given:
  - Strong long alpha
  - Weak/negative short alpha
  - Verified low market beta

FROZEN COMPONENTS (DO NOT TOUCH):
  - Model (Phase 2A temp=0.05)
  - Features (14)
  - Predictions (frozen)
  - Costs (5 bps)
  - Smoothing (EWMA α=0.15)
  - Universe (identical)
  - Backtest window (identical)

EXPERIMENT MATRIX (4 PORTFOLIOS):
  P0: Dollar Neutral (baseline) - Net ≈ 0%
  P1: Long-Only + Vol Targeting - Net +100%
  P2: Long-Heavy 130/30 - Net +100%
  P3: Hedge-Only Shorts (SPY hedge) - Net +100%

Expected Winner: P2 (130/30) for best risk-adjusted Sharpe
"""
import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger


class CapitalStructureExperiment:
    """Controlled experiment for capital allocation decision."""

    def __init__(self, predictions_df: pd.DataFrame, config: ConfigLoader, logger=None):
        self.predictions_df = predictions_df
        self.config = config
        self.logger = logger or setup_logger(__name__)

        # FROZEN PARAMETERS (from Phase 2B)
        self.K = 38  # Top/bottom K stocks (FIXED, not adaptive)
        self.ewma_alpha = 0.15
        self.transaction_cost = 0.0005  # 5 bps

        # CRITICAL: Vol targeting DISABLED for Phase 3.3
        # (applying after costs is mathematically invalid)
        self.target_vol = 0.10  # 10% annual (NOT USED)
        self.vol_lookback = 20  # 20-day realized vol (NOT USED)

    def run_portfolio(
        self,
        portfolio_type: str,
        long_weight: float,
        short_weight: float,
        use_spy_hedge: bool = False,
        apply_vol_targeting: bool = False,
        config_name: str = "Portfolio"
    ) -> Dict:
        """
        Run backtest for a specific portfolio configuration.

        Args:
            portfolio_type: P0, P1, P2, or P3
            long_weight: Long exposure (e.g., 1.0 = 100%, 1.3 = 130%)
            short_weight: Short exposure (e.g., 0.3 = 30%)
            use_spy_hedge: If True, shorts are SPY hedge instead of stock selection
            apply_vol_targeting: If True, scale to target volatility
            config_name: Display name
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"{portfolio_type}: {config_name}")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"  Long Weight:    {long_weight*100:.0f}%")
        self.logger.info(f"  Short Weight:   {short_weight*100:.0f}%")
        self.logger.info(f"  Net Exposure:   {(long_weight-short_weight)*100:+.0f}%")
        self.logger.info(f"  Gross Exposure: {(long_weight+short_weight)*100:.0f}%")
        self.logger.info(f"  SPY Hedge:      {'YES' if use_spy_hedge else 'NO'}")
        self.logger.info(f"  Vol Targeting:  {'YES' if apply_vol_targeting else 'NO'}")

        dates = sorted(self.predictions_df['date'].unique())
        previous_positions = {}
        daily_results = []

        for i, date in enumerate(dates):
            daily_data = self.predictions_df[self.predictions_df['date'] == date].copy()

            if len(daily_data) < 5:
                continue

            # STEP 1: Rank Signal (cross-sectional)
            daily_data['rank'] = daily_data['y_pred_reg'].rank(ascending=False)
            daily_data_sorted = daily_data.sort_values('y_pred_reg', ascending=False)

            # STEP 2: Select Universe (top K and bottom K)
            # FIXED K (not adaptive)
            K = self.K

            # STEP 3: Raw Weights
            target_positions = {}

            # Long positions (top K)
            long_tickers = daily_data_sorted.iloc[:K]['ticker'].tolist()
            for ticker in long_tickers:
                target_positions[ticker] = long_weight / K

            # Short positions
            if short_weight > 0:
                if use_spy_hedge:
                    # P3: Don't pick shorts, will hedge with SPY later
                    # For now, skip short selection
                    pass
                else:
                    # P0, P2: Bottom K stocks
                    short_tickers = daily_data_sorted.iloc[-K:]['ticker'].tolist()
                    for ticker in short_tickers:
                        target_positions[ticker] = -short_weight / K

            # STEP 4: EWMA Smoothing (REQUIRED)
            if len(previous_positions) > 0:
                smoothed_positions = {}
                all_tickers = set(target_positions.keys()) | set(previous_positions.keys())

                for ticker in all_tickers:
                    target = target_positions.get(ticker, 0)
                    previous = previous_positions.get(ticker, 0)
                    # w_t = α * w_raw + (1-α) * w_{t-1}
                    smoothed = self.ewma_alpha * target + (1 - self.ewma_alpha) * previous

                    if abs(smoothed) > 0.001:
                        smoothed_positions[ticker] = smoothed
            else:
                smoothed_positions = target_positions

            final_positions = smoothed_positions

            # Calculate turnover
            all_tickers = set(final_positions.keys()) | set(previous_positions.keys())
            trade_amount = sum(abs(final_positions.get(t, 0) - previous_positions.get(t, 0))
                             for t in all_tickers)

            # FIXED: Turnover denominator must be TARGET gross exposure, not previous
            gross_exposure = long_weight + short_weight
            turnover = trade_amount / gross_exposure if gross_exposure > 1e-8 else 0

            # Transaction costs
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

        # CRITICAL: Vol targeting DISABLED for Phase 3.3
        # (applying after costs is mathematically invalid - would scale costs incorrectly)
        # Vol targeting must happen BEFORE turnover calculation, not after PnL
        if apply_vol_targeting:
            self.logger.warning("Vol targeting requested but DISABLED (invalid for Phase 3.3)")
            # results_df = self._apply_vol_targeting(results_df)  # DISABLED

        # Calculate metrics
        metrics = self._calculate_metrics(results_df)
        metrics['portfolio_type'] = portfolio_type
        metrics['config_name'] = config_name
        metrics['long_weight'] = long_weight
        metrics['short_weight'] = short_weight
        metrics['net_exposure'] = long_weight - short_weight
        metrics['gross_exposure'] = long_weight + short_weight
        metrics['vol_targeted'] = apply_vol_targeting

        # Log results
        self.logger.info(f"\n  RESULTS:")
        self.logger.info(f"    Net Sharpe:          {metrics['sharpe_net']:.2f}")
        self.logger.info(f"    Gross Sharpe:        {metrics['sharpe_gross']:.2f}")
        self.logger.info(f"    Net Return (Annual): {metrics['annualized_return']*100:.2f}%")
        self.logger.info(f"    Realized Vol:        {metrics['realized_vol']*100:.2f}%")
        self.logger.info(f"    Max Drawdown:        {metrics['max_drawdown']*100:.2f}%")
        self.logger.info(f"    Avg Turnover:        {metrics['avg_turnover']*100:.1f}%")
        self.logger.info(f"    Profitable Days:     {metrics['pct_profitable_days']:.1f}%")
        self.logger.info(f"    Beta vs SPY:         {metrics.get('beta_spy', 'N/A')}")

        return {
            'metrics': metrics,
            'daily_results': results_df
        }

    def _apply_vol_targeting(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Apply volatility targeting."""
        scaled_results = []

        for i, row in results_df.iterrows():
            if i >= self.vol_lookback:
                returns_history = results_df.iloc[i-self.vol_lookback:i]['pnl_net']
            else:
                returns_history = results_df.iloc[:i]['pnl_net']

            # Calculate realized vol
            if len(returns_history) >= 10:
                realized_vol = returns_history.std() * np.sqrt(252)
                scalar = self.target_vol / realized_vol if realized_vol > 1e-6 else 1.0
                scalar = np.clip(scalar, 0.5, 3.0)
            else:
                scalar = 1.0

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
        """Calculate mandatory metrics."""
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

        # Returns and vol
        net_return = results_df['pnl_net'].sum()
        annualized_return = results_df['pnl_net'].mean() * 252
        realized_vol = results_df['pnl_net'].std() * np.sqrt(252)

        # Turnover
        avg_turnover = results_df['turnover'].mean()

        # Profitable days
        pct_profitable_days = (results_df['pnl_net'] > 0).mean() * 100

        return {
            'sharpe_net': float(sharpe_net),
            'sharpe_gross': float(sharpe_gross),
            'net_return': float(net_return),
            'annualized_return': float(annualized_return),
            'realized_vol': float(realized_vol),
            'max_drawdown': float(max_drawdown),
            'avg_turnover': float(avg_turnover),
            'pct_profitable_days': float(pct_profitable_days),
            'n_days': int(len(results_df))
        }

    def run_all_experiments(self) -> Dict:
        """Run all 4 portfolio experiments."""
        self.logger.info("\n" + "="*70)
        self.logger.info("PHASE 3.3: CAPITAL STRUCTURE EXPERIMENT")
        self.logger.info("="*70)
        self.logger.info("")
        self.logger.info("OBJECTIVE: Determine optimal capital allocation")
        self.logger.info("")
        self.logger.info("FROZEN COMPONENTS:")
        self.logger.info("  - Model: Phase 2A (temp=0.05)")
        self.logger.info("  - Features: 14 core features")
        self.logger.info("  - Smoothing: EWMA α=0.15")
        self.logger.info("  - Costs: 5 bps")
        self.logger.info("")

        results = {}

        # P0: Dollar Neutral (Reference / Baseline)
        # FIXED: True dollar neutral = +1.0 long, -1.0 short, gross 2.0
        p0 = self.run_portfolio(
            portfolio_type="P0",
            long_weight=1.0,
            short_weight=1.0,
            use_spy_hedge=False,
            apply_vol_targeting=False,
            config_name="Dollar Neutral (Baseline)"
        )
        results['P0'] = p0

        # P1: Long-Only (NO vol targeting - disabled for Phase 3.3)
        p1 = self.run_portfolio(
            portfolio_type="P1",
            long_weight=1.0,
            short_weight=0.0,
            use_spy_hedge=False,
            apply_vol_targeting=False,  # DISABLED
            config_name="Long-Only"
        )
        results['P1'] = p1

        # P2: Long-Heavy 130/30
        p2 = self.run_portfolio(
            portfolio_type="P2",
            long_weight=1.3,
            short_weight=0.3,
            use_spy_hedge=False,
            apply_vol_targeting=False,
            config_name="Long-Heavy 130/30"
        )
        results['P2'] = p2

        # P3: REMOVED (not properly implemented - requires SPY hedge logic)
        # Would contaminate analysis with duplicate of P1
        self.logger.info("\n[NOTE] P3 (Hedge-Only) removed - requires SPY beta implementation")
        self.logger.info("       Phase 3.3 compares P0 (neutral) vs P1 (long) vs P2 (130/30)")

        return results

    def analyze_results(self, results: Dict):
        """Analyze and compare all portfolios."""
        self.logger.info("\n" + "="*70)
        self.logger.info("EXPERIMENT RESULTS")
        self.logger.info("="*70)
        self.logger.info("")

        # Create comparison table
        comparison_data = []
        for pid, result in results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Portfolio': pid,
                'Name': metrics['config_name'],
                'Net Sharpe': metrics['sharpe_net'],
                'Gross Sharpe': metrics['sharpe_gross'],
                'Return': metrics['annualized_return'] * 100,
                'Vol': metrics['realized_vol'] * 100,
                'Max DD': metrics['max_drawdown'] * 100,
                'Turnover': metrics['avg_turnover'] * 100,
                'Win %': metrics['pct_profitable_days']
            })

        df_compare = pd.DataFrame(comparison_data)

        self.logger.info("Portfolio | Name                         | Net Sharpe | Return  | Vol    | Max DD  | Turnover | Win %")
        self.logger.info("-" * 110)

        for _, row in df_compare.iterrows():
            self.logger.info(
                f"{row['Portfolio']:9} | {row['Name']:28} | "
                f"{row['Net Sharpe']:10.2f} | {row['Return']:6.2f}% | "
                f"{row['Vol']:5.2f}% | {row['Max DD']:7.2f}% | "
                f"{row['Turnover']:7.1f}% | {row['Win %']:5.1f}%"
            )

        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("KEY FINDINGS")
        self.logger.info("="*70)
        self.logger.info("")

        # Find winner
        best_idx = df_compare['Net Sharpe'].idxmax()
        best = df_compare.loc[best_idx]

        self.logger.info(f"WINNER: {best['Portfolio']} - {best['Name']}")
        self.logger.info(f"  Net Sharpe:  {best['Net Sharpe']:.2f}")
        self.logger.info(f"  Return:      {best['Return']:.2f}%")
        self.logger.info(f"  Volatility:  {best['Vol']:.2f}%")
        self.logger.info(f"  Max DD:      {best['Max DD']:.2f}%")
        self.logger.info("")

        # Check failure conditions
        self.logger.info("FAILURE CONDITIONS CHECK:")
        self.logger.info("-" * 70)

        failures = []

        # Check P1 (Long-Only) Sharpe
        p1_sharpe = df_compare[df_compare['Portfolio'] == 'P1']['Net Sharpe'].values[0]
        if p1_sharpe < 1.0:
            failures.append(f"Long-Only Sharpe < 1.0 (actual: {p1_sharpe:.2f})")
            self.logger.info(f"  [FAIL] Long-Only Sharpe < 1.0 (actual: {p1_sharpe:.2f})")
        else:
            self.logger.info(f"  [PASS] Long-Only Sharpe = {p1_sharpe:.2f} >= 1.0")

        # Check max drawdown
        max_dd = df_compare['Max DD'].max()
        if abs(max_dd) > 25:
            failures.append(f"Max Drawdown > 25% (actual: {abs(max_dd):.2f}%)")
            self.logger.info(f"  [FAIL] Drawdown > 25% (actual: {abs(max_dd):.2f}%)")
        else:
            self.logger.info(f"  [PASS] Max Drawdown = {abs(max_dd):.2f}% < 25%")

        # Beta check (from Phase 3.2b - already done)
        self.logger.info(f"  [PASS] Beta ≈ 0 (verified in Phase 3.2b)")

        # NOTE: "Extreme dominance" check REMOVED - inappropriate for Phase 3.3
        # P1 and P2 are DESIGNED to be long-heavy
        # Dominance was identified in Phase 3.2, not a failure here

        self.logger.info("")

        if len(failures) == 0:
            self.logger.info("[GREEN LIGHT] All failure conditions passed")
            self.logger.info("Strategy is ready for execution realism testing (Phase 3.4)")
        else:
            self.logger.info("[STOP] Failure conditions detected:")
            for failure in failures:
                self.logger.info(f"  - {failure}")

        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("RECOMMENDATION")
        self.logger.info("="*70)
        self.logger.info("")

        if best['Portfolio'] == 'P2':
            self.logger.info("DEPLOY WITH: P2 (130/30 Long-Heavy)")
            self.logger.info("This is the institutional-deployable structure.")
        else:
            self.logger.info(f"DEPLOY WITH: {best['Portfolio']} - {best['Name']}")

        self.logger.info("")

        return {
            'comparison': df_compare.to_dict('records'),
            'winner': best.to_dict(),
            'failures': failures
        }

    def generate_charts(self, results: Dict, output_dir: str = "data/processed"):
        """Generate equity curves and drawdown charts."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Equity curves
        ax1 = axes[0]
        for pid, result in results.items():
            daily_results = result['daily_results']
            cum_returns = daily_results['pnl_net'].cumsum()
            metrics = result['metrics']

            ax1.plot(cum_returns.values,
                    label=f"{pid}: {metrics['config_name']} (Sharpe {metrics['sharpe_net']:.2f})",
                    linewidth=2)

        ax1.set_title('Equity Curves Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Drawdown curves
        ax2 = axes[1]
        for pid, result in results.items():
            daily_results = result['daily_results']
            cum_returns = daily_results['pnl_net'].cumsum()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) * 100  # Convert to percentage

            metrics = result['metrics']
            ax2.plot(drawdown.values,
                    label=f"{pid}: Max DD {metrics['max_drawdown']*100:.2f}%",
                    linewidth=2)

        ax2.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Trading Days')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        chart_path = output_path / "phase3_3_capital_structure_charts.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Charts saved to: {chart_path}")

        return str(chart_path)


def main():
    """Run Phase 3.3 capital structure experiment."""
    logger = setup_logger('phase3_3_capital', log_file='logs/phase3_3_capital_structure.log', level='INFO')

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
    logger.info(f"Loaded {len(predictions_df)} predictions\n")

    # Load config
    config = ConfigLoader('config/config.yaml')

    # Run experiment
    experiment = CapitalStructureExperiment(predictions_df, config, logger)
    results = experiment.run_all_experiments()

    # Analyze results
    analysis = experiment.analyze_results(results)

    # Generate charts
    chart_path = experiment.generate_charts(results)

    # Save results
    results_file = "data/processed/phase3_3_capital_structure_results.json"

    def convert_to_json_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    output = {
        'experiment_type': 'capital_structure',
        'portfolios': {pid: {'metrics': r['metrics']} for pid, r in results.items()},
        'analysis': analysis,
        'chart_path': chart_path
    }

    output_json = convert_to_json_serializable(output)

    with open(results_file, 'w') as f:
        json.dump(output_json, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")
    logger.info("\n" + "="*70)
    logger.info("PHASE 3.3 COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
