"""
PHASE 3.3: PORTFOLIO COMPARISON (SPEC-CORRECT)

This is NOT exploratory. This is a controlled capital-allocation decision.

HARD CONSTRAINTS:
- Same predictions (frozen)
- Same dates (identical)
- Same K (fixed, not adaptive)
- No vol targeting (disabled)
- Gross exposure fixed per portfolio
- Turnover correctly normalized

PASS/FAIL GATES:
- Sharpe(P2) - Sharpe(P0) >= 0.30
- Turnover(P2) <= 1.5 × Turnover(P0)
- MaxDD(P2) <= 1.5 × MaxDD(P0)

If any fail → P0 wins (dollar neutral is optimal)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger

TRADING_DAYS = 252


class Phase33PortfolioExperiment:
    """Spec-correct portfolio comparison experiment."""

    def __init__(
        self,
        predictions: pd.DataFrame,
        K: int = 38,  # Fixed from Phase 2B
        transaction_cost_bps: float = 5.0,
        ewma_alpha: float = 0.15,
        logger=None
    ):
        """
        Initialize experiment.

        Args:
            predictions: DataFrame with [date, ticker, y_pred_reg, y_true_reg]
            K: Number of stocks per bucket (FIXED, not adaptive)
            transaction_cost_bps: Transaction cost in basis points
            ewma_alpha: EWMA smoothing parameter (frozen from Phase 2B)
            logger: Logger instance
        """
        self.df = predictions.copy()
        self.K = K
        self.tc = transaction_cost_bps / 10_000  # bps → decimal
        self.ewma_alpha = ewma_alpha
        self.logger = logger or setup_logger(__name__)

    def run_portfolio(
        self,
        name: str,
        long_weight: float,
        short_weight: float,
        long_only: bool = False
    ) -> dict:
        """
        Run single portfolio configuration.

        Args:
            name: Portfolio identifier
            long_weight: Long exposure (e.g., 1.0 = 100%, 1.3 = 130%)
            short_weight: Short exposure (e.g., 1.0 = 100%, 0.7 = 70%)
            long_only: If True, skip short selection

        Returns:
            Performance metrics dict
        """
        self.logger.info(f"\nRunning: {name}")
        self.logger.info(f"  Long: {long_weight*100:.0f}%, Short: {short_weight*100:.0f}%")
        self.logger.info(f"  Gross: {(long_weight+short_weight)*100:.0f}%")

        pnl_series = []
        turnover_series = []

        prev_positions = defaultdict(float)
        gross_target = long_weight + short_weight

        dates = sorted(self.df['date'].unique())

        for date in dates:
            day = self.df[self.df['date'] == date].copy()

            if len(day) < 5:
                continue

            # STEP 1: Rank by prediction
            day = day.sort_values('y_pred_reg', ascending=False)

            # STEP 2: Select universe (FIXED K)
            longs = day.head(self.K)
            shorts = day.tail(self.K) if not long_only else pd.DataFrame()

            # STEP 3: Raw target positions
            target_positions = {}

            # Long positions
            for _, row in longs.iterrows():
                target_positions[row['ticker']] = long_weight / self.K

            # Short positions
            if not long_only and len(shorts) > 0:
                for _, row in shorts.iterrows():
                    target_positions[row['ticker']] = -short_weight / self.K

            # STEP 4: EWMA smoothing (REQUIRED)
            if len(prev_positions) > 0:
                smoothed_positions = {}
                all_tickers = set(target_positions.keys()) | set(prev_positions.keys())

                for ticker in all_tickers:
                    target = target_positions.get(ticker, 0.0)
                    previous = prev_positions.get(ticker, 0.0)
                    # w_t = α * target + (1-α) * previous
                    smoothed = self.ewma_alpha * target + (1 - self.ewma_alpha) * previous

                    if abs(smoothed) > 0.001:
                        smoothed_positions[ticker] = smoothed
            else:
                smoothed_positions = target_positions

            positions = smoothed_positions

            # STEP 5: Calculate turnover (SPEC-CORRECT)
            # Denominator must be TARGET gross exposure, not previous
            traded = 0.0
            for ticker in set(prev_positions) | set(positions):
                traded += abs(positions.get(ticker, 0.0) - prev_positions.get(ticker, 0.0))

            turnover = traded / gross_target if gross_target > 1e-8 else 0.0
            # CRITICAL FIX: Costs should be on notional traded, not double-counted
            # For dollar-neutral: traded includes both sides, but each side is independent
            # Cost = (traded / 2) * tc, OR equivalently: cost on each leg separately
            # Phase 3.2 used (trade_amount / 2.0) * tc
            cost = (traded / 2.0) * self.tc

            # STEP 6: Calculate daily PnL
            pnl = 0.0
            returns_map = day.set_index('ticker')['y_true_reg'].to_dict()

            for ticker, pos in positions.items():
                if ticker in returns_map:
                    pnl += pos * returns_map[ticker]

            pnl -= cost  # Net of costs

            pnl_series.append(pnl)
            turnover_series.append(turnover)

            prev_positions = positions

        # Calculate metrics
        pnl = pd.Series(pnl_series)
        sharpe = pnl.mean() / pnl.std() * np.sqrt(TRADING_DAYS) if pnl.std() > 0 else 0.0

        # Drawdown
        cum_returns = pnl.cumsum()
        running_max = cum_returns.expanding().max()
        drawdown = cum_returns - running_max
        max_dd = drawdown.min()

        metrics = {
            "name": name,
            "sharpe": float(sharpe),
            "annual_return": float(pnl.mean() * TRADING_DAYS),
            "vol": float(pnl.std() * np.sqrt(TRADING_DAYS)),
            "avg_turnover": float(np.mean(turnover_series)),
            "max_dd": float(max_dd),
            "long_weight": long_weight,
            "short_weight": short_weight,
            "gross_exposure": long_weight + short_weight,
            "net_exposure": long_weight - short_weight
        }

        self.logger.info(f"  Results: Sharpe {sharpe:.2f}, Turnover {metrics['avg_turnover']*100:.1f}%")

        return metrics

    def run(self) -> pd.DataFrame:
        """
        Run all 3 portfolio configurations.

        Returns:
            DataFrame with results
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("PHASE 3.3: PORTFOLIO COMPARISON (SPEC-CORRECT)")
        self.logger.info("="*70)
        self.logger.info("")
        self.logger.info("FROZEN PARAMETERS:")
        self.logger.info(f"  K = {self.K} (fixed)")
        self.logger.info(f"  EWMA alpha = {self.ewma_alpha}")
        self.logger.info(f"  Transaction cost = {self.tc*10000:.1f} bps")
        self.logger.info("")

        results = []

        # P0: TRUE DOLLAR NEUTRAL (gross = 2.0)
        results.append(self.run_portfolio(
            name="P0_DollarNeutral",
            long_weight=1.0,
            short_weight=1.0
        ))

        # P1: LONG-ONLY (gross = 1.0)
        results.append(self.run_portfolio(
            name="P1_LongOnly",
            long_weight=1.0,
            short_weight=0.0,
            long_only=True
        ))

        # P2: LONG-BIASED (gross = 2.0, same as P0 for fair comparison)
        results.append(self.run_portfolio(
            name="P2_LongBiased",
            long_weight=1.3,
            short_weight=0.7
        ))

        return pd.DataFrame(results)

    def validate_results(self, results: pd.DataFrame) -> dict:
        """
        Apply HARD GATES to results.

        Returns:
            dict with pass/fail status and decision
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("VALIDATION GATES (HARD CONSTRAINTS)")
        self.logger.info("="*70)
        self.logger.info("")

        p0 = results[results['name'] == 'P0_DollarNeutral'].iloc[0]
        p1 = results[results['name'] == 'P1_LongOnly'].iloc[0]
        p2 = results[results['name'] == 'P2_LongBiased'].iloc[0]

        # GATE 1: Sharpe improvement
        sharpe_delta = p2['sharpe'] - p0['sharpe']
        gate1 = sharpe_delta >= 0.30

        self.logger.info("GATE 1: Sharpe(P2) - Sharpe(P0) >= 0.30")
        self.logger.info(f"  Actual: {sharpe_delta:.2f}")
        self.logger.info(f"  Status: {'PASS' if gate1 else 'FAIL'}")
        self.logger.info("")

        # GATE 2: Turnover control
        turnover_ratio = p2['avg_turnover'] / p0['avg_turnover'] if p0['avg_turnover'] > 0 else 999
        gate2 = turnover_ratio <= 1.5

        self.logger.info("GATE 2: Turnover(P2) <= 1.5 × Turnover(P0)")
        self.logger.info(f"  P0 Turnover: {p0['avg_turnover']*100:.1f}%")
        self.logger.info(f"  P2 Turnover: {p2['avg_turnover']*100:.1f}%")
        self.logger.info(f"  Ratio: {turnover_ratio:.2f}x")
        self.logger.info(f"  Status: {'PASS' if gate2 else 'FAIL'}")
        self.logger.info("")

        # GATE 3: Drawdown control
        dd_ratio = abs(p2['max_dd']) / abs(p0['max_dd']) if abs(p0['max_dd']) > 0 else 999
        gate3 = dd_ratio <= 1.5

        self.logger.info("GATE 3: MaxDD(P2) <= 1.5 × MaxDD(P0)")
        self.logger.info(f"  P0 MaxDD: {p0['max_dd']*100:.2f}%")
        self.logger.info(f"  P2 MaxDD: {p2['max_dd']*100:.2f}%")
        self.logger.info(f"  Ratio: {dd_ratio:.2f}x")
        self.logger.info(f"  Status: {'PASS' if gate3 else 'FAIL'}")
        self.logger.info("")

        # FINAL DECISION
        all_pass = gate1 and gate2 and gate3

        self.logger.info("="*70)
        if all_pass:
            self.logger.info("DECISION: DEPLOY P2 (Long-Biased)")
            self.logger.info(f"  Sharpe: {p2['sharpe']:.2f}")
            self.logger.info(f"  Annual Return: {p2['annual_return']*100:.2f}%")
            self.logger.info(f"  Turnover: {p2['avg_turnover']*100:.1f}%")
            self.logger.info("")
            self.logger.info("[GREEN LIGHT] Ready for Phase 3.4 (Execution Realism)")
            decision = "P2_LongBiased"
        else:
            self.logger.info("DECISION: DEPLOY P0 (Dollar Neutral)")
            self.logger.info(f"  Sharpe: {p0['sharpe']:.2f}")
            self.logger.info(f"  Annual Return: {p0['annual_return']*100:.2f}%")
            self.logger.info("")
            self.logger.info("[STOP] P2 did not pass all gates")
            decision = "P0_DollarNeutral"

        self.logger.info("="*70)

        return {
            'gate1_sharpe': gate1,
            'gate2_turnover': gate2,
            'gate3_drawdown': gate3,
            'all_pass': all_pass,
            'decision': decision,
            'sharpe_delta': sharpe_delta,
            'turnover_ratio': turnover_ratio,
            'dd_ratio': dd_ratio
        }


def main():
    """Run Phase 3.3 portfolio comparison experiment."""
    logger = setup_logger('phase3_3_portfolio', log_file='logs/phase3_3_portfolio_comparison.log', level='INFO')

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
    preds_file = Path("data/processed/phase1_predictions.parquet")
    if not preds_file.exists():
        logger.error(f"Predictions not found: {preds_file}")
        logger.error("Run Phase 1 or Phase 2A training first")
        return

    logger.info(f"Loading predictions from {preds_file}...")
    preds = pd.read_parquet(preds_file)
    logger.info(f"Loaded {len(preds)} predictions\n")

    # Run experiment
    experiment = Phase33PortfolioExperiment(
        predictions=preds,
        K=38,  # From Phase 2B
        transaction_cost_bps=5.0,
        ewma_alpha=0.15,
        logger=logger
    )

    results = experiment.run()

    # Display results
    logger.info("\n" + "="*70)
    logger.info("RESULTS SUMMARY")
    logger.info("="*70)
    logger.info("")
    logger.info(results[['name', 'sharpe', 'annual_return', 'vol', 'avg_turnover', 'max_dd']].to_string(index=False))
    logger.info("")

    # Validate with hard gates
    validation = experiment.validate_results(results)

    # Save results
    output_file = "data/processed/phase3_3_portfolio_comparison_results.json"

    def convert_to_json_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    output = {
        'results': results.to_dict('records'),
        'validation': validation
    }

    output_json = convert_to_json_serializable(output)

    with open(output_file, 'w') as f:
        import json
        json.dump(output_json, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")
    logger.info("\n" + "="*70)
    logger.info("PHASE 3.3 COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
