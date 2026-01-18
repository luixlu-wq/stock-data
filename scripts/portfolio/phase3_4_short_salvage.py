"""
PHASE 3.4: SHORT-LEG SALVAGE EXPERIMENTS

Empirical test: Can improved short construction rescue 130/30?

Phase 3.3 rejected P2 (130/30) due to:
- Insufficient Sharpe improvement (0.25 < 0.30 gate)
- Excessive drawdown (2.59x > 1.5x gate)

This phase tests 5 short construction methods:
S1: Bottom-K (baseline, already tested)
S2: Filter negative predictions only
S3: Vol-scaled shorts
S4: SPY hedge
S5: Long-only (control)

PASS CRITERIA:
- Any S config must beat P2 baseline (Sharpe 0.98)
- AND pass Phase 3.3 gates when tested as P2_improved
- Otherwise → accept P0 or P1 deployment
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

TRADING_DAYS = 252


class ShortMode(Enum):
    """Short leg construction modes."""
    BOTTOM_K = "bottom_k"           # Baseline: bottom-K stocks
    FILTER_NEGATIVE = "filter_neg"  # Only stocks with y_pred < 0
    VOL_SCALED = "vol_scaled"       # Inverse vol weighting
    SPY_HEDGE = "spy_hedge"         # Market beta hedge
    NONE = "none"                   # Long-only control


class Phase34ShortSalvage:
    """Test short leg construction methods."""

    def __init__(
        self,
        predictions: pd.DataFrame,
        K: int = 38,
        transaction_cost_bps: float = 5.0,
        ewma_alpha: float = 0.15,
        logger=None
    ):
        self.df = predictions.copy()
        self.K = K
        self.tc = transaction_cost_bps / 10_000
        self.ewma_alpha = ewma_alpha
        self.logger = logger or setup_logger(__name__)

        # Calculate rolling volatility for vol-scaled shorts
        self._calculate_volatility()

        # Estimate SPY beta for hedge mode
        self.spy_beta = None

    def _calculate_volatility(self):
        """Calculate 20-day rolling volatility per ticker."""
        self.logger.info("Calculating rolling volatility for vol-scaling...")

        # Group by ticker and calculate rolling vol
        vol_results = []

        for ticker in self.df['ticker'].unique():
            ticker_data = self.df[self.df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date')

            # Rolling std of returns (20-day window)
            ticker_data['vol_20d'] = ticker_data['y_true_reg'].rolling(
                window=20, min_periods=10
            ).std()

            # Fill initial NaN with overall std
            ticker_data['vol_20d'] = ticker_data['vol_20d'].fillna(
                ticker_data['y_true_reg'].std()
            )

            vol_results.append(ticker_data)

        self.df = pd.concat(vol_results, ignore_index=True)
        self.logger.info(f"Volatility calculated for {len(self.df)} rows")

    def _estimate_spy_beta(self) -> float:
        """Estimate portfolio beta to SPY (simplified)."""
        # For this salvage experiment, use a fixed conservative estimate
        # In production, would regress daily PnL vs SPY returns
        return 0.15  # Conservative estimate based on Phase 3.2b findings

    def run_portfolio(
        self,
        name: str,
        long_weight: float,
        short_weight: float,
        short_mode: ShortMode = ShortMode.BOTTOM_K
    ) -> dict:
        """
        Run portfolio with specified short construction mode.

        Args:
            name: Portfolio identifier
            long_weight: Long exposure
            short_weight: Short exposure
            short_mode: How to construct short leg

        Returns:
            Performance metrics dict
        """
        self.logger.info(f"\nRunning: {name}")
        self.logger.info(f"  Long: {long_weight*100:.0f}%, Short: {short_weight*100:.0f}%")
        self.logger.info(f"  Short mode: {short_mode.value}")

        pnl_series = []
        turnover_series = []
        long_pnl_series = []
        short_pnl_series = []
        dates_traded = []  # Track actual trading dates
        positions_history = []  # Track daily positions

        prev_positions = defaultdict(float)
        gross_target = long_weight + short_weight

        dates = sorted(self.df['date'].unique())

        for date in dates:
            day = self.df[self.df['date'] == date].copy()

            if len(day) < 5:
                continue

            day = day.sort_values('y_pred_reg', ascending=False)

            # LONG LEG: Always top-K
            longs = day.head(self.K)

            # SHORT LEG: Varies by mode
            target_positions = {}

            # Long positions (always equal weight)
            for _, row in longs.iterrows():
                target_positions[row['ticker']] = long_weight / self.K

            # Short positions (mode-dependent)
            if short_weight > 0:
                if short_mode == ShortMode.BOTTOM_K:
                    # Baseline: bottom-K stocks
                    shorts = day.tail(self.K)
                    for _, row in shorts.iterrows():
                        target_positions[row['ticker']] = -short_weight / self.K

                elif short_mode == ShortMode.FILTER_NEGATIVE:
                    # Only short stocks with negative predictions
                    shorts = day[day['y_pred_reg'] < 0].tail(self.K)
                    if len(shorts) > 0:
                        for _, row in shorts.iterrows():
                            target_positions[row['ticker']] = -short_weight / len(shorts)

                elif short_mode == ShortMode.VOL_SCALED:
                    # Vol-scaled: inverse volatility weighting
                    shorts = day[day['y_pred_reg'] < 0].tail(self.K).copy()
                    if len(shorts) > 0:
                        shorts['inv_vol'] = 1.0 / shorts['vol_20d'].clip(lower=1e-4)
                        shorts['weight'] = shorts['inv_vol'] / shorts['inv_vol'].sum()

                        for _, row in shorts.iterrows():
                            target_positions[row['ticker']] = -short_weight * row['weight']

                elif short_mode == ShortMode.SPY_HEDGE:
                    # SPY hedge: short SPY based on beta
                    if self.spy_beta is None:
                        self.spy_beta = self._estimate_spy_beta()

                    # Note: Would need SPY in predictions to execute this properly
                    # For now, simulate by reducing short exposure proportionally
                    # In production: target_positions['SPY'] = -long_weight * self.spy_beta
                    self.logger.warning("SPY hedge mode requires SPY data - using reduced shorts instead")
                    shorts = day.tail(int(self.K * 0.5))  # Reduced universe
                    for _, row in shorts.iterrows():
                        target_positions[row['ticker']] = -short_weight / len(shorts)

            # EWMA smoothing
            if len(prev_positions) > 0:
                smoothed_positions = {}
                all_tickers = set(target_positions.keys()) | set(prev_positions.keys())

                for ticker in all_tickers:
                    target = target_positions.get(ticker, 0.0)
                    previous = prev_positions.get(ticker, 0.0)
                    smoothed = self.ewma_alpha * target + (1 - self.ewma_alpha) * previous

                    if abs(smoothed) > 0.001:
                        smoothed_positions[ticker] = smoothed
            else:
                smoothed_positions = target_positions

            positions = smoothed_positions

            # Turnover
            traded = 0.0
            for ticker in set(prev_positions) | set(positions):
                traded += abs(positions.get(ticker, 0.0) - prev_positions.get(ticker, 0.0))

            turnover = traded / gross_target if gross_target > 1e-8 else 0.0
            cost = (traded / 2.0) * self.tc  # Corrected cost calculation

            # PnL (separated by long/short)
            pnl = 0.0
            long_pnl = 0.0
            short_pnl = 0.0
            returns_map = day.set_index('ticker')['y_true_reg'].to_dict()

            for ticker, pos in positions.items():
                if ticker in returns_map:
                    ret = pos * returns_map[ticker]
                    pnl += ret

                    if pos > 0:
                        long_pnl += ret
                    else:
                        short_pnl += ret

            pnl -= cost

            pnl_series.append(pnl)
            turnover_series.append(turnover)
            long_pnl_series.append(long_pnl)
            short_pnl_series.append(short_pnl)
            dates_traded.append(date)
            positions_history.append(positions.copy())

            prev_positions = positions

        # Calculate metrics
        pnl = pd.Series(pnl_series)
        long_pnl = pd.Series(long_pnl_series)
        short_pnl = pd.Series(short_pnl_series)

        sharpe = pnl.mean() / pnl.std() * np.sqrt(TRADING_DAYS) if pnl.std() > 0 else 0.0
        long_sharpe = long_pnl.mean() / long_pnl.std() * np.sqrt(TRADING_DAYS) if long_pnl.std() > 0 else 0.0
        short_sharpe = short_pnl.mean() / short_pnl.std() * np.sqrt(TRADING_DAYS) if short_pnl.std() > 0 else 0.0

        # Drawdown
        cum_returns = pnl.cumsum()
        running_max = cum_returns.expanding().max()
        drawdown = cum_returns - running_max
        max_dd = drawdown.min()

        # Short contribution analysis
        short_contribution_pct = (short_pnl.sum() / long_pnl.sum() * 100) if long_pnl.sum() > 0 else 0

        metrics = {
            "name": name,
            "sharpe": float(sharpe),
            "long_sharpe": float(long_sharpe),
            "short_sharpe": float(short_sharpe),
            "annual_return": float(pnl.mean() * TRADING_DAYS),
            "vol": float(pnl.std() * np.sqrt(TRADING_DAYS)),
            "avg_turnover": float(np.mean(turnover_series)),
            "max_dd": float(max_dd),
            "long_weight": long_weight,
            "short_weight": short_weight,
            "gross_exposure": long_weight + short_weight,
            "short_mode": short_mode.value,
            "short_contribution_pct": float(short_contribution_pct),
            # Store detailed history for Phase 3.5
            "daily_history": {
                "dates": dates_traded,
                "pnl": pnl_series,
                "long_pnl": long_pnl_series,
                "short_pnl": short_pnl_series,
                "turnover": turnover_series,
                "positions": positions_history
            }
        }

        self.logger.info(f"  Results: Sharpe {sharpe:.2f}, Short Contrib {short_contribution_pct:.1f}%")

        return metrics

    def run_all_experiments(self) -> pd.DataFrame:
        """Run all 5 short construction experiments."""
        self.logger.info("\n" + "="*70)
        self.logger.info("PHASE 3.4: SHORT-LEG SALVAGE EXPERIMENTS")
        self.logger.info("="*70)
        self.logger.info("")
        self.logger.info("Testing 5 short construction methods to salvage 130/30 structure")
        self.logger.info("")

        results = []

        # S1: Bottom-K (baseline - already tested in Phase 3.3)
        results.append(self.run_portfolio(
            name="S1_BottomK",
            long_weight=1.3,
            short_weight=0.7,
            short_mode=ShortMode.BOTTOM_K
        ))

        # S2: Filter negative predictions
        results.append(self.run_portfolio(
            name="S2_FilterNegative",
            long_weight=1.3,
            short_weight=0.7,
            short_mode=ShortMode.FILTER_NEGATIVE
        ))

        # S3: Vol-scaled shorts
        results.append(self.run_portfolio(
            name="S3_VolScaled",
            long_weight=1.3,
            short_weight=0.7,
            short_mode=ShortMode.VOL_SCALED
        ))

        # S4: SPY hedge (simulated)
        results.append(self.run_portfolio(
            name="S4_SPYHedge",
            long_weight=1.3,
            short_weight=0.7,
            short_mode=ShortMode.SPY_HEDGE
        ))

        # S5: Long-only (control)
        results.append(self.run_portfolio(
            name="S5_LongOnly",
            long_weight=1.0,
            short_weight=0.0,
            short_mode=ShortMode.NONE
        ))

        return pd.DataFrame(results)

    def analyze_results(self, results: pd.DataFrame, baseline_sharpe: float = 0.98):
        """Analyze salvage experiment results."""
        self.logger.info("\n" + "="*70)
        self.logger.info("SALVAGE EXPERIMENT RESULTS")
        self.logger.info("="*70)
        self.logger.info("")

        # Sort by Sharpe
        results_sorted = results.sort_values('sharpe', ascending=False)

        self.logger.info(results_sorted[[
            'name', 'sharpe', 'long_sharpe', 'short_sharpe',
            'avg_turnover', 'max_dd', 'short_contribution_pct'
        ]].to_string(index=False))
        self.logger.info("")

        # Find best short construction method
        short_methods = results_sorted[results_sorted['short_weight'] > 0]
        best_short = short_methods.iloc[0] if len(short_methods) > 0 else None

        self.logger.info("="*70)
        self.logger.info("ANALYSIS")
        self.logger.info("="*70)
        self.logger.info("")

        if best_short is not None:
            self.logger.info(f"BEST SHORT METHOD: {best_short['name']}")
            self.logger.info(f"  Sharpe: {best_short['sharpe']:.2f}")
            self.logger.info(f"  Short Sharpe: {best_short['short_sharpe']:.2f}")
            self.logger.info(f"  Short Contribution: {best_short['short_contribution_pct']:.1f}%")
            self.logger.info(f"  Max DD: {best_short['max_dd']*100:.2f}%")
            self.logger.info("")

            # Compare to baseline (P2 from Phase 3.3)
            improvement = best_short['sharpe'] - baseline_sharpe

            self.logger.info(f"COMPARISON TO BASELINE (P2 from Phase 3.3):")
            self.logger.info(f"  Baseline Sharpe: {baseline_sharpe:.2f}")
            self.logger.info(f"  Best Method Sharpe: {best_short['sharpe']:.2f}")
            self.logger.info(f"  Improvement: {improvement:+.2f}")
            self.logger.info("")

            # Verdict
            if improvement > 0.1:
                self.logger.info("VERDICT: SHORT SALVAGE SUCCESSFUL")
                self.logger.info(f"  {best_short['name']} shows meaningful improvement")
                self.logger.info("  RECOMMENDATION: Rerun Phase 3.3 gates with this method")
            elif improvement > 0:
                self.logger.info("VERDICT: MARGINAL IMPROVEMENT")
                self.logger.info("  Improvement exists but may not pass Phase 3.3 gates")
                self.logger.info("  RECOMMENDATION: Test against gates before deployment")
            else:
                self.logger.info("VERDICT: SHORT SALVAGE FAILED")
                self.logger.info("  No short construction method beats baseline")
                self.logger.info("  RECOMMENDATION: Deploy P0 (Dollar Neutral) or P1 (Long-Only)")
        else:
            self.logger.info("ERROR: No short methods found")

        self.logger.info("")

        # Long-only comparison
        long_only = results[results['name'] == 'S5_LongOnly'].iloc[0]
        self.logger.info("LONG-ONLY PERFORMANCE:")
        self.logger.info(f"  Sharpe: {long_only['sharpe']:.2f}")
        self.logger.info(f"  Max DD: {long_only['max_dd']*100:.2f}%")
        self.logger.info("")

        if best_short is not None:
            if long_only['sharpe'] > best_short['sharpe']:
                self.logger.info("  -> Long-only BEATS best short method")
                self.logger.info("  -> Shorts are destructive regardless of construction")
            else:
                self.logger.info("  -> Best short method BEATS long-only")
                self.logger.info("  -> Shorts provide value with proper construction")

        self.logger.info("")
        self.logger.info("="*70)


def main():
    """Run Phase 3.4 short-leg salvage experiments."""
    logger = setup_logger('phase3_4_salvage', log_file='logs/phase3_4_short_salvage.log', level='INFO')

    # Load predictions
    preds_file = Path("data/processed/phase1_predictions.parquet")
    if not preds_file.exists():
        logger.error(f"Predictions not found: {preds_file}")
        return

    logger.info(f"Loading predictions from {preds_file}...")
    preds = pd.read_parquet(preds_file)
    logger.info(f"Loaded {len(preds)} predictions\n")

    # Run salvage experiments
    salvage = Phase34ShortSalvage(
        predictions=preds,
        K=38,
        transaction_cost_bps=5.0,
        ewma_alpha=0.15,
        logger=logger
    )

    results = salvage.run_all_experiments()

    # Analyze
    salvage.analyze_results(results, baseline_sharpe=0.98)  # P2 baseline from Phase 3.3

    # Extract S2 (winner) and save detailed history for Phase 3.5
    s2_result = results[results['name'] == 'S2_FilterNegative'].iloc[0]
    s2_history = s2_result['daily_history']

    # Save S2 daily history to parquet for Phase 3.5
    s2_history_df = pd.DataFrame({
        'date': s2_history['dates'],
        'pnl': s2_history['pnl'],
        'long_pnl': s2_history['long_pnl'],
        'short_pnl': s2_history['short_pnl'],
        'turnover': s2_history['turnover']
    })

    s2_history_file = "data/processed/s2_daily_pnl_history.parquet"
    s2_history_df.to_parquet(s2_history_file)
    logger.info(f"\nS2 daily history saved to: {s2_history_file}")
    logger.info(f"  (Required for Phase 3.5 vol targeting)")

    # Save results
    output_file = "data/processed/phase3_4_short_salvage_results.json"

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

    # Remove daily_history from results before saving to JSON (too large)
    # (already saved to parquet separately)
    results_for_json = results.copy()
    for idx in results_for_json.index:
        if 'daily_history' in results_for_json.at[idx, 'name']:
            results_for_json.at[idx, 'daily_history'] = "Saved separately to s2_daily_pnl_history.parquet"
        elif isinstance(results_for_json.at[idx, 'daily_history'], dict):
            results_for_json.at[idx, 'daily_history'] = "Excluded from JSON (see parquet file)"

    output = {
        'results': results_for_json.to_dict('records'),
        'baseline_sharpe': 0.98,
        'conclusion': 'See logs for verdict',
        's2_history_file': 's2_daily_pnl_history.parquet'
    }

    output_json = convert_to_json_serializable(output)

    with open(output_file, 'w') as f:
        import json
        json.dump(output_json, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")
    logger.info("\n" + "="*70)
    logger.info("PHASE 3.4 COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
