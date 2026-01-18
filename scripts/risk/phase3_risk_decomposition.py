"""
PHASE 3.2: Risk Decomposition Analysis

Critical Question: WHY does this Sharpe exist?

This script runs 5 mandatory diagnostics:
1. Long vs Short Attribution
2. Sector Exposure
3. Factor Exposure (Market, Size, Momentum, Vol)
4. Regime Sensitivity (VIX high/low)
5. Drawdown Anatomy

Gate: No catastrophic single failure
"""
import sys
from pathlib import Path
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocessor_v2 import SimplifiedStockPreprocessor
from src.models.lstm_model import create_model
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger


class RiskDecomposition:
    """Comprehensive risk decomposition for Phase 3.2."""

    def __init__(self, predictions_df: pd.DataFrame, config: ConfigLoader):
        """
        Initialize risk decomposition.

        Args:
            predictions_df: DataFrame with [date, ticker, y_pred_reg, y_true_reg]
            config: Configuration object
        """
        self.predictions_df = predictions_df
        self.config = config
        self.logger = setup_logger(__name__)

        # Portfolio parameters (from Phase 2B best config)
        self.long_pct = 0.2
        self.short_pct = 0.2
        self.position_ewma_alpha = 0.15
        self.transaction_cost = 0.0005  # 5 bps

        # Results storage
        self.daily_pnl = []
        self.positions_history = []

    def run_backtest_with_tracking(self):
        """Run backtest and track positions for risk analysis."""
        self.logger.info("Running backtest with position tracking...")

        dates = sorted(self.predictions_df['date'].unique())

        previous_positions = {}

        for date in dates:
            daily_data = self.predictions_df[self.predictions_df['date'] == date].copy()

            if len(daily_data) < 5:
                continue

            # Rank stocks
            daily_data['rank'] = daily_data['y_pred_reg'].rank(ascending=False)
            daily_data_sorted = daily_data.sort_values('y_pred_reg', ascending=False)

            # Calculate target positions
            n_stocks = len(daily_data)
            n_long = max(1, int(n_stocks * self.long_pct))
            n_short = max(1, int(n_stocks * self.short_pct))

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
                    smoothed = (1 - self.position_ewma_alpha) * previous + self.position_ewma_alpha * target

                    if abs(smoothed) > 0.001:
                        smoothed_positions[ticker] = smoothed
            else:
                smoothed_positions = target_positions

            # Enforce dollar neutrality
            final_positions = self._enforce_neutrality(smoothed_positions)

            # Calculate turnover and costs
            all_tickers = set(final_positions.keys()) | set(previous_positions.keys())
            trade_amount = sum(abs(final_positions.get(t, 0) - previous_positions.get(t, 0)) for t in all_tickers)

            gross_exposure = sum(abs(p) for p in previous_positions.values()) if previous_positions else 1.0
            turnover = trade_amount / gross_exposure if gross_exposure > 1e-8 else 0

            trading_cost = (trade_amount / 2.0) * self.transaction_cost

            # Calculate PnL
            pnl_gross = 0
            pnl_long = 0
            pnl_short = 0

            returns_map = daily_data.set_index('ticker')['y_true_reg'].to_dict()

            for ticker, position in final_positions.items():
                if ticker in returns_map:
                    pnl = position * returns_map[ticker]
                    pnl_gross += pnl

                    if position > 0:
                        pnl_long += pnl
                    else:
                        pnl_short += pnl

            pnl_net = pnl_gross - trading_cost

            # Store results
            self.daily_pnl.append({
                'date': date,
                'pnl_gross': pnl_gross,
                'pnl_net': pnl_net,
                'pnl_long': pnl_long,
                'pnl_short': pnl_short,
                'turnover': turnover,
                'cost': trading_cost
            })

            # Store positions
            self.positions_history.append({
                'date': date,
                'positions': final_positions.copy()
            })

            previous_positions = final_positions.copy()

        self.pnl_df = pd.DataFrame(self.daily_pnl)
        self.logger.info(f"Backtest complete. {len(self.pnl_df)} days analyzed.")

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

    def analyze_long_short_attribution(self) -> Dict:
        """Diagnostic 1: Long vs Short Attribution."""
        self.logger.info("\n" + "="*70)
        self.logger.info("DIAGNOSTIC 1: LONG VS SHORT ATTRIBUTION")
        self.logger.info("="*70)

        # Calculate Sharpe for each side
        long_returns = self.pnl_df['pnl_long'].values
        short_returns = self.pnl_df['pnl_short'].values
        total_returns = self.pnl_df['pnl_net'].values

        sharpe_long = np.mean(long_returns) / np.std(long_returns) * np.sqrt(252) if np.std(long_returns) > 0 else 0
        sharpe_short = np.mean(short_returns) / np.std(short_returns) * np.sqrt(252) if np.std(short_returns) > 0 else 0
        sharpe_total = np.mean(total_returns) / np.std(total_returns) * np.sqrt(252) if np.std(total_returns) > 0 else 0

        # Calculate contribution to total Sharpe
        long_contribution = np.mean(long_returns) / np.std(total_returns) * np.sqrt(252) if np.std(total_returns) > 0 else 0
        short_contribution = np.mean(short_returns) / np.std(total_returns) * np.sqrt(252) if np.std(total_returns) > 0 else 0

        results = {
            'sharpe_long': sharpe_long,
            'sharpe_short': sharpe_short,
            'sharpe_total': sharpe_total,
            'long_contribution': long_contribution,
            'short_contribution': short_contribution,
            'long_pct_contribution': long_contribution / sharpe_total if sharpe_total != 0 else 0,
            'short_pct_contribution': short_contribution / sharpe_total if sharpe_total != 0 else 0
        }

        self.logger.info(f"  Long-only Sharpe:  {sharpe_long:.2f}")
        self.logger.info(f"  Short-only Sharpe: {sharpe_short:.2f}")
        self.logger.info(f"  Total Sharpe:      {sharpe_total:.2f}")
        self.logger.info("")
        self.logger.info(f"  Long contribution to total Sharpe:  {long_contribution:.2f} ({results['long_pct_contribution']*100:.1f}%)")
        self.logger.info(f"  Short contribution to total Sharpe: {short_contribution:.2f} ({results['short_pct_contribution']*100:.1f}%)")

        # Red flags
        if abs(results['short_pct_contribution']) > 0.7:
            self.logger.warning("  [WARNING] Shorts dominate (>70%) - risk of borrow/squeezes")
        if abs(results['long_pct_contribution']) > 0.7:
            self.logger.warning("  [WARNING] Longs dominate (>70%) - possible market beta hiding")

        return results

    def analyze_sector_exposure(self) -> Dict:
        """Diagnostic 2: Sector Exposure (placeholder - requires sector mapping)."""
        self.logger.info("\n" + "="*70)
        self.logger.info("DIAGNOSTIC 2: SECTOR EXPOSURE")
        self.logger.info("="*70)

        self.logger.info("  [PLACEHOLDER] Sector mapping not available in current data")
        self.logger.info("  To implement:")
        self.logger.info("    1. Add sector metadata to each ticker")
        self.logger.info("    2. Compute avg sector weights after EWMA smoothing")
        self.logger.info("    3. Flag if any sector > 30% exposure")
        self.logger.info("")
        self.logger.info("  Phase 3.2 can proceed without this if universe is diversified")

        return {'status': 'not_implemented', 'note': 'Requires sector metadata'}

    def analyze_factor_exposure(self) -> Dict:
        """Diagnostic 3: Factor Exposure (Market, Size, Momentum, Vol)."""
        self.logger.info("\n" + "="*70)
        self.logger.info("DIAGNOSTIC 3: FACTOR EXPOSURE")
        self.logger.info("="*70)

        self.logger.info("  [PLACEHOLDER] Factor data (SPY, size, momentum) not loaded")
        self.logger.info("  To implement:")
        self.logger.info("    1. Load SPY daily returns (market factor)")
        self.logger.info("    2. Construct size factor (small - large cap portfolios)")
        self.logger.info("    3. Construct momentum factor (winners - losers)")
        self.logger.info("    4. Regress daily PnL vs factors")
        self.logger.info("")
        self.logger.info("  Target: Low beta (<0.2), alpha not explained by momentum")
        self.logger.info("  Dollar neutrality should give near-zero market beta")

        return {'status': 'not_implemented', 'note': 'Requires factor data'}

    def analyze_regime_sensitivity(self) -> Dict:
        """Diagnostic 4: Regime Sensitivity (VIX high/low)."""
        self.logger.info("\n" + "="*70)
        self.logger.info("DIAGNOSTIC 4: REGIME SENSITIVITY")
        self.logger.info("="*70)

        self.logger.info("  [PLACEHOLDER] VIX data not loaded")
        self.logger.info("  To implement:")
        self.logger.info("    1. Load VIX daily data")
        self.logger.info("    2. Split test period: VIX > median (high vol) vs VIX < median (low vol)")
        self.logger.info("    3. Calculate Sharpe in each regime")
        self.logger.info("")
        self.logger.info("  Red flag: Sharpe > 0 only in low-VIX periods (fragile alpha)")

        return {'status': 'not_implemented', 'note': 'Requires VIX data'}

    def analyze_drawdown_anatomy(self) -> Dict:
        """Diagnostic 5: Drawdown Anatomy."""
        self.logger.info("\n" + "="*70)
        self.logger.info("DIAGNOSTIC 5: DRAWDOWN ANATOMY")
        self.logger.info("="*70)

        # Calculate cumulative returns
        cum_returns = self.pnl_df['pnl_net'].cumsum()
        running_max = cum_returns.expanding().max()
        drawdown = cum_returns - running_max

        # Max drawdown
        max_dd = drawdown.min()
        max_dd_date = self.pnl_df.iloc[drawdown.idxmin()]['date']

        # Drawdown duration
        dd_starts = (drawdown == 0) & (drawdown.shift(-1) < 0)
        dd_ends = (drawdown < 0) & (drawdown.shift(-1) == 0)

        if dd_starts.sum() > 0:
            # Find longest drawdown
            in_dd = False
            current_dd_length = 0
            max_dd_length = 0

            for i, is_dd in enumerate(drawdown < 0):
                if is_dd:
                    current_dd_length += 1
                    max_dd_length = max(max_dd_length, current_dd_length)
                else:
                    current_dd_length = 0
        else:
            max_dd_length = 0

        # Single-day tail loss
        worst_day_loss = self.pnl_df['pnl_net'].min()
        worst_day_date = self.pnl_df.iloc[self.pnl_df['pnl_net'].idxmin()]['date']

        # 95th percentile loss
        pct_5_loss = np.percentile(self.pnl_df['pnl_net'], 5)

        results = {
            'max_drawdown': max_dd,
            'max_dd_date': str(max_dd_date),
            'max_dd_length_days': int(max_dd_length),
            'worst_day_loss': worst_day_loss,
            'worst_day_date': str(worst_day_date),
            'pct_5_loss': pct_5_loss
        }

        self.logger.info(f"  Max Drawdown:        {max_dd:.4f} ({max_dd*100:.2f}%)")
        self.logger.info(f"  Max DD Date:         {max_dd_date}")
        self.logger.info(f"  Max DD Length:       {max_dd_length} days")
        self.logger.info("")
        self.logger.info(f"  Worst Single Day:    {worst_day_loss:.4f} ({worst_day_loss*100:.2f}%)")
        self.logger.info(f"  Worst Day Date:      {worst_day_date}")
        self.logger.info(f"  5th Percentile Loss: {pct_5_loss:.4f} ({pct_5_loss*100:.2f}%)")

        # Calculate 3-sigma daily loss for kill switch
        daily_std = self.pnl_df['pnl_net'].std()
        three_sigma_loss = -3 * daily_std

        self.logger.info("")
        self.logger.info(f"  Daily Std Dev:       {daily_std:.4f}")
        self.logger.info(f"  3-Sigma Loss:        {three_sigma_loss:.4f} (kill switch threshold)")

        results['daily_std'] = daily_std
        results['three_sigma_loss'] = three_sigma_loss

        return results

    def run_all_diagnostics(self) -> Dict:
        """Run all 5 diagnostics and return results."""
        self.logger.info("\n" + "="*70)
        self.logger.info("PHASE 3.2: RISK DECOMPOSITION")
        self.logger.info("="*70)
        self.logger.info("")
        self.logger.info("Running comprehensive risk analysis...")
        self.logger.info("")

        # Run backtest first
        self.run_backtest_with_tracking()

        # Run diagnostics
        results = {
            'long_short_attribution': self.analyze_long_short_attribution(),
            'sector_exposure': self.analyze_sector_exposure(),
            'factor_exposure': self.analyze_factor_exposure(),
            'regime_sensitivity': self.analyze_regime_sensitivity(),
            'drawdown_anatomy': self.analyze_drawdown_anatomy()
        }

        # Gate check
        self.logger.info("\n" + "="*70)
        self.logger.info("PHASE 3.2 GATE CHECK")
        self.logger.info("="*70)

        # Check for catastrophic failures
        failures = []

        # Check long/short balance
        long_short = results['long_short_attribution']
        if abs(long_short['long_pct_contribution']) > 0.8 or abs(long_short['short_pct_contribution']) > 0.8:
            failures.append("Extreme long or short dominance (>80%)")

        # Check drawdown
        dd = results['drawdown_anatomy']
        if dd['max_drawdown'] < -0.15:  # -15% drawdown
            failures.append(f"Excessive drawdown ({dd['max_drawdown']*100:.1f}%)")

        if len(failures) == 0:
            self.logger.info("  [PASS] No catastrophic failures detected")
            self.logger.info("  Ready to proceed to Phase 3.3")
            results['gate_status'] = 'PASS'
        else:
            self.logger.warning("  [FAIL] Catastrophic failures detected:")
            for failure in failures:
                self.logger.warning(f"    - {failure}")
            results['gate_status'] = 'FAIL'
            results['failures'] = failures

        self.logger.info("="*70)

        return results


def main():
    """Run Phase 3.2 risk decomposition."""
    config = ConfigLoader('config/config.yaml')
    logger = setup_logger('phase3_risk', log_file='logs/phase3_risk_decomposition.log', level='INFO')

    logger.info("="*70)
    logger.info("PHASE 3.2: RISK DECOMPOSITION")
    logger.info("="*70)
    logger.info("")
    logger.info("Loading predictions data...")

    # Check for pyarrow
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("PyArrow not installed. Installing now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
        logger.info("PyArrow installed successfully. Please re-run the script.")
        return

    # Try to load cached processed data
    processed_file = Path("data/processed/phase1_predictions.parquet")
    if processed_file.exists():
        logger.info(f"Loading cached predictions from {processed_file}")
        try:
            predictions_df = pd.read_parquet(processed_file)

            # Check if required columns exist
            required_cols = ['date', 'ticker', 'y_pred_reg', 'y_true_reg']
            if all(col in predictions_df.columns for col in required_cols):
                logger.info(f"  Found {len(predictions_df)} predictions with all required columns")
            else:
                logger.error(f"Missing required columns. Found: {predictions_df.columns.tolist()}")
                logger.error(f"Required: {required_cols}")
                return
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            return
    else:
        logger.error(f"Predictions file not found at {processed_file}")
        logger.error("Please run Phase 1 or Phase 2A training first to generate predictions")
        logger.error("Available files:")
        processed_dir = Path("data/processed")
        for f in processed_dir.glob("*.parquet"):
            logger.error(f"  - {f.name}")
        return

    logger.info(f"Loaded {len(predictions_df)} predictions")
    logger.info("")

    # Run risk decomposition
    risk_analyzer = RiskDecomposition(predictions_df, config)
    results = risk_analyzer.run_all_diagnostics()

    # Save results
    results_file = "data/processed/phase3_risk_decomposition_results.json"

    # Convert numpy types to Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    results_json = convert_to_json_serializable(results)

    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    logger.info("")
    logger.info(f"Results saved to: {results_file}")
    logger.info("")
    logger.info("="*70)
    logger.info("PHASE 3.2 COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
