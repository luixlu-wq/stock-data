"""
PHASE 2B: Portfolio Monetization Engineering (NOT MORE ML)

ChatGPT's Critical Insight:
- Temperature = 0.05 produces Gross Sharpe 2.47, Net Sharpe 0.10
- This is REAL, FRAGILE, CROSS-SECTIONAL ALPHA
- Problem: High turnover (costs destroy profit)
- Solution: Portfolio engineering, NOT model changes

🔒 FROZEN (No more changes):
- Model architecture
- Features (14)
- Sequence length (90)
- Temperature (0.05)
- Rank weight (0.7)

Phase 2B Techniques:
1. Rank-Change Trading Filter (only trade large rank changes)
2. Position EWMA Smoothing (reduce churn)
3. Cross-Sectional Neutralization (z-score, dollar neutral)

Expected Outcome:
- Gross Sharpe: 1.8-2.3 (down from 2.47 but stable)
- Turnover: 60-75% (down from ~100%)
- Net Sharpe: 0.4-0.6 (UP from 0.10)
- TRADEABLE!
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocessor_v2 import SimplifiedStockPreprocessor
from src.models.lstm_model import create_model
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger


class MonetizationBacktest:
    """Advanced backtest with portfolio engineering techniques."""

    def __init__(
        self,
        long_pct: float = 0.2,
        short_pct: float = 0.2,
        transaction_cost_bps: float = 5.0,
        # Phase 2B parameters
        rank_change_threshold: int = 10,  # Only trade if rank changes by this much
        position_ewma_alpha: float = 0.15,  # EWMA weight for position smoothing
        use_cross_sectional_zscore: bool = True
    ):
        self.long_pct = long_pct
        self.short_pct = short_pct
        self.transaction_cost = transaction_cost_bps / 10000
        self.rank_change_threshold = rank_change_threshold
        self.position_ewma_alpha = position_ewma_alpha
        self.use_cross_sectional_zscore = use_cross_sectional_zscore

        self.logger = get_logger(__name__)

    def apply_rank_change_filter(
        self,
        smoothed_positions: dict,
        current_ranks: pd.Series,
        previous_ranks: pd.Series,
        previous_positions: dict
    ) -> dict:
        """
        FIXED (ChatGPT Bug #1): Only trade stocks whose rank changed significantly.

        CRITICAL: If rank change < threshold, FREEZE position completely.
        Don't allow any trading or resizing.

        Args:
            smoothed_positions: Desired positions after EWMA
            current_ranks: Current day rankings
            previous_ranks: Previous day rankings
            previous_positions: {ticker: position} from yesterday

        Returns:
            Filtered positions (trade only if rank changed enough)
        """
        final_positions = {}

        for ticker, target_pos in smoothed_positions.items():
            # If new stock, allow trading
            if ticker not in previous_ranks.index:
                final_positions[ticker] = target_pos
                continue

            # Calculate rank change
            rank_change = abs(current_ranks.get(ticker, 0) - previous_ranks.get(ticker, 0))

            # Only update position if rank changed significantly
            if rank_change >= self.rank_change_threshold:
                final_positions[ticker] = target_pos
            else:
                # FREEZE: Keep previous position, no trading
                if ticker in previous_positions:
                    final_positions[ticker] = previous_positions[ticker]
                # else: don't add to portfolio

        return final_positions

    def apply_position_smoothing(
        self,
        target_positions: dict,
        previous_positions: dict
    ) -> dict:
        """
        Smooth positions using EWMA to reduce churn.

        pos_t = (1-alpha) * pos_{t-1} + alpha * target_t

        Args:
            target_positions: {ticker: target_position}
            previous_positions: {ticker: previous_position}

        Returns:
            Smoothed positions
        """
        smoothed = {}

        all_tickers = set(target_positions.keys()) | set(previous_positions.keys())

        for ticker in all_tickers:
            target = target_positions.get(ticker, 0)
            previous = previous_positions.get(ticker, 0)

            # EWMA smoothing
            smoothed_pos = (1 - self.position_ewma_alpha) * previous + self.position_ewma_alpha * target

            # Only include if position is significant
            if abs(smoothed_pos) > 0.001:
                smoothed[ticker] = smoothed_pos

        return smoothed

    def enforce_dollar_neutrality(self, positions: dict) -> dict:
        """
        FIXED (ChatGPT Bug #3): Enforce dollar neutrality.

        Z-scoring predictions does NOT guarantee dollar neutrality
        because long/short counts differ after filtering.

        Args:
            positions: {ticker: position}

        Returns:
            Dollar-neutral positions
        """
        if not positions:
            return positions

        # Separate long and short
        total_long = sum(p for p in positions.values() if p > 0)
        total_short = abs(sum(p for p in positions.values() if p < 0))

        if total_long < 1e-8 or total_short < 1e-8:
            return positions

        # Scale to enforce neutrality
        neutral_positions = {}
        for ticker, pos in positions.items():
            if pos > 0:
                neutral_positions[ticker] = pos / total_long * 0.5  # Long side = 50%
            else:
                neutral_positions[ticker] = pos / total_short * 0.5  # Short side = -50%

        return neutral_positions

    def run_backtest(self, predictions_df: pd.DataFrame) -> dict:
        """
        Run backtest with Phase 2B monetization techniques.

        Args:
            predictions_df: DataFrame with [date, ticker, y_pred_reg, y_true_reg]

        Returns:
            Performance metrics
        """
        self.logger.info("="*70)
        self.logger.info("PHASE 2B: MONETIZATION BACKTEST")
        self.logger.info("="*70)
        self.logger.info(f"Rank change threshold: {self.rank_change_threshold} percentiles")
        self.logger.info(f"Position EWMA alpha: {self.position_ewma_alpha}")
        self.logger.info(f"Cross-sectional z-score: {self.use_cross_sectional_zscore}")
        self.logger.info("")

        df = predictions_df.copy()

        # Cross-sectional z-scoring (normalize predictions per day)
        if self.use_cross_sectional_zscore:
            df['y_pred_reg'] = df.groupby('date')['y_pred_reg'].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )

        # Sort by date
        df = df.sort_values('date')
        dates = sorted(df['date'].unique())

        self.logger.info(f"Simulating {len(dates)} trading days...")

        # Track state
        previous_positions = {}  # {ticker: position}
        previous_ranks = pd.Series(dtype=float)
        daily_pnl = []
        daily_turnover = []
        daily_costs = []

        for date in dates:
            daily_data = df[df['date'] == date].copy()

            if len(daily_data) < 5:
                continue

            # Rank stocks by prediction
            daily_data['rank'] = daily_data['y_pred_reg'].rank(ascending=False)
            current_ranks = daily_data.set_index('ticker')['rank']

            # STEP 1: Calculate raw target positions
            n_stocks = len(daily_data)
            n_long = max(1, int(n_stocks * self.long_pct))
            n_short = max(1, int(n_stocks * self.short_pct))

            daily_data_sorted = daily_data.sort_values('y_pred_reg', ascending=False)

            target_positions = {}

            # Long positions (equal-weighted)
            long_tickers = daily_data_sorted.iloc[:n_long]['ticker'].tolist()
            for ticker in long_tickers:
                target_positions[ticker] = 1.0 / n_long

            # Short positions (equal-weighted)
            short_tickers = daily_data_sorted.iloc[-n_short:]['ticker'].tolist()
            for ticker in short_tickers:
                target_positions[ticker] = -1.0 / n_short

            # STEP 2: Apply position smoothing (EWMA)
            # FIXED (ChatGPT Bug #2): Apply EWMA BEFORE filtering
            if len(previous_positions) > 0:
                smoothed_positions = self.apply_position_smoothing(
                    target_positions, previous_positions
                )
            else:
                smoothed_positions = target_positions

            # STEP 3: Apply rank-change filter (as a gate)
            # FIXED (ChatGPT Bug #1 & #2): Filter AFTER smoothing
            if len(previous_ranks) > 0 and self.rank_change_threshold > 0:
                final_positions = self.apply_rank_change_filter(
                    smoothed_positions, current_ranks, previous_ranks, previous_positions
                )
            else:
                final_positions = smoothed_positions

            # STEP 4: Enforce dollar neutrality
            # FIXED (ChatGPT Bug #3): Explicit dollar neutrality
            final_positions = self.enforce_dollar_neutrality(final_positions)

            # STEP 5: Calculate turnover (FIXED Bug #4)
            # Normalize by gross exposure, not fixed 2.0
            gross_exposure = sum(abs(p) for p in previous_positions.values())
            if gross_exposure < 1e-8:
                gross_exposure = 1.0  # First day

            all_tickers = set(final_positions.keys()) | set(previous_positions.keys())
            trade_amount = sum(
                abs(final_positions.get(t, 0) - previous_positions.get(t, 0))
                for t in all_tickers
            )
            turnover = trade_amount / gross_exposure

            daily_turnover.append(turnover)

            # Calculate trading costs
            trading_cost = (trade_amount / 2.0) * self.transaction_cost  # Divide by 2 for one-way
            daily_costs.append(trading_cost)

            # Calculate PnL
            pnl_gross = 0
            returns_map = daily_data.set_index('ticker')['y_true_reg'].to_dict()

            for ticker, position in final_positions.items():
                if ticker in returns_map:
                    pnl_gross += position * returns_map[ticker]

            pnl_net = pnl_gross - trading_cost

            daily_pnl.append({
                'date': date,
                'pnl_gross': pnl_gross,
                'pnl_net': pnl_net,
                'turnover': turnover,
                'cost': trading_cost
            })

            # Update state
            previous_positions = final_positions.copy()
            previous_ranks = current_ranks.copy()

        # Calculate metrics
        pnl_df = pd.DataFrame(daily_pnl)

        metrics = self._calculate_metrics(pnl_df, daily_turnover)

        self._print_metrics(metrics)

        return metrics

    def _calculate_metrics(self, pnl_df: pd.DataFrame, turnovers: list) -> dict:
        """Calculate performance metrics."""

        gross_returns = pnl_df['pnl_gross'].values
        net_returns = pnl_df['pnl_net'].values

        # Annualize (252 trading days)
        annual_gross = np.mean(gross_returns) * 252
        annual_net = np.mean(net_returns) * 252

        # Sharpe ratios
        sharpe_gross = np.mean(gross_returns) / (np.std(gross_returns) + 1e-8) * np.sqrt(252)
        sharpe_net = np.mean(net_returns) / (np.std(net_returns) + 1e-8) * np.sqrt(252)

        # Volatility
        vol_annual = np.std(net_returns) * np.sqrt(252)

        # Max drawdown
        cumulative = np.cumsum(net_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_dd = np.min(drawdown)

        # Turnover
        avg_turnover = np.mean(turnovers)

        # Win rate
        win_rate = np.mean(net_returns > 0)

        # Total costs
        total_costs = pnl_df['cost'].sum()

        return {
            'sharpe_gross': sharpe_gross,
            'sharpe_net': sharpe_net,
            'annual_return_gross': annual_gross,
            'annual_return_net': annual_net,
            'volatility': vol_annual,
            'max_drawdown': max_dd,
            'avg_turnover': avg_turnover,
            'win_rate': win_rate,
            'total_costs': total_costs,
            'trading_days': len(pnl_df)
        }

    def _print_metrics(self, metrics: dict):
        """Print metrics."""
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("BACKTEST RESULTS")
        self.logger.info("="*70)
        self.logger.info(f"Trading days: {metrics['trading_days']}")
        self.logger.info("")
        self.logger.info("RETURNS (annualized):")
        self.logger.info(f"  Gross:   {metrics['annual_return_gross']*100:>6.2f}%")
        self.logger.info(f"  Net:     {metrics['annual_return_net']*100:>6.2f}%")
        self.logger.info("")
        self.logger.info("SHARPE RATIO:")
        self.logger.info(f"  Gross:   {metrics['sharpe_gross']:>6.2f}")
        self.logger.info(f"  Net:     {metrics['sharpe_net']:>6.2f}")
        self.logger.info("")
        self.logger.info("RISK:")
        self.logger.info(f"  Volatility (annual):   {metrics['volatility']*100:>6.2f}%")
        self.logger.info(f"  Max Drawdown (net):    {metrics['max_drawdown']*100:>6.2f}%")
        self.logger.info("")
        self.logger.info("TRADING:")
        self.logger.info(f"  Avg Turnover:       {metrics['avg_turnover']*100:>6.2f}%")
        self.logger.info(f"  Win Rate:           {metrics['win_rate']*100:>6.2f}%")
        self.logger.info(f"  Total Costs:        {metrics['total_costs']*100:>6.2f}%")
        self.logger.info("")


def get_logger(name):
    """Get logger (quick helper)."""
    import logging
    return logging.getLogger(name)


def main():
    """Main Phase 2B evaluation."""

    config = ConfigLoader('config/config.yaml')
    logger = setup_logger(
        name='phase2b',
        log_file='logs/phase2b_monetization.log',
        level='INFO'
    )

    logger.info("="*70)
    logger.info("PHASE 2B: PORTFOLIO MONETIZATION ENGINEERING")
    logger.info("="*70)
    logger.info("")
    logger.info("ChatGPT's Insight:")
    logger.info("  Temperature 0.05 -> Gross Sharpe 2.47, Net Sharpe 0.10")
    logger.info("  This is REAL alpha with monetization problem")
    logger.info("  Solution: Portfolio engineering, NOT more ML")
    logger.info("")
    logger.info("[FROZEN] Settings:")
    logger.info("  Model: Unchanged")
    logger.info("  Features: 14")
    logger.info("  Sequence: 90 days")
    logger.info("  Temperature: 0.05")
    logger.info("")
    logger.info("Phase 2B Techniques:")
    logger.info("  1. Rank-change trading filter")
    logger.info("  2. Position EWMA smoothing")
    logger.info("  3. Cross-sectional normalization")
    logger.info("="*70)
    logger.info("")

    # Load best model from Phase 2A (temperature 0.05)
    logger.info("Loading Phase 2A model (temperature 0.05)...")

    test_file = "data/processed/test_v2.parquet"
    test_df = pd.read_parquet(test_file)

    preprocessor = SimplifiedStockPreprocessor(config)
    preprocessor.feature_columns = preprocessor.get_feature_columns(test_df)

    sequence_length = 90

    X_test, y_test_reg, y_test_clf, test_metadata = preprocessor.create_sequences(
        test_df, sequence_length
    )

    logger.info(f"Test sequences: {X_test.shape}")

    # Load model
    input_size = X_test.shape[2]
    model_config = config.get('model.architecture')

    model = create_model(
        model_type="regression",
        input_size=input_size,
        config=model_config,
        num_classes=3
    )

    checkpoint_path = Path("models/checkpoints/lstm_phase2a_temp0.05_best.pth")

    if not checkpoint_path.exists():
        logger.error(f"Model not found: {checkpoint_path}")
        logger.error("Run phase2a_temperature_experiment.py first")
        return

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Model loaded from {checkpoint_path}")

    # Generate predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    X_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy().squeeze()

    results_df = test_metadata.copy()
    results_df['y_true_reg'] = y_test_reg
    results_df['y_pred_reg'] = predictions

    # Test different monetization configurations
    configs = [
        {
            'name': 'Baseline (No Engineering)',
            'rank_threshold': 0,
            'ewma_alpha': 1.0,
            'use_zscore': False
        },
        {
            'name': 'Rank Filter Only (K=10)',
            'rank_threshold': 10,
            'ewma_alpha': 1.0,
            'use_zscore': False
        },
        {
            'name': 'Position Smoothing Only',
            'rank_threshold': 0,
            'ewma_alpha': 0.15,
            'use_zscore': False
        },
        {
            'name': 'Cross-Sectional Z-Score Only',
            'rank_threshold': 0,
            'ewma_alpha': 1.0,
            'use_zscore': True
        },
        {
            'name': 'ALL TECHNIQUES (Phase 2B)',
            'rank_threshold': 10,
            'ewma_alpha': 0.15,
            'use_zscore': True
        }
    ]

    all_results = []

    for cfg in configs:
        logger.info("")
        logger.info("="*70)
        logger.info(f"TESTING: {cfg['name']}")
        logger.info("="*70)

        backtester = MonetizationBacktest(
            long_pct=0.2,
            short_pct=0.2,
            transaction_cost_bps=5.0,
            rank_change_threshold=cfg['rank_threshold'],
            position_ewma_alpha=cfg['ewma_alpha'],
            use_cross_sectional_zscore=cfg['use_zscore']
        )

        metrics = backtester.run_backtest(results_df.copy())
        metrics['config'] = cfg['name']

        all_results.append(metrics)

    # Summary comparison
    logger.info("")
    logger.info("="*70)
    logger.info("PHASE 2B: MONETIZATION COMPARISON")
    logger.info("="*70)
    logger.info("")
    logger.info(f"{'Configuration':<35} | {'Gross':<6} | {'Net':<6} | {'Turn':<6}")
    logger.info("-"*70)

    for result in all_results:
        name = result['config']
        gross = result['sharpe_gross']
        net = result['sharpe_net']
        turnover = result['avg_turnover'] * 100

        logger.info(f"{name:<35} | {gross:<6.2f} | {net:<6.2f} | {turnover:<5.0f}%")

    # Best result
    best = max(all_results, key=lambda x: x['sharpe_net'])

    logger.info("")
    logger.info("="*70)
    logger.info("BEST CONFIGURATION:")
    logger.info(f"  {best['config']}")
    logger.info(f"  Net Sharpe: {best['sharpe_net']:.2f}")
    logger.info(f"  Gross Sharpe: {best['sharpe_gross']:.2f}")
    logger.info(f"  Turnover: {best['avg_turnover']*100:.1f}%")
    logger.info("="*70)

    # Save results
    results_file = "data/processed/phase2b_monetization_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Results saved to: {results_file}")

    # Final verdict
    logger.info("")
    logger.info("="*70)
    logger.info("PHASE 2B VERDICT:")
    if best['sharpe_net'] >= 0.4:
        logger.info("[SUCCESS] SUCCESS: Strategy is TRADEABLE")
        logger.info(f"   Net Sharpe {best['sharpe_net']:.2f} >= 0.4")
        logger.info("   Ready for paper trading or live testing")
    elif best['sharpe_net'] > 0:
        logger.info("[OK] PROFITABLE but weak")
        logger.info(f"   Net Sharpe {best['sharpe_net']:.2f} > 0")
        logger.info("   Consider additional refinements")
    else:
        logger.info("[WARNING] Still unprofitable after portfolio engineering")
        logger.info("   May need different approach or data sources")
    logger.info("="*70)


if __name__ == "__main__":
    main()
