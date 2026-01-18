"""
PHASE 3.3 SHARPE AUDIT

Diagnose why P0 (Dollar Neutral) shows Sharpe 0.42 instead of expected ~1.42

Compare Phase 3.2 risk decomposition vs Phase 3.3 portfolio comparison
to identify mathematical discrepancies.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

TRADING_DAYS = 252


def run_phase32_style(predictions: pd.DataFrame, logger) -> dict:
    """
    Replicate Phase 3.2 risk decomposition logic (EWMA + costs).

    This is the reference implementation that showed:
    - Long Sharpe: 1.28
    - Short Sharpe: -0.45
    - Combined Sharpe: 1.42
    """
    logger.info("Running Phase 3.2 style (risk decomposition)...")

    K = 38
    tc = 0.0005
    ewma_alpha = 0.15

    long_pnl = []
    short_pnl = []
    total_pnl = []

    prev_long = defaultdict(float)
    prev_short = defaultdict(float)

    dates = sorted(predictions['date'].unique())

    for date in dates:
        day = predictions[predictions['date'] == date].copy()

        if len(day) < 5:
            continue

        day = day.sort_values('y_pred_reg', ascending=False)

        # Select top/bottom K
        longs = day.head(K)
        shorts = day.tail(K)

        # Target positions (equal weight)
        target_long = {}
        target_short = {}

        for _, row in longs.iterrows():
            target_long[row['ticker']] = 1.0 / K

        for _, row in shorts.iterrows():
            target_short[row['ticker']] = -1.0 / K

        # EWMA smoothing
        smoothed_long = {}
        smoothed_short = {}

        # Long smoothing
        all_long_tickers = set(target_long.keys()) | set(prev_long.keys())
        for ticker in all_long_tickers:
            target = target_long.get(ticker, 0.0)
            previous = prev_long.get(ticker, 0.0)
            smoothed = ewma_alpha * target + (1 - ewma_alpha) * previous
            if abs(smoothed) > 0.001:
                smoothed_long[ticker] = smoothed

        # Short smoothing
        all_short_tickers = set(target_short.keys()) | set(prev_short.keys())
        for ticker in all_short_tickers:
            target = target_short.get(ticker, 0.0)
            previous = prev_short.get(ticker, 0.0)
            smoothed = ewma_alpha * target + (1 - ewma_alpha) * previous
            if abs(smoothed) > 0.001:
                smoothed_short[ticker] = smoothed

        # Calculate turnover and costs
        long_traded = sum(abs(smoothed_long.get(t, 0) - prev_long.get(t, 0))
                         for t in all_long_tickers)
        short_traded = sum(abs(smoothed_short.get(t, 0) - prev_short.get(t, 0))
                          for t in all_short_tickers)

        long_cost = long_traded * tc
        short_cost = short_traded * tc

        # Calculate PnL
        returns_map = day.set_index('ticker')['y_true_reg'].to_dict()

        long_ret = sum(smoothed_long.get(t, 0) * returns_map.get(t, 0)
                      for t in smoothed_long.keys())
        short_ret = sum(smoothed_short.get(t, 0) * returns_map.get(t, 0)
                       for t in smoothed_short.keys())

        long_ret -= long_cost
        short_ret -= short_cost

        long_pnl.append(long_ret)
        short_pnl.append(short_ret)
        total_pnl.append(long_ret + short_ret)

        prev_long = smoothed_long
        prev_short = smoothed_short

    # Calculate metrics
    long_series = pd.Series(long_pnl)
    short_series = pd.Series(short_pnl)
    total_series = pd.Series(total_pnl)

    long_sharpe = long_series.mean() / long_series.std() * np.sqrt(TRADING_DAYS) if long_series.std() > 0 else 0
    short_sharpe = short_series.mean() / short_series.std() * np.sqrt(TRADING_DAYS) if short_series.std() > 0 else 0
    total_sharpe = total_series.mean() / total_series.std() * np.sqrt(TRADING_DAYS) if total_series.std() > 0 else 0

    logger.info(f"  Long Sharpe: {long_sharpe:.2f}")
    logger.info(f"  Short Sharpe: {short_sharpe:.2f}")
    logger.info(f"  Total Sharpe: {total_sharpe:.2f}")
    logger.info(f"  Avg daily PnL: {total_series.mean():.6f}")
    logger.info(f"  Daily vol: {total_series.std():.6f}")
    logger.info("")

    return {
        'long_sharpe': long_sharpe,
        'short_sharpe': short_sharpe,
        'total_sharpe': total_sharpe,
        'avg_pnl': total_series.mean(),
        'daily_vol': total_series.std(),
        'pnl_series': total_pnl
    }


def run_phase33_style(predictions: pd.DataFrame, logger) -> dict:
    """
    Replicate Phase 3.3 P0 logic (dollar neutral with gross=2.0).
    """
    logger.info("Running Phase 3.3 style (P0 dollar neutral)...")

    K = 38
    tc = 0.0005
    ewma_alpha = 0.15
    long_weight = 1.0
    short_weight = 1.0
    gross_target = long_weight + short_weight

    pnl_series = []
    turnover_series = []

    prev_positions = defaultdict(float)

    dates = sorted(predictions['date'].unique())

    for date in dates:
        day = predictions[predictions['date'] == date].copy()

        if len(day) < 5:
            continue

        day = day.sort_values('y_pred_reg', ascending=False)

        longs = day.head(K)
        shorts = day.tail(K)

        # Target positions
        target_positions = {}

        for _, row in longs.iterrows():
            target_positions[row['ticker']] = long_weight / K

        for _, row in shorts.iterrows():
            target_positions[row['ticker']] = -short_weight / K

        # EWMA smoothing
        if len(prev_positions) > 0:
            smoothed_positions = {}
            all_tickers = set(target_positions.keys()) | set(prev_positions.keys())

            for ticker in all_tickers:
                target = target_positions.get(ticker, 0.0)
                previous = prev_positions.get(ticker, 0.0)
                smoothed = ewma_alpha * target + (1 - ewma_alpha) * previous

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
        cost = traded * tc

        # PnL
        pnl = 0.0
        returns_map = day.set_index('ticker')['y_true_reg'].to_dict()

        for ticker, pos in positions.items():
            if ticker in returns_map:
                pnl += pos * returns_map[ticker]

        pnl -= cost

        pnl_series.append(pnl)
        turnover_series.append(turnover)

        prev_positions = positions

    # Calculate metrics
    pnl = pd.Series(pnl_series)
    sharpe = pnl.mean() / pnl.std() * np.sqrt(TRADING_DAYS) if pnl.std() > 0 else 0.0

    logger.info(f"  Sharpe: {sharpe:.2f}")
    logger.info(f"  Avg daily PnL: {pnl.mean():.6f}")
    logger.info(f"  Daily vol: {pnl.std():.6f}")
    logger.info(f"  Avg turnover: {np.mean(turnover_series):.2%}")
    logger.info("")

    return {
        'sharpe': sharpe,
        'avg_pnl': pnl.mean(),
        'daily_vol': pnl.std(),
        'avg_turnover': np.mean(turnover_series),
        'pnl_series': pnl_series
    }


def diagnose_discrepancy(phase32_results: dict, phase33_results: dict, logger):
    """Compare results and identify discrepancy."""
    logger.info("="*70)
    logger.info("DIAGNOSTIC COMPARISON")
    logger.info("="*70)
    logger.info("")

    logger.info("SHARPE COMPARISON:")
    logger.info(f"  Phase 3.2 (expected): {phase32_results['total_sharpe']:.2f}")
    logger.info(f"  Phase 3.3 (actual):   {phase33_results['sharpe']:.2f}")
    logger.info(f"  DELTA:                {phase33_results['sharpe'] - phase32_results['total_sharpe']:.2f}")
    logger.info("")

    logger.info("DAILY PNL COMPARISON:")
    logger.info(f"  Phase 3.2: {phase32_results['avg_pnl']:.6f}")
    logger.info(f"  Phase 3.3: {phase33_results['avg_pnl']:.6f}")
    logger.info(f"  Ratio:     {phase33_results['avg_pnl'] / phase32_results['avg_pnl']:.2f}x")
    logger.info("")

    logger.info("VOLATILITY COMPARISON:")
    logger.info(f"  Phase 3.2: {phase32_results['daily_vol']:.6f}")
    logger.info(f"  Phase 3.3: {phase33_results['daily_vol']:.6f}")
    logger.info(f"  Ratio:     {phase33_results['daily_vol'] / phase32_results['daily_vol']:.2f}x")
    logger.info("")

    # Check if PnL series match
    pnl32 = pd.Series(phase32_results['pnl_series'])
    pnl33 = pd.Series(phase33_results['pnl_series'])

    if len(pnl32) == len(pnl33):
        correlation = pnl32.corr(pnl33)
        logger.info(f"PNL SERIES CORRELATION: {correlation:.4f}")

        if correlation > 0.99:
            logger.info("  -> PnL series nearly identical")
        elif correlation > 0.90:
            logger.info("  -> PnL series highly correlated but different scale")
        else:
            logger.info("  -> PnL series DIVERGENT - different logic")
        logger.info("")

    # Hypothesis testing
    logger.info("HYPOTHESIS:")
    sharpe_ratio = phase33_results['sharpe'] / phase32_results['total_sharpe']

    if 0.95 <= sharpe_ratio <= 1.05:
        logger.info("  ✅ Results match - no discrepancy")
    elif 0.45 <= sharpe_ratio <= 0.55:
        logger.info("  ⚠️  Phase 3.3 shows ~50% of Phase 3.2 Sharpe")
        logger.info("  LIKELY CAUSE: Turnover normalization by gross=2.0 doubles costs")
        logger.info("  OR: Position sizing scaled by 0.5 somewhere")
    else:
        logger.info(f"  🚨 Unexpected Sharpe ratio: {sharpe_ratio:.2f}x")
        logger.info("  REQUIRES INVESTIGATION")

    logger.info("")


def main():
    logger = setup_logger('phase3_3_sharpe_audit', log_file='logs/phase3_3_sharpe_audit.log', level='INFO')

    logger.info("="*70)
    logger.info("PHASE 3.3 SHARPE AUDIT")
    logger.info("="*70)
    logger.info("")
    logger.info("Objective: Diagnose P0 Sharpe discrepancy")
    logger.info("  Expected (Phase 3.2): ~1.42")
    logger.info("  Actual (Phase 3.3):   0.42")
    logger.info("")

    # Load predictions
    preds_file = Path("data/processed/phase1_predictions.parquet")
    logger.info(f"Loading predictions from {preds_file}...")
    preds = pd.read_parquet(preds_file)
    logger.info(f"Loaded {len(preds)} predictions")
    logger.info("")

    # Run both implementations
    phase32_results = run_phase32_style(preds, logger)
    phase33_results = run_phase33_style(preds, logger)

    # Diagnose
    diagnose_discrepancy(phase32_results, phase33_results, logger)

    logger.info("="*70)
    logger.info("AUDIT COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
