"""
PHASE 3.2b: Factor Exposure Analysis

Critical Question: "Is this just momentum/market beta in disguise?"

This script:
1. Downloads SPY daily returns
2. Regresses daily PnL vs SPY
3. Calculates beta, R², alpha
4. Verifies beta < 0.2
"""
import sys
from pathlib import Path
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from sklearn.linear_model import LinearRegression

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger


def download_spy_returns(start_date: str, end_date: str) -> pd.DataFrame:
    """Download SPY daily returns."""
    print(f"Downloading SPY data from {start_date} to {end_date}...")

    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)

    # Calculate daily returns
    spy['return'] = spy['Close'].pct_change()
    spy = spy.dropna()

    # Remove timezone info to match PnL dataframe
    spy_returns = pd.DataFrame({
        'date': pd.to_datetime(spy.index).tz_localize(None),
        'spy_return': spy['return'].values
    })

    print(f"Downloaded {len(spy_returns)} days of SPY returns")

    return spy_returns


def run_factor_regression(pnl_df: pd.DataFrame, spy_returns: pd.DataFrame, logger) -> dict:
    """Run factor regression analysis."""

    logger.info("\n" + "="*70)
    logger.info("PHASE 3.2b: FACTOR EXPOSURE ANALYSIS")
    logger.info("="*70)
    logger.info("")

    # Ensure dates are timezone-naive for both dataframes
    pnl_df = pnl_df.copy()
    pnl_df['date'] = pd.to_datetime(pnl_df['date']).dt.tz_localize(None)
    spy_returns['date'] = pd.to_datetime(spy_returns['date']).dt.tz_localize(None)

    # Merge PnL with SPY returns
    merged = pnl_df.merge(spy_returns, on='date', how='inner')

    logger.info(f"Matched {len(merged)} days with SPY data")
    logger.info("")

    # Prepare data for regression
    X = merged['spy_return'].values.reshape(-1, 1)
    y = merged['pnl_net'].values

    # Run regression
    model = LinearRegression()
    model.fit(X, y)

    beta = model.coef_[0]
    alpha_daily = model.intercept_

    # Calculate R²
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate correlation
    correlation = np.corrcoef(X.flatten(), y)[0, 1]

    # Annualize metrics
    alpha_annual = alpha_daily * 252

    # Calculate t-stat for beta
    residuals = y - y_pred
    residual_std = np.std(residuals)
    se_beta = residual_std / (np.std(X) * np.sqrt(len(X)))
    t_stat = beta / se_beta
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(X) - 2))

    # Results
    results = {
        'beta': float(beta),
        'alpha_daily': float(alpha_daily),
        'alpha_annual': float(alpha_annual),
        'r_squared': float(r_squared),
        'correlation': float(correlation),
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'n_observations': int(len(merged))
    }

    # Print results
    logger.info("MARKET FACTOR REGRESSION (vs SPY)")
    logger.info("-" * 70)
    logger.info("")
    logger.info("Regression: PnL = alpha + beta * SPY_return")
    logger.info("")
    logger.info(f"  Beta (Market Exposure):     {beta:.4f}")
    logger.info(f"  Alpha (Daily):              {alpha_daily:.6f} ({alpha_daily*10000:.2f} bps)")
    logger.info(f"  Alpha (Annual):             {alpha_annual:.4f} ({alpha_annual*100:.2f}%)")
    logger.info(f"  R-squared:                  {r_squared:.4f}")
    logger.info(f"  Correlation:                {correlation:.4f}")
    logger.info("")
    logger.info(f"  Beta t-statistic:           {t_stat:.2f}")
    logger.info(f"  Beta p-value:               {p_value:.4f}")
    logger.info(f"  Beta significant?           {'YES' if p_value < 0.05 else 'NO'}")
    logger.info("")
    logger.info(f"  Observations:               {len(merged)} days")
    logger.info("")

    # Interpretation
    logger.info("INTERPRETATION:")
    logger.info("-" * 70)

    if abs(beta) < 0.1:
        logger.info("  [EXCELLENT] Beta < 0.1 - Very low market exposure")
        logger.info("  Dollar neutrality is working as designed")
        flag = "EXCELLENT"
    elif abs(beta) < 0.2:
        logger.info("  [GOOD] Beta < 0.2 - Low market exposure")
        logger.info("  Acceptable for market-neutral strategy")
        flag = "GOOD"
    elif abs(beta) < 0.3:
        logger.info("  [ACCEPTABLE] Beta < 0.3 - Moderate market exposure")
        logger.info("  May need beta hedging before scale-up")
        flag = "ACCEPTABLE"
    else:
        logger.info("  [WARNING] Beta > 0.3 - Significant market exposure")
        logger.info("  Dollar neutrality not working - investigate")
        flag = "WARNING"

    logger.info("")

    if r_squared < 0.1:
        logger.info("  [GOOD] R² < 0.1 - Alpha not explained by market")
        logger.info("  This is genuine cross-sectional alpha")
    elif r_squared < 0.2:
        logger.info("  [ACCEPTABLE] R² < 0.2 - Mostly unexplained by market")
    else:
        logger.info("  [WARNING] R² > 0.2 - Significant market dependence")
        logger.info("  May be capturing market timing, not stock selection")

    results['flag'] = flag

    logger.info("")
    logger.info("="*70)

    return results


def analyze_factor_exposure(predictions_file: str = "data/processed/phase1_predictions.parquet"):
    """Main function to run factor exposure analysis."""

    logger = setup_logger('phase3_2b_factor', log_file='logs/phase3_2b_factor_exposure.log', level='INFO')

    logger.info("="*70)
    logger.info("PHASE 3.2b: FACTOR EXPOSURE ANALYSIS")
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
    logger.info(f"Loading predictions from {predictions_file}...")
    predictions_df = pd.read_parquet(predictions_file)
    logger.info(f"Loaded {len(predictions_df)} predictions")
    logger.info("")

    # Run backtest to get daily PnL (simplified version)
    from scripts.risk.phase3_risk_decomposition import RiskDecomposition
    from src.utils.config_loader import ConfigLoader

    config = ConfigLoader('config/config.yaml')
    risk_analyzer = RiskDecomposition(predictions_df, config)

    logger.info("Running backtest to calculate daily PnL...")
    risk_analyzer.run_backtest_with_tracking()
    pnl_df = risk_analyzer.pnl_df

    logger.info(f"Calculated PnL for {len(pnl_df)} trading days")
    logger.info("")

    # Get date range
    start_date = pnl_df['date'].min()
    end_date = pnl_df['date'].max()

    # Add buffer for SPY download
    start_date_str = (pd.to_datetime(start_date) - timedelta(days=5)).strftime('%Y-%m-%d')
    end_date_str = (pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d')

    # Download SPY returns
    spy_returns = download_spy_returns(start_date_str, end_date_str)
    logger.info("")

    # Run factor regression
    results = run_factor_regression(pnl_df, spy_returns, logger)

    # Save results
    results_file = "data/processed/phase3_2b_factor_exposure_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("")
    logger.info(f"Results saved to: {results_file}")
    logger.info("")

    # Final assessment
    logger.info("="*70)
    logger.info("FINAL ASSESSMENT")
    logger.info("="*70)
    logger.info("")

    if results['flag'] in ['EXCELLENT', 'GOOD']:
        logger.info("  [PASS] Factor exposure within acceptable limits")
        logger.info("  Strategy exhibits genuine cross-sectional alpha")
        logger.info("")
        logger.info("  Ready to proceed to Phase 3.3")
    elif results['flag'] == 'ACCEPTABLE':
        logger.info("  [CONDITIONAL PASS] Moderate market exposure detected")
        logger.info("  Consider beta hedging in Phase 3.4")
        logger.info("")
        logger.info("  Can proceed to Phase 3.3 with caution")
    else:
        logger.info("  [WARNING] Significant market exposure detected")
        logger.info("  Dollar neutrality may not be working correctly")
        logger.info("")
        logger.info("  Investigate before proceeding to Phase 3.3")

    logger.info("="*70)

    return results


if __name__ == "__main__":
    analyze_factor_exposure()
