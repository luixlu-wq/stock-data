"""
Phase 4: Daily Data Pipeline

This script downloads fresh market data, generates predictions, and runs paper trading.
Should be executed once per day after market close (4:15 PM EST).

Workflow:
1. Download today's OHLCV data for all tickers
2. Apply feature engineering (14 core features)
3. Run model inference to generate predictions
4. Execute paper trading with today's predictions
5. Log results

Usage:
    python scripts/paper_trading/phase4_daily_pipeline.py

    # Or for a specific date (backtesting mode):
    python scripts/paper_trading/phase4_daily_pipeline.py --date 2025-01-19
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocessor_v2 import SimplifiedStockPreprocessor
from src.models.lstm_model import LSTMModel

# Setup logging
log_dir = Path("logs/paper_trading")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"daily_pipeline_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants from STRATEGY_DEFINITION.md v2.0.0
MODEL_PATH = "models/checkpoints/lstm_phase2a_temp0.05_best.pth"
SEQUENCE_LENGTH = 90
LOOKBACK_DAYS = 120  # Need extra for feature calculation
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DailyPipeline:
    """
    Daily data pipeline for paper trading.
    """

    def __init__(self, model_path: str, tickers_file: str = None):
        self.model_path = model_path

        # Load ticker universe
        if tickers_file:
            with open(tickers_file, 'r') as f:
                self.tickers = [line.strip() for line in f if line.strip()]
        else:
            # Default: Load from historical predictions
            logger.info("Loading ticker universe from phase1_predictions.parquet")
            df = pd.read_parquet("data/processed/phase1_predictions.parquet")
            self.tickers = sorted(df['ticker'].unique().tolist())

        logger.info(f"Ticker universe: {len(self.tickers)} stocks")

        # Load model
        self.model = self.load_model()

        # Feature engineer (preprocessor)
        self.preprocessor = SimplifiedStockPreprocessor()

    def load_model(self) -> LSTMModel:
        """Load the frozen production model."""
        logger.info(f"Loading model from {self.model_path}")

        # Model configuration (from STRATEGY_DEFINITION.md v2.0.0)
        model = LSTMModel(
            input_size=14,  # 14 core features
            hidden_size=128,
            num_layers=2,
            output_size=1,
            dropout=0.2
        )

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()

        logger.info(f"Model loaded successfully (device: {DEVICE})")
        return model

    def download_data(self, end_date: str) -> pd.DataFrame:
        """
        Download historical OHLCV data for all tickers.

        Args:
            end_date: End date (inclusive) in YYYY-MM-DD format

        Returns:
            DataFrame with OHLCV data for all tickers
        """
        # Calculate start date (need LOOKBACK_DAYS + some buffer)
        end_dt = pd.to_datetime(end_date)
        start_dt = end_dt - timedelta(days=LOOKBACK_DAYS + 30)  # Extra buffer for weekends/holidays
        start_date = start_dt.strftime('%Y-%m-%d')

        logger.info(f"Downloading data from {start_date} to {end_date}")
        logger.info(f"Tickers: {len(self.tickers)}")

        all_data = []
        failed_tickers = []

        for i, ticker in enumerate(self.tickers):
            try:
                if (i + 1) % 20 == 0:
                    logger.info(f"Progress: {i+1}/{len(self.tickers)} tickers")

                df = yf.download(ticker, start=start_date, end=end_date, progress=False)

                if df.empty:
                    logger.warning(f"No data for {ticker}")
                    failed_tickers.append(ticker)
                    continue

                df = df.reset_index()
                df['ticker'] = ticker
                df.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'ticker']

                all_data.append(df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']])

            except Exception as e:
                logger.error(f"Error downloading {ticker}: {e}")
                failed_tickers.append(ticker)

        if failed_tickers:
            logger.warning(f"Failed to download {len(failed_tickers)} tickers: {failed_tickers[:10]}...")

        if not all_data:
            raise ValueError("No data downloaded for any ticker!")

        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Downloaded {len(combined_df)} rows for {combined_df['ticker'].nunique()} tickers")

        return combined_df

    def generate_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to raw OHLCV data.

        Returns:
            DataFrame with 14 core features
        """
        logger.info("Generating features (14 core features)")

        # Use the preprocessor to calculate features
        features_df = self.preprocessor.calculate_core_features(raw_data)

        if features_df is None or features_df.empty:
            raise ValueError("Feature engineering failed!")

        logger.info(f"Generated features: {len(features_df)} rows")

        return features_df

    def generate_predictions(self, features_df: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """
        Run model inference to generate predictions for target_date.

        Args:
            features_df: DataFrame with features for all tickers
            target_date: Date to generate predictions for (YYYY-MM-DD)

        Returns:
            DataFrame with predictions for target_date
        """
        logger.info(f"Generating predictions for {target_date}")

        target_dt = pd.to_datetime(target_date)
        predictions = []

        for ticker in features_df['ticker'].unique():
            ticker_data = features_df[features_df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date')

            # Get sequence ending at target_date
            ticker_data['date'] = pd.to_datetime(ticker_data['date'])
            sequence_data = ticker_data[ticker_data['date'] <= target_dt].tail(SEQUENCE_LENGTH)

            if len(sequence_data) < SEQUENCE_LENGTH:
                logger.debug(f"Skipping {ticker}: insufficient history ({len(sequence_data)} < {SEQUENCE_LENGTH})")
                continue

            # Prepare features (14 core features)
            feature_cols = [
                'return_1d', 'return_5d', 'return_20d',
                'volatility_20d', 'volatility_5d', 'volume_volatility',
                'dist_from_high', 'dist_from_low', 'price_range_compression',
                'momentum_20d', 'momentum_5d',
                'volume_ratio', 'volume_trend',
                'market_return'
            ]

            X = sequence_data[feature_cols].values

            # Check for NaN
            if np.isnan(X).any():
                logger.debug(f"Skipping {ticker}: NaN in features")
                continue

            # Convert to tensor
            X_tensor = torch.FloatTensor(X).unsqueeze(0).to(DEVICE)  # (1, seq_len, features)

            # Run inference
            with torch.no_grad():
                y_pred = self.model(X_tensor)
                y_pred = y_pred.cpu().numpy()[0, 0]

            # Get actual return (if available)
            target_row = ticker_data[ticker_data['date'] == target_dt]
            if not target_row.empty:
                y_actual = target_row['return_1d'].values[0]
                close_price = target_row['close'].values[0] if 'close' in target_row.columns else np.nan
            else:
                y_actual = np.nan
                close_price = np.nan

            predictions.append({
                'ticker': ticker,
                'date': target_date,
                'y_pred_reg': y_pred,
                'y_true_reg': y_actual,
                'close': close_price
            })

        predictions_df = pd.DataFrame(predictions)
        logger.info(f"Generated {len(predictions_df)} predictions for {target_date}")

        return predictions_df

    def run_paper_trading(self, predictions_df: pd.DataFrame, target_date: str):
        """
        Run paper trading for the given predictions.

        This saves the predictions and calls the paper trading runner.
        """
        logger.info(f"Running paper trading for {target_date}")

        # Save predictions to temporary file
        temp_file = Path("data/paper_trading/daily") / f"predictions_{target_date}.parquet"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_parquet(temp_file)

        # Append to ongoing paper trading results
        # (The paper trading runner expects cumulative predictions)
        cumulative_file = Path("data/paper_trading") / "cumulative_predictions.parquet"

        if cumulative_file.exists():
            existing_df = pd.read_parquet(cumulative_file)
            # Remove existing data for this date (if re-running)
            existing_df = existing_df[existing_df['date'] != target_date]
            combined_df = pd.concat([existing_df, predictions_df], ignore_index=True)
        else:
            combined_df = predictions_df

        combined_df.to_parquet(cumulative_file)
        logger.info(f"Saved predictions to {cumulative_file}")

        # Now run the paper trading runner with the cumulative file
        from scripts.paper_trading.phase4_paper_trading_runner import PaperTradingRunner

        runner = PaperTradingRunner(
            predictions_path=str(cumulative_file),
            output_dir="data/processed/phase4"
        )

        # Run for just this date
        result = runner.run_single_day(target_date)

        if result:
            logger.info(f"Paper trading complete for {target_date}")
            logger.info(f"  PnL Net: {result['pnl_net']:.4f}")
            logger.info(f"  PnL Scaled: {result['pnl_scaled']:.4f}")
            logger.info(f"  Turnover: {result['turnover']:.2%}")
            logger.info(f"  Kill Switches: {result['ks_triggered']}")

            # Save daily result summary
            daily_summary_file = Path("reports/paper_trading/daily") / f"summary_{target_date}.json"
            daily_summary_file.parent.mkdir(parents=True, exist_ok=True)

            with open(daily_summary_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        else:
            logger.error(f"Paper trading failed for {target_date}")

    def run_daily(self, target_date: str = None):
        """
        Run the complete daily pipeline.

        Args:
            target_date: Date to run for (YYYY-MM-DD). If None, uses today.
        """
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"="*60)
        logger.info(f"DAILY PIPELINE: {target_date}")
        logger.info(f"="*60)

        try:
            # Step 1: Download data
            raw_data = self.download_data(target_date)

            # Step 2: Generate features
            features_df = self.generate_features(raw_data)

            # Step 3: Generate predictions
            predictions_df = self.generate_predictions(features_df, target_date)

            # Step 4: Run paper trading
            self.run_paper_trading(predictions_df, target_date)

            logger.info(f"="*60)
            logger.info(f"DAILY PIPELINE COMPLETE: {target_date}")
            logger.info(f"="*60)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    parser = argparse.ArgumentParser(description='Phase 4 Daily Pipeline')
    parser.add_argument('--date', type=str, default=None,
                       help='Target date (YYYY-MM-DD). If not provided, uses today.')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                       help='Path to model checkpoint')
    parser.add_argument('--tickers', type=str, default=None,
                       help='Path to tickers file (one per line)')

    args = parser.parse_args()

    # Create pipeline
    pipeline = DailyPipeline(
        model_path=args.model,
        tickers_file=args.tickers
    )

    # Run daily
    pipeline.run_daily(target_date=args.date)


if __name__ == "__main__":
    main()
