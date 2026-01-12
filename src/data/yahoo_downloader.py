"""
Stock data downloader from Yahoo Finance.
"""
import time
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger


class YahooDownloader:
    """Download stock data from Yahoo Finance."""

    def __init__(self, config: ConfigLoader):
        """
        Initialize Yahoo Finance downloader.

        Args:
            config: Configuration loader instance
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Get cache directory
        cache_dir = config.get('data.polygon.cache_dir', 'data/raw/cache')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download stock data for a single ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with OHLCV data
        """
        cache_file = self.cache_dir / f"{ticker}_{start_date}_{end_date}_day.parquet"

        # Check cache
        if use_cache and cache_file.exists():
            self.logger.info(f"Loading {ticker} from cache...")
            return pd.read_parquet(cache_file)

        self.logger.info(f"Downloading {ticker} from Yahoo Finance...")

        try:
            # Download data from Yahoo Finance
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval='1d')

            if df.empty:
                self.logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            # Reset index to get date as column
            df = df.reset_index()

            # Rename columns to match Polygon.io format
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Select only needed columns
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

            # Add ticker column
            df['ticker'] = ticker

            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])

            # Save to cache
            df.to_parquet(cache_file)
            self.logger.info(f"Cached {ticker} data to {cache_file}")

            return df

        except Exception as e:
            self.logger.error(f"Failed to download {ticker}: {e}")
            return pd.DataFrame()

    def download_multiple_stocks(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        output_file: str
    ) -> pd.DataFrame:
        """
        Download data for multiple stocks.

        Args:
            tickers: List of stock ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_file: Path to save combined data

        Returns:
            Combined DataFrame for all stocks
        """
        all_data = []

        self.logger.info(f"Downloading {len(tickers)} stocks...")

        for ticker in tqdm(tickers, desc="Downloading stocks"):
            df = self.download_stock_data(ticker, start_date, end_date)

            if not df.empty:
                all_data.append(df)
            else:
                self.logger.warning(f"No data for {ticker}")

            # Small delay to be respectful to Yahoo Finance
            time.sleep(0.1)

        if not all_data:
            self.logger.error("No data downloaded for any ticker")
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Sort by ticker and date
        combined_df = combined_df.sort_values(['ticker', 'date']).reset_index(drop=True)

        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_parquet(output_file)

        self.logger.info(f"Downloaded {len(combined_df)} total records")
        self.logger.info(f"Tickers: {sorted(combined_df['ticker'].unique().tolist())}")
        self.logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")

        return combined_df
