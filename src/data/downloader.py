"""
Stock data downloader from Polygon.io with rate limiting.
"""
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger


class PolygonDownloader:
    """Download stock data from Polygon.io API with rate limiting."""

    def __init__(self, config: ConfigLoader):
        """
        Initialize Polygon.io downloader.

        Args:
            config: Configuration loader instance
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Get API key from environment
        api_key_env = config.get('polygon.api_key_env', 'POLYGON_API_KEY')
        self.api_key = os.getenv(api_key_env)

        if not self.api_key:
            raise ValueError(
                f"Polygon.io API key not found. Please set {api_key_env} "
                "environment variable in .env file"
            )

        # Rate limiting settings
        self.calls_per_minute = config.get('polygon.rate_limit.calls_per_minute', 5)
        self.safety_margin = config.get('polygon.rate_limit.safety_margin', 0.8)
        self.effective_calls_per_minute = int(self.calls_per_minute * self.safety_margin)

        # Base URL
        self.base_url = config.get('polygon.base_url', 'https://api.polygon.io')

        # Cache directory
        self.cache_dir = Path(config.get('polygon.cache_dir', 'data/raw/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Track API calls for rate limiting
        self.call_times: List[float] = []

    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits."""
        current_time = time.time()

        # Remove calls older than 60 seconds
        self.call_times = [t for t in self.call_times if current_time - t < 60]

        # If we've hit the limit, wait
        if len(self.call_times) >= self.effective_calls_per_minute:
            oldest_call = self.call_times[0]
            wait_time = 60 - (current_time - oldest_call)

            if wait_time > 0:
                self.logger.debug(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

                # Clean up old calls again
                current_time = time.time()
                self.call_times = [t for t in self.call_times if current_time - t < 60]

    def _make_request(self, url: str, params: Optional[dict] = None) -> dict:
        """
        Make API request with rate limiting.

        Args:
            url: API endpoint URL
            params: Query parameters

        Returns:
            JSON response data
        """
        self._wait_for_rate_limit()

        if params is None:
            params = {}

        params['apiKey'] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=30)
            self.call_times.append(time.time())

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise

    def download_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        timespan: str = "day",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download stock data for a single ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timespan: Time span (day, hour, minute)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with OHLCV data
        """
        cache_file = self.cache_dir / f"{ticker}_{start_date}_{end_date}_{timespan}.parquet"

        # Check cache
        if use_cache and cache_file.exists():
            self.logger.info(f"Loading {ticker} from cache...")
            return pd.read_parquet(cache_file)

        self.logger.info(f"Downloading {ticker} from Polygon.io...")

        # Construct API URL
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start_date}/{end_date}"

        try:
            data = self._make_request(url, params={"adjusted": "true", "sort": "asc", "limit": 50000})

            if data.get('status') != 'OK' or not data.get('results'):
                self.logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(data['results'])

            # Rename columns to standard names
            df = df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                't': 'timestamp',
                'n': 'num_trades'
            })

            # Convert timestamp to datetime
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['ticker'] = ticker

            # Reorder columns
            columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
            if 'num_trades' in df.columns:
                columns.append('num_trades')

            df = df[columns]

            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)

            # Save to cache
            df.to_parquet(cache_file)
            self.logger.info(f"Downloaded {len(df)} rows for {ticker}")

            return df

        except Exception as e:
            self.logger.error(f"Failed to download {ticker}: {e}")
            return pd.DataFrame()

    def download_multiple_stocks(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        output_file: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download data for multiple stocks with progress bar.

        Args:
            tickers: List of stock ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_file: Path to save combined data (optional)
            use_cache: Whether to use cached data

        Returns:
            Combined DataFrame with all stocks
        """
        all_data = []

        self.logger.info(f"Downloading {len(tickers)} stocks...")
        self.logger.info(f"Date range: {start_date} to {end_date}")
        self.logger.info(f"Rate limit: {self.effective_calls_per_minute} calls/minute")

        with tqdm(total=len(tickers), desc="Downloading stocks") as pbar:
            for ticker in tickers:
                try:
                    df = self.download_stock_data(ticker, start_date, end_date, use_cache=use_cache)

                    if not df.empty:
                        all_data.append(df)
                    else:
                        self.logger.warning(f"No data for {ticker}")

                except Exception as e:
                    self.logger.error(f"Error downloading {ticker}: {e}")

                pbar.update(1)

        if not all_data:
            self.logger.error("No data downloaded for any ticker")
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['ticker', 'date']).reset_index(drop=True)

        self.logger.info(f"Total rows downloaded: {len(combined_df)}")
        self.logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")

        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            combined_df.to_parquet(output_path)
            self.logger.info(f"Saved data to {output_file}")

        return combined_df

    def get_ticker_details(self, ticker: str) -> dict:
        """
        Get ticker details (company info, market cap, etc.).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Ticker details dictionary
        """
        url = f"{self.base_url}/v3/reference/tickers/{ticker}"

        try:
            data = self._make_request(url)
            return data.get('results', {})

        except Exception as e:
            self.logger.error(f"Failed to get details for {ticker}: {e}")
            return {}
