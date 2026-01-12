"""
Data preprocessing and feature engineering for stock data.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import ta

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger


class StockPreprocessor:
    """Preprocess stock data and engineer features."""

    def __init__(self, config: ConfigLoader):
        """
        Initialize preprocessor.

        Args:
            config: Configuration loader instance
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Scalers for normalization (fit on training data only)
        self.feature_scaler: Optional[StandardScaler] = None
        self.target_scaler: Optional[StandardScaler] = None

        # Feature columns (will be set after feature engineering)
        self.feature_columns: List[str] = []

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for stock data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional technical indicator columns
        """
        self.logger.info("Calculating technical indicators...")

        # Work with a copy
        df = df.copy()

        # Calculate returns
        df['returns'] = df.groupby('ticker')['close'].pct_change()

        # Moving Averages
        df['SMA_10'] = df.groupby('ticker')['close'].transform(lambda x: ta.trend.sma_indicator(x, window=10))
        df['SMA_20'] = df.groupby('ticker')['close'].transform(lambda x: ta.trend.sma_indicator(x, window=20))
        df['SMA_50'] = df.groupby('ticker')['close'].transform(lambda x: ta.trend.sma_indicator(x, window=50))
        df['EMA_12'] = df.groupby('ticker')['close'].transform(lambda x: ta.trend.ema_indicator(x, window=12))
        df['EMA_26'] = df.groupby('ticker')['close'].transform(lambda x: ta.trend.ema_indicator(x, window=26))

        # MACD
        df['MACD'] = df.groupby('ticker')['close'].transform(lambda x: ta.trend.macd(x))
        df['MACD_signal'] = df.groupby('ticker')['close'].transform(lambda x: ta.trend.macd_signal(x))
        df['MACD_diff'] = df.groupby('ticker')['close'].transform(lambda x: ta.trend.macd_diff(x))

        # RSI
        df['RSI_14'] = df.groupby('ticker')['close'].transform(lambda x: ta.momentum.rsi(x, window=14))

        # Bollinger Bands
        bb_indicator = df.groupby('ticker')['close'].transform(lambda x: ta.volatility.BollingerBands(x, window=20))
        df['BB_upper'] = df.groupby('ticker')['close'].transform(lambda x: ta.volatility.bollinger_hband(x, window=20))
        df['BB_middle'] = df.groupby('ticker')['close'].transform(lambda x: ta.volatility.bollinger_mavg(x, window=20))
        df['BB_lower'] = df.groupby('ticker')['close'].transform(lambda x: ta.volatility.bollinger_lband(x, window=20))

        # ATR (Average True Range)
        df['ATR_14'] = df.groupby('ticker', group_keys=False).apply(
            lambda x: ta.volatility.average_true_range(x['high'], x['low'], x['close'], window=14),
            include_groups=False
        ).reset_index(level=0, drop=True)

        # Volume indicators
        df['Volume_SMA_20'] = df.groupby('ticker')['volume'].transform(lambda x: ta.trend.sma_indicator(x, window=20))
        df['OBV'] = df.groupby('ticker', group_keys=False).apply(
            lambda x: ta.volume.on_balance_volume(x['close'], x['volume']),
            include_groups=False
        ).reset_index(level=0, drop=True)

        # Price momentum
        df['momentum_5'] = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(periods=5))
        df['momentum_10'] = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(periods=10))

        # Volatility (rolling standard deviation of returns)
        df['volatility_10'] = df.groupby('ticker')['returns'].transform(lambda x: x.rolling(window=10).std())
        df['volatility_20'] = df.groupby('ticker')['returns'].transform(lambda x: x.rolling(window=20).std())

        # Time-based features
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 4=Friday
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)

        # Volume analysis
        df['volume_ratio'] = df['volume'] / (df['Volume_SMA_20'] + 1e-8)
        df['volume_change'] = df.groupby('ticker')['volume'].pct_change()

        # Price-volume relationship
        df['price_volume_trend'] = df['close'] * df['volume']

        # Calculate rolling price-volume correlation per ticker
        def calc_pv_corr(group):
            return group['close'].rolling(window=20).corr(group['volume'])

        df['price_volume_corr'] = df.groupby('ticker', group_keys=False).apply(
            calc_pv_corr, include_groups=False
        ).reset_index(level=0, drop=True)

        # Cross-stock features (market-wide indicators)
        # Calculate daily market average return
        daily_market_avg = df.groupby('date')['returns'].transform('mean')
        df['market_return'] = daily_market_avg

        # Relative strength vs market
        df['vs_market'] = df['returns'] - df['market_return']

        # Stock's correlation with market (rolling 20 days)
        def calc_market_corr(group):
            return group['returns'].rolling(window=20).corr(group['market_return'])

        df['market_correlation'] = df.groupby('ticker', group_keys=False).apply(
            calc_market_corr, include_groups=False
        ).reset_index(level=0, drop=True)

        # High-Low range
        df['high_low_range'] = (df['high'] - df['low']) / df['close']

        # Gap (open vs previous close)
        df['gap'] = df.groupby('ticker', group_keys=False).apply(
            lambda x: (x['open'] - x['close'].shift(1)) / x['close'].shift(1),
            include_groups=False
        ).reset_index(level=0, drop=True)

        # Replace infinite values with NaN (will be handled later)
        df = df.replace([np.inf, -np.inf], np.nan)

        self.logger.info(f"Calculated {len(df.columns)} features including technical indicators and market features")

        return df

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target labels for regression and classification.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with labels added
        """
        self.logger.info("Creating target labels...")

        df = df.copy()

        # Get prediction horizon from config
        prediction_horizon = self.config.get('data.prediction_horizon', 1)

        # Regression target: N-day ahead percentage return
        df['target_return'] = df.groupby('ticker')['close'].pct_change(periods=prediction_horizon).shift(-prediction_horizon)

        # Keep absolute price for reference/metadata only
        df['target_price'] = df.groupby('ticker')['close'].shift(-prediction_horizon)

        # Classification target based on thresholds
        threshold_up = self.config.get('features.classification.threshold_up', 0.005)
        threshold_down = self.config.get('features.classification.threshold_down', -0.005)

        # Binary classification: UP (1) vs DOWN (0)
        # Check if we should use binary or three-class classification
        # Binary if thresholds are very close (within 0.001 of each other)
        use_binary = abs(threshold_up - abs(threshold_down)) < 0.001

        if use_binary:
            # Binary: UP (1) if above threshold_up, DOWN (0) otherwise
            df['target_class'] = (df['target_return'] > threshold_up).astype(int)
            self.logger.info(f"Using binary classification: UP (1) if return > {threshold_up}, DOWN (0) otherwise")
        else:
            # Three-class: UP, DOWN, NEUTRAL
            df['target_class'] = 2  # Default: NEUTRAL
            df.loc[df['target_return'] > threshold_up, 'target_class'] = 1
            df.loc[df['target_return'] < threshold_down, 'target_class'] = 0
            self.logger.info(f"Using three-class classification: UP (1) if > {threshold_up}, DOWN (0) if < {threshold_down}, NEUTRAL (2) otherwise")

        # Log statistics
        class_counts = df['target_class'].value_counts().sort_index()
        self.logger.info(f"Target class distribution: {class_counts.to_dict()}")
        self.logger.info(f"Prediction horizon: {prediction_horizon} days")
        self.logger.info(f"Target return stats: mean={df['target_return'].mean():.6f}, std={df['target_return'].std():.6f}")

        return df

    def split_by_date(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by date ranges (temporal split).

        Args:
            df: Complete DataFrame

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_end = pd.to_datetime(self.config.get('data.date_ranges.train_end'))
        val_start = pd.to_datetime(self.config.get('data.date_ranges.validation_start'))
        val_end = pd.to_datetime(self.config.get('data.date_ranges.validation_end'))
        test_start = pd.to_datetime(self.config.get('data.date_ranges.test_start'))

        # Make dates timezone-aware if the dataframe has timezone-aware dates
        if df['date'].dt.tz is not None:
            tz = df['date'].dt.tz
            train_end = train_end.tz_localize(tz)
            val_start = val_start.tz_localize(tz)
            val_end = val_end.tz_localize(tz)
            test_start = test_start.tz_localize(tz)

        train_df = df[df['date'] <= train_end].copy()
        val_df = df[(df['date'] >= val_start) & (df['date'] <= val_end)].copy()
        test_df = df[df['date'] >= test_start].copy()

        self.logger.info(f"Train set: {len(train_df)} rows ({train_df['date'].min()} to {train_df['date'].max()})")
        self.logger.info(f"Validation set: {len(val_df)} rows ({val_df['date'].min()} to {val_df['date'].max()})")
        self.logger.info(f"Test set: {len(test_df)} rows ({test_df['date'].min()} to {test_df['date'].max()})")

        return train_df, val_df, test_df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (exclude metadata and targets).

        Args:
            df: DataFrame with all columns

        Returns:
            List of feature column names
        """
        # Columns to exclude
        exclude_cols = [
            'date', 'ticker', 'timestamp',
            'target_price', 'target_return', 'target_class'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols

    def normalize_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Normalize features using scaler fit on training data only.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame

        Returns:
            Tuple of normalized (train_df, val_df, test_df)
        """
        self.logger.info("Normalizing features...")

        # Get feature columns
        self.feature_columns = self.get_feature_columns(train_df)

        # Choose scaler
        scaler_method = self.config.get('features.normalization.method', 'standard')

        if scaler_method == 'standard':
            self.feature_scaler = StandardScaler()
        elif scaler_method == 'minmax':
            self.feature_scaler = MinMaxScaler()
        elif scaler_method == 'robust':
            self.feature_scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler method: {scaler_method}")

        # Fit scaler on training data only
        train_df[self.feature_columns] = self.feature_scaler.fit_transform(
            train_df[self.feature_columns]
        )

        # Transform validation and test data
        val_df[self.feature_columns] = self.feature_scaler.transform(
            val_df[self.feature_columns]
        )

        test_df[self.feature_columns] = self.feature_scaler.transform(
            test_df[self.feature_columns]
        )

        self.logger.info(f"Normalized {len(self.feature_columns)} features using {scaler_method} scaler")

        return train_df, val_df, test_df

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.

        Args:
            df: DataFrame with features and targets
            sequence_length: Number of time steps to look back

        Returns:
            Tuple of (X_seq, y_regression, y_classification, metadata)
        """
        self.logger.info(f"Creating sequences with length {sequence_length}...")

        X_list = []
        y_reg_list = []
        y_clf_list = []
        metadata_list = []

        # Group by ticker to avoid mixing different stocks
        for ticker, group in df.groupby('ticker'):
            group = group.sort_values('date').reset_index(drop=True)

            # Remove rows with NaN in features or targets
            group = group.dropna(subset=self.feature_columns + ['target_return', 'target_class'])

            if len(group) < sequence_length + 1:
                continue

            # Create sequences
            for i in range(len(group) - sequence_length):
                # Features: sequence_length days of historical data
                X_seq = group.iloc[i:i + sequence_length][self.feature_columns].values

                # Targets: next day after sequence
                y_reg = group.iloc[i + sequence_length]['target_return']  # Use percentage return
                y_clf = group.iloc[i + sequence_length]['target_class']

                # Metadata (keep current close price for reference)
                meta = {
                    'ticker': ticker,
                    'date': group.iloc[i + sequence_length]['date'],
                    'close': group.iloc[i + sequence_length]['close'],
                    'target_price': group.iloc[i + sequence_length]['target_price']
                }

                X_list.append(X_seq)
                y_reg_list.append(y_reg)
                y_clf_list.append(y_clf)
                metadata_list.append(meta)

        X = np.array(X_list)
        y_reg = np.array(y_reg_list)
        y_clf = np.array(y_clf_list)
        metadata = pd.DataFrame(metadata_list)

        self.logger.info(f"Created {len(X)} sequences")
        self.logger.info(f"X shape: {X.shape}, y_reg shape: {y_reg.shape}, y_clf shape: {y_clf.shape}")

        return X, y_reg, y_clf, metadata

    def preprocess_pipeline(
        self,
        input_file: str,
        output_dir: str
    ) -> Dict[str, str]:
        """
        Complete preprocessing pipeline.

        Args:
            input_file: Path to raw data file
            output_dir: Directory to save processed data

        Returns:
            Dictionary of output file paths
        """
        self.logger.info("Starting preprocessing pipeline...")

        # Load raw data
        self.logger.info(f"Loading raw data from {input_file}...")
        df = pd.read_parquet(input_file)

        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)

        # Create labels
        df = self.create_labels(df)

        # Split by date
        train_df, val_df, test_df = self.split_by_date(df)

        # Remove rows with NaN (from technical indicators)
        train_df = train_df.dropna()
        val_df = val_df.dropna()
        test_df = test_df.dropna()

        # Normalize features
        train_df, val_df, test_df = self.normalize_features(train_df, val_df, test_df)

        # Save processed data
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_file = output_path / "train.parquet"
        val_file = output_path / "validation.parquet"
        test_file = output_path / "test.parquet"

        train_df.to_parquet(train_file)
        val_df.to_parquet(val_file)
        test_df.to_parquet(test_file)

        self.logger.info(f"Saved training data to {train_file}")
        self.logger.info(f"Saved validation data to {val_file}")
        self.logger.info(f"Saved test data to {test_file}")

        return {
            'train': str(train_file),
            'validation': str(val_file),
            'test': str(test_file)
        }
