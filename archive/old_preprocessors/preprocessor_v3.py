"""
PHASE 2: Enhanced Feature Engineering

Builds on Phase 1's 14 core features by adding:
- Momentum indicators (3 features)
- Advanced volatility metrics (2 features)
- Microstructure features (2 features)

Total: 21 features (was 14 in Phase 1)

Key improvements:
- Capture momentum effects (trend continuation)
- Better volatility characterization (skewness, regime)
- Price microstructure (bid-ask proxy, volume momentum)
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger


class EnhancedStockPreprocessor:
    """Enhanced preprocessor with momentum and advanced volatility features."""

    def __init__(self, config: ConfigLoader):
        """
        Initialize preprocessor.

        Args:
            config: Configuration loader instance
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Scalers (fit on training data only)
        self.feature_scaler: Optional[RobustScaler] = None

        # Feature columns (will be set after feature engineering)
        self.feature_columns: List[str] = []

    def calculate_core_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Phase 1's 14 core features + Phase 2's 7 new features = 21 total.

        Phase 1 (14 features):
        1-3: Returns (1d, 5d, 20d)
        4-5: Volatility (10d, 20d)
        6-7: Price structure (hl_range, oc_gap)
        8-9: Trend (SMA distance 10d, 20d)
        10-11: Volume (log_volume, volume_change)
        12-14: Market context (market_return, vs_market, market_correlation)

        Phase 2 NEW (7 features):
        15-17: MOMENTUM (mom_5d, mom_20d, mom_60d) - trend continuation
        18-19: ADVANCED VOLATILITY (vol_skew, vol_regime) - distribution & regime
        20-21: MICROSTRUCTURE (amihud_illiq, volume_momentum) - trading dynamics

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all 21 features
        """
        self.logger.info("Calculating 21 enhanced features...")

        df = df.copy()

        # Ensure sorted by date within each ticker
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

        # ===== PHASE 1: CORE FEATURES (14) =====

        # Returns (3 features)
        df['ret_1d'] = df.groupby('ticker')['close'].transform(
            lambda x: np.log(x / x.shift(1))
        )
        df['ret_5d'] = df.groupby('ticker')['close'].transform(
            lambda x: np.log(x / x.shift(5))
        )
        df['ret_20d'] = df.groupby('ticker')['close'].transform(
            lambda x: np.log(x / x.shift(20))
        )

        # Volatility (2 features)
        df['vol_10d'] = df.groupby('ticker')['ret_1d'].transform(
            lambda x: x.rolling(window=10, min_periods=5).std()
        )
        df['vol_20d'] = df.groupby('ticker')['ret_1d'].transform(
            lambda x: x.rolling(window=20, min_periods=10).std()
        )

        # Price structure (2 features)
        df['hl_range'] = (df['high'] - df['low']) / df['close']

        df['oc_gap'] = df.groupby('ticker').apply(
            lambda x: (x['open'] - x['close'].shift(1)) / x['close'].shift(1),
            include_groups=False
        ).reset_index(level=0, drop=True)

        # Trend (2 features)
        sma_10 = df.groupby('ticker')['close'].transform(
            lambda x: x.rolling(window=10, min_periods=5).mean()
        )
        sma_20 = df.groupby('ticker')['close'].transform(
            lambda x: x.rolling(window=20, min_periods=10).mean()
        )

        df['sma_10_dist'] = (df['close'] - sma_10) / sma_10
        df['sma_20_dist'] = (df['close'] - sma_20) / sma_20

        # Volume (2 features)
        df['log_volume'] = np.log(df['volume'] + 1)

        df['volume_change'] = df.groupby('ticker')['volume'].transform(
            lambda x: np.log((x + 1) / (x.shift(1) + 1))
        )

        # Cross-sectional features (3 features)
        df['market_return'] = df.groupby('date')['ret_1d'].transform('mean')
        df['vs_market'] = df['ret_1d'] - df['market_return']

        def calc_market_corr(group):
            return group['ret_1d'].rolling(window=20, min_periods=10).corr(
                group['market_return']
            )

        df['market_correlation'] = df.groupby('ticker', group_keys=False).apply(
            calc_market_corr, include_groups=False
        ).reset_index(level=0, drop=True)

        # ===== PHASE 2: NEW FEATURES (7) =====

        # MOMENTUM (3 features) - Trend continuation signals
        # Momentum = cumulative return over window (different from simple return)
        # Captures trend strength and persistence

        df['mom_5d'] = df.groupby('ticker')['ret_1d'].transform(
            lambda x: x.rolling(window=5, min_periods=3).sum()  # 5-day momentum
        )

        df['mom_20d'] = df.groupby('ticker')['ret_1d'].transform(
            lambda x: x.rolling(window=20, min_periods=10).sum()  # 20-day momentum
        )

        df['mom_60d'] = df.groupby('ticker')['ret_1d'].transform(
            lambda x: x.rolling(window=60, min_periods=30).sum()  # 60-day momentum (long-term)
        )

        # ADVANCED VOLATILITY (2 features) - Distribution characteristics

        # 1. Volatility skewness (asymmetry in return distribution)
        # Negative skew = more frequent large negative returns (crash risk)
        def calc_vol_skew(x):
            if len(x.dropna()) < 10:
                return np.nan
            return stats.skew(x.dropna())

        df['vol_skew'] = df.groupby('ticker')['ret_1d'].transform(
            lambda x: x.rolling(window=20, min_periods=10).apply(calc_vol_skew, raw=False)
        )

        # 2. Volatility regime (is current vol high or low relative to recent history?)
        # Ratio of short-term to long-term volatility
        df['vol_regime'] = df['vol_10d'] / (df['vol_20d'] + 1e-8)

        # MICROSTRUCTURE (2 features) - Trading dynamics

        # 1. Amihud illiquidity measure (price impact per dollar traded)
        # Higher = less liquid, more price impact from trades
        df['amihud_illiq'] = np.abs(df['ret_1d']) / (df['volume'] * df['close'] + 1e-8)

        # Take log to handle extreme values
        df['amihud_illiq'] = np.log(df['amihud_illiq'] + 1e-10)

        # 2. Volume momentum (is volume increasing or decreasing?)
        # Captures accumulation/distribution patterns
        df['volume_momentum'] = df.groupby('ticker')['volume_change'].transform(
            lambda x: x.rolling(window=5, min_periods=3).mean()
        )

        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        self.logger.info("Enhanced features calculated:")
        self.logger.info("  PHASE 1 (14 features):")
        self.logger.info("    Returns: ret_1d, ret_5d, ret_20d")
        self.logger.info("    Volatility: vol_10d, vol_20d")
        self.logger.info("    Price: hl_range, oc_gap")
        self.logger.info("    Trend: sma_10_dist, sma_20_dist")
        self.logger.info("    Volume: log_volume, volume_change")
        self.logger.info("    Market: market_return, vs_market, market_correlation")
        self.logger.info("  PHASE 2 NEW (7 features):")
        self.logger.info("    Momentum: mom_5d, mom_20d, mom_60d")
        self.logger.info("    Advanced Vol: vol_skew, vol_regime")
        self.logger.info("    Microstructure: amihud_illiq, volume_momentum")
        self.logger.info("  Total: 21 features")

        return df

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target labels (same as Phase 1).

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with labels added
        """
        self.logger.info("Creating target labels...")

        df = df.copy()

        # Get prediction horizon from config
        prediction_horizon = self.config.get('data.prediction_horizon', 1)

        # Regression target: N-day ahead percentage return (log return)
        df['target_return'] = df.groupby('ticker')['close'].transform(
            lambda x: np.log(x.shift(-prediction_horizon) / x)
        )

        # Keep absolute price for reference
        df['target_price'] = df.groupby('ticker')['close'].shift(-prediction_horizon)

        # Classification target
        threshold_up = self.config.get('features.classification.threshold_up', 0.005)
        threshold_down = self.config.get('features.classification.threshold_down', -0.005)

        # Binary or three-class
        use_binary = abs(threshold_up - abs(threshold_down)) < 0.001

        if use_binary:
            df['target_class'] = (df['target_return'] > threshold_up).astype(int)
            self.logger.info(f"Binary classification: UP (1) if return > {threshold_up}")
        else:
            df['target_class'] = 2  # NEUTRAL
            df.loc[df['target_return'] > threshold_up, 'target_class'] = 1  # UP
            df.loc[df['target_return'] < threshold_down, 'target_class'] = 0  # DOWN
            self.logger.info(f"Three-class: UP (1) > {threshold_up}, DOWN (0) < {threshold_down}, NEUTRAL (2)")

        # Log distribution
        class_counts = df['target_class'].value_counts().sort_index()
        self.logger.info(f"Target class distribution: {class_counts.to_dict()}")

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

        # Handle timezone if present
        if df['date'].dt.tz is not None:
            tz = df['date'].dt.tz
            train_end = train_end.tz_localize(tz)
            val_start = val_start.tz_localize(tz)
            val_end = val_end.tz_localize(tz)
            test_start = test_start.tz_localize(tz)

        train_df = df[df['date'] <= train_end].copy()
        val_df = df[(df['date'] >= val_start) & (df['date'] <= val_end)].copy()
        test_df = df[df['date'] >= test_start].copy()

        self.logger.info(f"Train: {len(train_df)} rows ({train_df['date'].min()} to {train_df['date'].max()})")
        self.logger.info(f"Val: {len(val_df)} rows ({val_df['date'].min()} to {val_df['date'].max()})")
        self.logger.info(f"Test: {len(test_df)} rows ({test_df['date'].min()} to {test_df['date'].max()})")

        return train_df, val_df, test_df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns.

        Returns exactly 21 features (14 from Phase 1 + 7 new from Phase 2).

        Args:
            df: DataFrame with all columns

        Returns:
            List of feature column names
        """
        # Define the exact 21 features
        feature_cols = [
            # Phase 1: Core features (14)
            'ret_1d', 'ret_5d', 'ret_20d',  # Returns
            'vol_10d', 'vol_20d',  # Volatility
            'hl_range', 'oc_gap',  # Price structure
            'sma_10_dist', 'sma_20_dist',  # Trend
            'log_volume', 'volume_change',  # Volume
            'market_return', 'vs_market', 'market_correlation',  # Market
            # Phase 2: New features (7)
            'mom_5d', 'mom_20d', 'mom_60d',  # Momentum
            'vol_skew', 'vol_regime',  # Advanced volatility
            'amihud_illiq', 'volume_momentum'  # Microstructure
        ]

        # Verify all features exist
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            self.logger.warning(f"Missing features: {missing}")

        available_features = [col for col in feature_cols if col in df.columns]
        self.logger.info(f"Using {len(available_features)} features")

        return available_features

    def normalize_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Normalize features using RobustScaler (better for outliers).

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame

        Returns:
            Tuple of normalized (train_df, val_df, test_df)
        """
        self.logger.info("Normalizing features with RobustScaler...")

        # Get feature columns
        self.feature_columns = self.get_feature_columns(train_df)

        # Use RobustScaler (less sensitive to outliers than StandardScaler)
        self.feature_scaler = RobustScaler()

        # Fit on training data only
        train_df[self.feature_columns] = self.feature_scaler.fit_transform(
            train_df[self.feature_columns]
        )

        # Transform val and test
        val_df[self.feature_columns] = self.feature_scaler.transform(
            val_df[self.feature_columns]
        )

        test_df[self.feature_columns] = self.feature_scaler.transform(
            test_df[self.feature_columns]
        )

        self.logger.info(f"Normalized {len(self.feature_columns)} features")

        return train_df, val_df, test_df

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
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

        # Group by ticker
        for ticker, group in df.groupby('ticker'):
            group = group.sort_values('date').reset_index(drop=True)

            # Remove rows with NaN
            group = group.dropna(subset=self.feature_columns + ['target_return', 'target_class'])

            if len(group) < sequence_length + 1:
                continue

            # Create sequences
            for i in range(len(group) - sequence_length):
                X_seq = group.iloc[i:i + sequence_length][self.feature_columns].values
                y_reg = group.iloc[i + sequence_length]['target_return']
                y_clf = group.iloc[i + sequence_length]['target_class']

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
        Complete preprocessing pipeline with enhanced features.

        Args:
            input_file: Path to raw data file
            output_dir: Directory to save processed data

        Returns:
            Dictionary of output file paths
        """
        self.logger.info("="*70)
        self.logger.info("PHASE 2: ENHANCED FEATURE ENGINEERING")
        self.logger.info("="*70)
        self.logger.info("Starting preprocessing with 21 enhanced features...")

        # Load raw data
        self.logger.info(f"Loading raw data from {input_file}...")
        df = pd.read_parquet(input_file)

        # Calculate enhanced features (21 total)
        df = self.calculate_core_features(df)

        # Create labels
        df = self.create_labels(df)

        # Split by date
        train_df, val_df, test_df = self.split_by_date(df)

        # Remove NaN
        train_df = train_df.dropna()
        val_df = val_df.dropna()
        test_df = test_df.dropna()

        # Normalize
        train_df, val_df, test_df = self.normalize_features(train_df, val_df, test_df)

        # Save
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_file = output_path / "train_v3.parquet"
        val_file = output_path / "validation_v3.parquet"
        test_file = output_path / "test_v3.parquet"

        train_df.to_parquet(train_file)
        val_df.to_parquet(val_file)
        test_df.to_parquet(test_file)

        self.logger.info(f"Saved training data to {train_file}")
        self.logger.info(f"Saved validation data to {val_file}")
        self.logger.info(f"Saved test data to {test_file}")
        self.logger.info("="*70)

        return {
            'train': str(train_file),
            'validation': str(val_file),
            'test': str(test_file)
        }
