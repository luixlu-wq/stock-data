# Feature Engineering

**14 Core Features for Stock Return Prediction**

## Overview

The LSTM model uses 14 carefully engineered features that capture:
- Price momentum and mean reversion
- Volatility regimes
- Volume dynamics
- Market context

These features are calculated for each stock on each trading day, creating a 90-day sequence as input to the LSTM.

## Feature Catalog

### Returns (3 features)

**1. Daily Return (ret_1d)**
```python
ret_1d = (close[t] - close[t-1]) / close[t-1]
```
- Captures very short-term momentum
- Highly mean-reverting at daily frequency
- Most noisy feature but contains alpha

**2. 5-Day Return (ret_5d)**
```python
ret_5d = (close[t] - close[t-5]) / close[t-5]
```
- Short-term momentum signal
- Less noisy than 1-day
- Balances momentum and reversion

**3. 20-Day Return (ret_20d)**
```python
ret_20d = (close[t] - close[t-20]) / close[t-20]
```
- Intermediate momentum
- Trend-following component
- More stable signal

**Why multiple horizons?**
- Different alpha decay rates
- Momentum vs mean reversion trade-off
- Provides temporal context to LSTM

### Volatility (3 features)

**4. 20-Day Realized Volatility (volatility)**
```python
volatility = returns[-20:].std() * sqrt(252)
```
- Annualized realized volatility
- Risk regime indicator
- Mean-reverting process

**5. 5-Day Volatility (vol_5d)**
```python
vol_5d = returns[-5:].std() * sqrt(252)
```
- Short-term volatility spike detection
- Captures recent market stress
- Leading indicator for reversals

**6. Volume Volatility (vol_volume)**
```python
vol_volume = volume[-20:].std() / volume[-20:].mean()
```
- Normalized volume volatility
- Indicates unusual trading activity
- Precedes price moves

**Why volatility features?**
- Predict return magnitude (not just direction)
- Risk-adjusted signals
- Regime changes (low vol → high vol transitions)

### Price Structure (3 features)

**7. Distance from 20-Day High (dist_from_high)**
```python
dist_from_high = (high_20d - close[t]) / close[t]
```
- How far from recent peak
- Breakout vs pullback indicator
- Asymmetric: closer to high = momentum

**8. Distance from 20-Day Low (dist_from_low)**
```python
dist_from_low = (close[t] - low_20d) / close[t]
```
- How far from recent trough
- Oversold/bounce indicator
- Mean reversion signal

**9. Price Range Compression (price_range)**
```python
price_range = (high[t] - low[t]) / close[t]
```
- Intraday volatility
- Compression → expansion pattern
- Liquidity/activity measure

**Why price structure?**
- Technical patterns (support/resistance)
- Volatility expansion/compression
- Non-linear price dynamics

### Trend (2 features)

**10. 20-Day Momentum (momentum)**
```python
momentum = (close[t] - close[t-20]) / close[t-20]
```
- Identical to ret_20d (normalized)
- Trend strength indicator
- Cross-sectional ranking signal

**11. 5-Day Momentum (momentum_5d)**
```python
momentum_5d = (close[t] - close[t-5]) / close[t-5]
```
- Short-term trend
- Reversal detection
- Complements 20-day momentum

**Why separate momentum features?**
- Explicit trend indicators
- LSTM learns momentum persistence
- Multi-scale trend analysis

### Volume (2 features)

**12. Volume Ratio (volume_ratio)**
```python
volume_ratio = volume[t] / volume[-20:].mean()
```
- Relative trading activity
- Conviction indicator (high volume = strong signal)
- Breakout confirmation

**13. Volume Trend (volume_trend)**
```python
volume_trend = volume[-5:].mean() / volume[-20:].mean()
```
- Volume momentum
- Increasing volume = accumulation/distribution
- Leading indicator for price moves

**Why volume features?**
- Confirms price moves (volume leads price)
- Differentiates noise from signal
- Institutional activity proxy

### Market Context (1 feature)

**14. Market Return (market_return)**
```python
market_return = equal_weighted_return_of_universe[t]
```
- Cross-sectional context
- Market regime (bull/bear/neutral)
- Factor exposure control

**Why market context?**
- Model learns stock-specific alpha (not beta)
- Adjusts for market environment
- Implicit factor model

## Feature Engineering Code

Implementation in `src/data/preprocessor_v2.py`:

```python
class SimplifiedStockPreprocessor:
    def calculate_core_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate 14 core features."""

        # Returns
        df['ret_1d'] = df['close'].pct_change()
        df['ret_5d'] = df['close'].pct_change(5)
        df['ret_20d'] = df['close'].pct_change(20)

        # Volatility
        df['volatility'] = df['ret_1d'].rolling(20).std() * np.sqrt(252)
        df['vol_5d'] = df['ret_1d'].rolling(5).std() * np.sqrt(252)
        df['vol_volume'] = (
            df['volume'].rolling(20).std() /
            df['volume'].rolling(20).mean()
        )

        # Price structure
        df['high_20d'] = df['high'].rolling(20).max()
        df['low_20d'] = df['low'].rolling(20).min()
        df['dist_from_high'] = (df['high_20d'] - df['close']) / df['close']
        df['dist_from_low'] = (df['close'] - df['low_20d']) / df['close']
        df['price_range'] = (df['high'] - df['low']) / df['close']

        # Trend
        df['momentum'] = df['close'].pct_change(20)
        df['momentum_5d'] = df['close'].pct_change(5)

        # Volume
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        df['volume_trend'] = (
            df['volume'].rolling(5).mean() /
            df['volume'].rolling(20).mean()
        )

        # Market context (calculated at portfolio level)
        # Added during sequence creation

        return df
```

## Feature Statistics (from training data)

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| ret_1d | 0.0005 | 0.02 | -0.20 | 0.25 |
| ret_5d | 0.0025 | 0.05 | -0.40 | 0.50 |
| ret_20d | 0.010 | 0.10 | -0.60 | 0.80 |
| volatility | 0.35 | 0.20 | 0.10 | 2.50 |
| vol_5d | 0.40 | 0.30 | 0.10 | 3.00 |
| vol_volume | 0.50 | 0.25 | 0.10 | 2.00 |
| dist_from_high | 0.05 | 0.04 | 0.00 | 0.30 |
| dist_from_low | 0.05 | 0.04 | 0.00 | 0.30 |
| price_range | 0.02 | 0.01 | 0.00 | 0.15 |
| momentum | 0.010 | 0.10 | -0.60 | 0.80 |
| momentum_5d | 0.0025 | 0.05 | -0.40 | 0.50 |
| volume_ratio | 1.00 | 0.60 | 0.10 | 5.00 |
| volume_trend | 1.00 | 0.30 | 0.20 | 3.00 |
| market_return | 0.0005 | 0.01 | -0.10 | 0.10 |

## Feature Importance

From LSTM attention analysis (Phase 1):

**Top 5 Most Important Features**:
1. **ret_20d** (20-day momentum) - 18.5%
2. **volatility** (20-day vol) - 15.2%
3. **volume_ratio** - 12.8%
4. **ret_5d** (5-day momentum) - 11.3%
5. **dist_from_high** - 9.7%

**Why these matter**:
- Momentum features dominate (ret_20d, ret_5d)
- Volatility regime crucial for risk-adjusted returns
- Volume confirms price moves
- Distance from high captures breakouts

## Feature Correlations

**High Correlations** (expected and OK):
- ret_1d ↔ ret_5d (0.7)
- ret_5d ↔ ret_20d (0.6)
- momentum ↔ ret_20d (1.0) - identical by construction
- volume_ratio ↔ volume_trend (0.5)

**Low Correlations** (good for model):
- Returns ↔ volatility (0.1)
- Price structure ↔ volume (-0.05)
- Market return ↔ stock returns (0.3) - appropriate level

## Feature Normalization

**Method**: Z-score standardization (per feature, over time)
```python
feature_normalized = (feature - mean) / std
```

**Why**:
- LSTM performs better with normalized inputs
- Different feature scales (returns vs ratios vs volatility)
- Prevents gradient issues during training

**Applied**:
- At sequence creation time (before LSTM input)
- Using rolling statistics (to avoid look-ahead bias)

## Feature Sequence Construction

**Input to LSTM**: 90-day sequences

```python
# For prediction on day T:
sequence = features[T-89:T+1]  # 90 days ending on T
target = return[T+1]  # Next day return

# Shape: (90, 14)
# - 90 time steps
# - 14 features per time step
```

**Why 90 days?**
- Captures ~4 months of history
- Long enough for trend learning
- Short enough to avoid stale information

## Feature Engineering Principles

### 1. No Look-Ahead Bias
All features use only information available at time T:
```python
# WRONG (uses future data)
volatility = returns[T-10:T+10].std()

# CORRECT (only past data)
volatility = returns[T-19:T+1].std()
```

### 2. Standardization
Z-score normalization using expanding/rolling window:
```python
mean = features[:T].mean()  # Only past data
std = features[:T].std()
feature_norm = (feature[T] - mean) / std
```

### 3. Robustness
- Winsorize extreme outliers (1st/99th percentile)
- Handle missing data (forward fill, max 5 days)
- Avoid division by zero (add epsilon)

### 4. Interpretability
- Features have clear economic meaning
- Can be explained to traders/risk managers
- Debuggable when model fails

## Alpha Decay Analysis

**How quickly does predictive power decay?**

| Prediction Horizon | Spearman Correlation | Sharpe Ratio |
|--------------------|----------------------|--------------|
| T+1 (next day) | 0.12 | 2.53 |
| T+2 (2 days) | 0.08 | 1.45 |
| T+5 (1 week) | 0.05 | 0.82 |
| T+20 (1 month) | 0.02 | 0.31 |

**Conclusion**: Alpha decays rapidly. Daily rebalancing is essential.

## Feature Stability

**Features ranked by stability** (low turnover in rankings):
1. volatility (most stable)
2. ret_20d
3. volume_ratio
4. ret_5d
5. ret_1d (least stable - most noise)

**Implication**: Use EWMA smoothing in portfolio construction to avoid chasing noise.

## Future Feature Ideas (not implemented)

**Potentially valuable but not in current frozen strategy**:
- Earnings-based features (EPS growth, surprises)
- Options-implied volatility
- Short interest
- Analyst sentiment
- Macroeconomic factors
- Industry/sector indicators

**Why not included?**:
- Current 14 features achieved Sharpe 2.20
- More features = more overfitting risk
- Simpler = more robust
- Strategy is frozen at v2.0.0

---

**See Also**:
- [ARCHITECTURE.md](ARCHITECTURE.md) - System overview
- [MODEL_DETAILS.md](MODEL_DETAILS.md) - LSTM architecture
- [../../STRATEGY_DEFINITION.md](../../STRATEGY_DEFINITION.md) - Frozen strategy specification

**Last Updated**: January 20, 2026
