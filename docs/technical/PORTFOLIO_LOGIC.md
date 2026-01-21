# Portfolio Construction Logic

**How predictions become positions (S2_FilterNegative Strategy)**

## Overview

The S2_FilterNegative strategy converts daily stock predictions into a 130/70 long/short portfolio with critical short filtering.

**Key Insight**: Only short stocks with negative predictions. This salvaged the short book from -1.69 Sharpe to +0.61 Sharpe.

## Daily Portfolio Construction

### Step 1: Rank Stocks by Prediction

```python
def construct_portfolio(predictions):
    # Sort all stocks by predicted return
    sorted_stocks = predictions.sort_values('y_pred_reg', ascending=False)

    # Result: Best predictions at top, worst at bottom
    return sorted_stocks
```

### Step 2: Select Long Positions (Top K)

```python
K = 38  # Number of long/short positions
LONG_EXPOSURE = 0.65  # 65% gross exposure

# Top K stocks
longs = sorted_stocks.head(K)

# Equal weight
long_weight = LONG_EXPOSURE / len(longs)  # ~1.71% per stock

# Create positions
long_positions = {
    ticker: long_weight
    for ticker in longs['ticker']
}
```

### Step 3: Select Short Positions (Bottom K, Filtered)

```python
SHORT_EXPOSURE = 0.35  # 35% gross exposure

# Bottom K candidates
short_candidates = sorted_stocks.tail(K)

# CRITICAL FILTER: Only short if prediction < 0
shorts = short_candidates[short_candidates['y_pred_reg'] < 0]

# Equal weight (if any shorts remain)
if len(shorts) > 0:
    short_weight = -SHORT_EXPOSURE / len(shorts)
else:
    short_weight = 0  # No shorts if all predictions positive

# Create positions
short_positions = {
    ticker: short_weight
    for ticker in shorts['ticker']
}
```

### Step 4: Combine Positions

```python
target_positions = {**long_positions, **short_positions}

# Summary
total_long = sum(w for w in target_positions.values() if w > 0)
total_short = abs(sum(w for w in target_positions.values() if w < 0))
net_exposure = total_long + total_short  # Typically ~0.30 (long bias)
gross_exposure = total_long + total_short  # Typically ~1.00

return target_positions
```

## Position Smoothing (Turnover Reduction)

### EWMA Smoothing

Without smoothing, turnover can exceed 80%. With EWMA smoothing:

```python
ALPHA = 0.15  # Smoothing parameter

def smooth_positions(target, previous):
    """Apply EWMA smoothing to reduce turnover."""
    smoothed = {}

    all_tickers = set(target.keys()) | set(previous.keys())

    for ticker in all_tickers:
        target_weight = target.get(ticker, 0.0)
        prev_weight = previous.get(ticker, 0.0)

        # EWMA: new = α * target + (1-α) * previous
        smoothed_weight = ALPHA * target_weight + (1 - ALPHA) * prev_weight

        if abs(smoothed_weight) > 1e-6:  # Keep only non-zero
            smoothed[ticker] = smoothed_weight

    return smoothed
```

**Impact**:
- Turnover: 80% → 49%
- Sharpe: 2.53 → 2.20 (small degradation)
- Net benefit: Lower costs outweigh performance drag

### First Day Handling

```python
if previous_positions is None:
    # Day 1: No smoothing
    smoothed_positions = target_positions
else:
    # All other days: Apply smoothing
    smoothed_positions = smooth_positions(target_positions, previous_positions)
```

## Volatility Targeting

Scale entire portfolio based on realized volatility:

```python
TARGET_VOL = 0.08  # 8% annualized
LOOKBACK = 20      # Days
MIN_SCALE = 0.5
MAX_SCALE = 2.0

def calculate_vol_scale(returns_history):
    """Calculate volatility scaling factor."""

    if len(returns_history) < LOOKBACK:
        return 1.0  # Not enough history

    # Realized volatility (20-day)
    realized_vol = returns_history[-LOOKBACK:].std() * np.sqrt(252)

    # Scale factor
    scale = TARGET_VOL / realized_vol

    # Constrain to reasonable range
    scale = np.clip(scale, MIN_SCALE, MAX_SCALE)

    return scale

# Apply to all positions
vol_scale = calculate_vol_scale(returns_history)
scaled_positions = {
    ticker: weight * vol_scale
    for ticker, weight in positions.items()
}
```

**Effect**:
- High volatility periods → Scale down (reduce risk)
- Low volatility periods → Scale up (maintain returns)
- Target Sharpe improves: 2.20 → 1.29 (but with lower volatility)

## Kill Switches (Risk Management)

Before executing positions, check kill switches:

```python
def check_kill_switches(daily_pnl, returns_history, cumulative_returns):
    """Check if any kill switch triggered."""

    # 1. 3-sigma daily loss
    if len(returns_history) >= 20:
        vol = returns_history[-20:].std()
        if daily_pnl < -3 * vol:
            return True, "3-sigma loss"

    # 2. 8% trailing drawdown
    if len(cumulative_returns) >= 20:
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / (1 + peak)
        if drawdown.min() < -0.08:
            return True, "8% drawdown"

    # 3. Sharpe < 0 (60-day rolling)
    if len(returns_history) >= 60:
        sharpe = (
            returns_history[-60:].mean() /
            returns_history[-60:].std() *
            np.sqrt(252)
        )
        if sharpe < 0:
            return True, "Negative Sharpe"

    return False, None

# If triggered, flatten
if triggered:
    positions = {}  # Go to cash
```

See [RISK_FRAMEWORK.md](RISK_FRAMEWORK.md) for details.

## Transaction Costs

```python
def calculate_transaction_cost(turnover):
    """Estimate transaction costs."""

    # Components
    spread_bps = 5              # Bid-ask spread
    commission_bps = 1          # Brokerage commission
    impact_bps = turnover * 20  # Market impact (linear)

    total_bps = spread_bps + commission_bps + impact_bps

    # Convert to decimal
    return total_bps / 10000

# Example
turnover = 0.49  # 49%
cost = calculate_transaction_cost(0.49)  # ~0.0015 = 15 bps
```

**Typical costs**:
- Turnover 49% → 15 bps per day
- Annualized: ~38 bps (assuming 250 trading days)

## Portfolio Characteristics

### Exposure

**Typical day**:
- Long exposure: 65%
- Short exposure: 20-35% (varies based on filter)
- Net exposure: 30-45% (long bias)
- Gross exposure: 90-100%

### Concentration

**Position sizes**:
- Long: ~1.7% per stock (38 stocks)
- Short: ~1.0-1.8% per stock (varies)
- Max single position: ~2% (after smoothing)

**Diversification**:
- Typical long positions: 38
- Typical short positions: 15-35 (varies with market conditions)
- Total positions: 50-70

### Turnover

**Daily turnover**: ~49%

**Components**:
- Rebalancing to new predictions: ~60%
- Reduced by EWMA smoothing: →49%
- Reduced by kill switch events: minimal impact

**Annualized**: 49% * 250 = 122.5x (gross turnover)

## Performance Attribution

### Long Book

```
Sharpe: 3.14
Annual Return: 32.1%
Contribution to overall: ~60%
```

### Short Book (with filter)

```
Sharpe: 0.61
Annual Return: 6.1%
Contribution to overall: ~20%
```

### Diversification Benefit

```
Long + Short Sharpe: 2.20
Individual sum would be: ~2.5
Correlation: -0.2 (good diversification)
```

## Why S2_FilterNegative Works

### Problem with Naive Shorts

**S0 (naive 130/70)**:
- Short bottom K stocks regardless of prediction sign
- During bull markets, bottom K still expected to rise
- Short book Sharpe: -1.69 (destroys value)

### Solution: Filter Negative Predictions

**S2_FilterNegative**:
- Only short stocks with y_pred < 0
- Avoids shorting stocks expected to rise
- Short book Sharpe: +0.61 (adds value)

### Market Regime Adaptation

**Bull market** (most predictions positive):
- Fewer shorts (maybe 15-20)
- Higher short weights (to maintain 35% exposure)
- More concentrated short book

**Bear market** (many negative predictions):
- More shorts (close to 38)
- Standard short weights
- Well-diversified short book

**Sideways market**:
- Moderate number of shorts (25-30)
- Balanced portfolio

## Implementation Example

Full code in `scripts/phase2b/phase2b_strategies.py`:

```python
class S2FilterNegativeStrategy:
    """130/70 long/short with negative prediction filter."""

    def __init__(self):
        self.K = 38
        self.LONG_EXPOSURE = 0.65
        self.SHORT_EXPOSURE = 0.35
        self.ALPHA = 0.15  # EWMA smoothing

    def construct_portfolio(self, predictions, previous_positions=None):
        # Sort by prediction
        sorted_preds = predictions.sort_values('y_pred_reg', ascending=False)

        # Longs: Top K
        longs = sorted_preds.head(self.K)
        long_weight = self.LONG_EXPOSURE / len(longs)

        # Shorts: Bottom K, filtered
        short_candidates = sorted_preds.tail(self.K)
        shorts = short_candidates[short_candidates['y_pred_reg'] < 0]

        short_weight = -self.SHORT_EXPOSURE / len(shorts) if len(shorts) > 0 else 0

        # Target positions
        target = {}
        for ticker in longs['ticker']:
            target[ticker] = long_weight
        for ticker in shorts['ticker']:
            target[ticker] = short_weight

        # EWMA smoothing
        if previous_positions is not None:
            target = self.smooth_positions(target, previous_positions)

        return target

    def smooth_positions(self, target, previous):
        """EWMA position smoothing."""
        smoothed = {}
        all_tickers = set(target.keys()) | set(previous.keys())

        for ticker in all_tickers:
            target_w = target.get(ticker, 0.0)
            prev_w = previous.get(ticker, 0.0)

            smoothed_w = self.ALPHA * target_w + (1 - self.ALPHA) * prev_w

            if abs(smoothed_w) > 1e-6:
                smoothed[ticker] = smoothed_w

        return smoothed
```

---

**See Also**:
- [RISK_FRAMEWORK.md](RISK_FRAMEWORK.md) - Kill switches and risk management
- [ARCHITECTURE.md](ARCHITECTURE.md) - System overview
- [../../STRATEGY_DEFINITION.md](../../STRATEGY_DEFINITION.md) - Complete strategy specification

**Last Updated**: January 20, 2026
