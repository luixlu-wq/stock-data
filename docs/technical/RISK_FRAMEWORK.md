# Risk Management Framework

**Volatility targeting and kill switches to control risk**

## Overview

The risk framework has two main components:
1. **Volatility Targeting**: Scale positions to maintain target volatility
2. **Kill Switches**: Flatten portfolio when risk thresholds breached

Together they transform Sharpe 2.20 (net) to Sharpe 1.29 (vol-targeted) with reduced drawdowns.

## Volatility Targeting

### Objective

Maintain consistent **8% annualized volatility** regardless of market conditions.

### Implementation

```python
TARGET_VOL = 0.08      # 8% annual target
LOOKBACK = 20          # 20-day rolling window
MIN_SCALE = 0.5        # Minimum scale factor
MAX_SCALE = 2.0        # Maximum scale factor

def calculate_vol_scale(returns_history):
    """
    Calculate scaling factor based on realized volatility.

    Returns:
        float: Scale factor to apply to all positions
    """
    if len(returns_history) < LOOKBACK:
        return 1.0  # Not enough history, no scaling

    # Calculate 20-day realized volatility
    daily_vol = returns_history[-LOOKBACK:].std()
    annual_vol = daily_vol * np.sqrt(252)

    # Calculate scale factor
    scale = TARGET_VOL / annual_vol

    # Constrain to reasonable range
    scale = np.clip(scale, MIN_SCALE, MAX_SCALE)

    return scale
```

### Scaling Logic

**Low volatility regime** (realized vol < 8%):
```
realized_vol = 5%
scale = 8% / 5% = 1.6
→ Scale positions UP by 1.6x (up to 2.0x max)
```

**High volatility regime** (realized vol > 8%):
```
realized_vol = 12%
scale = 8% / 12% = 0.67
→ Scale positions DOWN by 0.67x (down to 0.5x min)
```

**Target volatility regime** (realized vol ≈ 8%):
```
realized_vol = 8%
scale = 8% / 8% = 1.0
→ No scaling
```

### Application to Positions

```python
# Original positions (from portfolio construction)
base_positions = construct_portfolio(predictions)

# Calculate vol scale
vol_scale = calculate_vol_scale(returns_history)

# Apply scaling to all positions
scaled_positions = {
    ticker: weight * vol_scale
    for ticker, weight in base_positions.items()
}
```

**Example**:
```
Base long position: +1.71%
Vol scale: 1.5 (low vol regime)
Scaled position: +2.57%

Base short position: -1.00%
Vol scale: 0.7 (high vol regime)
Scaled position: -0.70%
```

### Performance Impact

**Without vol targeting (Phase 2B)**:
- Sharpe: 2.20
- Annual Return: 22.4%
- Volatility: 10.2%
- Max DD: -6.3%

**With vol targeting (Phase 3.5)**:
- Vol-Targeted Sharpe: 1.29
- Annual Return: 11.95%
- Volatility: 9.27% (closer to 8% target)
- Max DD: -5.21%

**Trade-off**: Lower Sharpe but more consistent volatility and lower drawdowns.

## Kill Switches

### Objective

Automatically flatten portfolio when experiencing:
1. Extreme daily losses
2. Deep drawdowns
3. Persistent negative performance

### Kill Switch 1: 3-Sigma Daily Loss

**Logic**: Flatten if single-day loss exceeds 3 standard deviations.

```python
def check_3sigma_loss(daily_pnl, returns_history, min_history=20):
    """
    Check if daily P&L is extreme outlier.

    Returns:
        (bool, str): (triggered, reason)
    """
    if len(returns_history) < min_history:
        return False, None  # Not enough history

    # Calculate historical volatility
    vol = returns_history[-20:].std()

    # Check if loss > 3 sigma
    if daily_pnl < -3 * vol:
        return True, f"3-sigma loss: {daily_pnl:.4f} < {-3*vol:.4f}"

    return False, None
```

**Example**:
```
Historical volatility (20-day): 0.8% per day
3-sigma threshold: -2.4%

Day T P&L: -3.1%
→ TRIGGER: -3.1% < -2.4%
→ Flatten all positions
```

**Why 3 sigma?**
- Rare event (0.3% probability under normal distribution)
- Indicates something fundamentally wrong
- Prevents catastrophic losses from model failure

### Kill Switch 2: 8% Trailing Drawdown

**Logic**: Flatten if portfolio declines 8% from recent peak.

```python
def check_trailing_drawdown(cumulative_returns, threshold=-0.08, min_history=20):
    """
    Check if trailing drawdown exceeds threshold.

    Returns:
        (bool, str): (triggered, reason)
    """
    if len(cumulative_returns) < min_history:
        return False, None

    # Calculate trailing peak
    peak = cumulative_returns.cummax()

    # Calculate drawdown from peak
    drawdown = (cumulative_returns - peak) / (1 + peak)

    current_dd = drawdown.iloc[-1]

    if current_dd < threshold:
        return True, f"Drawdown: {current_dd:.2%} < {threshold:.2%}"

    return False, None
```

**Example**:
```
Peak cumulative return: +15%
Current cumulative return: +6%
Drawdown: (6% - 15%) / (1 + 15%) = -7.8%

→ Still OK (threshold is -8%)

If current drops to +5.5%:
Drawdown: (5.5% - 15%) / 1.15 = -8.3%
→ TRIGGER: Flatten all positions
```

**Why 8%?**
- Historically, strategy has max DD of 6.3%
- 8% is ~1.3x historical max (allows some breathing room)
- Prevents deep drawdowns (>10%) that damage psychology

### Kill Switch 3: Negative Sharpe (60-day rolling)

**Logic**: Flatten if strategy stops working (Sharpe < 0 over 60 days).

```python
def check_negative_sharpe(returns_history, window=60, min_history=60):
    """
    Check if rolling Sharpe ratio is negative.

    Returns:
        (bool, str): (triggered, reason)
    """
    if len(returns_history) < min_history:
        return False, None

    # Calculate 60-day Sharpe
    recent_returns = returns_history[-window:]

    mean_return = recent_returns.mean()
    vol = recent_returns.std()

    sharpe = (mean_return / vol) * np.sqrt(252) if vol > 0 else 0

    if sharpe < 0:
        return True, f"Negative Sharpe: {sharpe:.2f}"

    return False, None
```

**Example**:
```
60-day average return: -0.05% per day
60-day volatility: 0.8% per day
Sharpe = (-0.05 / 0.8) * sqrt(252) = -0.99

→ TRIGGER: Sharpe -0.99 < 0
→ Flatten all positions
```

**Why negative Sharpe?**
- Indicates strategy no longer has edge
- Could be model degradation or regime change
- Better to go to cash than keep losing money

### Combined Kill Switch Check

```python
def check_all_kill_switches(daily_pnl, returns_history, cumulative_returns):
    """
    Check all kill switches.

    Returns:
        (bool, str): (any_triggered, reason)
    """
    # Check each kill switch
    ks1_triggered, ks1_reason = check_3sigma_loss(daily_pnl, returns_history)
    ks2_triggered, ks2_reason = check_trailing_drawdown(cumulative_returns)
    ks3_triggered, ks3_reason = check_negative_sharpe(returns_history)

    # Return first triggered
    if ks1_triggered:
        return True, ks1_reason
    if ks2_triggered:
        return True, ks2_reason
    if ks3_triggered:
        return True, ks3_reason

    return False, None


# In trading loop
ks_triggered, ks_reason = check_all_kill_switches(
    daily_pnl, returns_history, cumulative_returns
)

if ks_triggered:
    logger.warning(f"Kill switch triggered: {ks_reason}")

    # Flatten all positions
    positions = {}

    # Record event
    record_kill_switch_event(date, ks_reason)
```

### Kill Switch Recovery

**Question**: When do we resume trading after kill switch?

**Current approach**: Manual decision
- Review what went wrong
- Determine if it's fixable
- Re-enable trading manually

**Future enhancement**: Automatic recovery
- After X days
- When Sharpe returns to positive
- When drawdown recovers to -4%

## Risk Metrics Monitoring

### Daily Metrics

Calculated and logged every day:

```python
# Volatility
realized_vol_20d = returns[-20:].std() * np.sqrt(252)
vol_scale = TARGET_VOL / realized_vol_20d

# Drawdown
peak = cumulative_returns.cummax()
current_dd = (cumulative_returns[-1] - peak[-1]) / (1 + peak[-1])

# Sharpe (60-day rolling)
sharpe_60d = (returns[-60:].mean() / returns[-60:].std()) * np.sqrt(252)

# VaR (Value at Risk, 95%)
var_95 = np.percentile(returns[-60:], 5)  # 5th percentile

# Log all metrics
logger.info(f"Risk Metrics - Vol: {realized_vol_20d:.2%}, "
            f"DD: {current_dd:.2%}, Sharpe60d: {sharpe_60d:.2f}, "
            f"VaR95: {var_95:.2%}")
```

### Performance Report

Generated weekly/monthly:

```
=== RISK REPORT ===

Volatility:
  Target:           8.00%
  Realized (20d):   9.27%
  Vol Scale:        0.86

Drawdown:
  Current:         -2.1%
  Max (all-time):  -5.2%
  Kill switch:     -8.0% (not triggered)

Sharpe Ratios:
  60-day:           1.45
  All-time:         1.29
  Kill switch:      0.00 (not triggered)

Kill Switch Events:
  Total:            2
  Last triggered:   2025-07-15 (3-sigma loss)

VaR (95%, 60-day):  -1.8% (daily)
```

## Historical Performance (Phase 3.5)

### Backtest Results

**Period**: 2025-04-01 to 2025-10-31 (159 days)

**Vol-Targeted Performance**:
- Sharpe: 1.29
- Annual Return: 11.95%
- Volatility: 9.27%
- Max DD: -5.21%

**Kill Switch Events**: 0 (none triggered)

**Vol Scaling Distribution**:
```
Mean: 0.95
Std: 0.25
Min: 0.50 (high vol days)
Max: 2.00 (low vol days)

Percentiles:
  5%:  0.60 (scaled down in high vol)
 25%:  0.80
 50%:  0.95
 75%:  1.10
 95%:  1.40 (scaled up in low vol)
```

## Risk Budget

### Position Limits

**Single stock**:
- Max long: 3% (rare, after smoothing)
- Max short: 3%
- Typical: 1-2%

**Sector** (implicit, not enforced):
- Max: ~20% (38 stocks → natural diversification)

**Gross exposure**:
- Target: 100%
- After vol scaling: 50% to 200%
- Typical: 95%

**Net exposure**:
- Target: ~30% (long bias)
- After filtering shorts: 20-45%

### Leverage Constraints

**No explicit leverage** in current strategy:
- Long: 65%
- Short: 35%
- Cash: 0%
- Total: 100% of capital

**With vol scaling**:
- Max gross: 200% (scale 2.0x in low vol)
- Requires leverage (margin)

## Risk Attribution

### Where Does Risk Come From?

**Market risk (beta)**: ~40%
- Net long bias (30% average)
- Correlation to SPY: 0.4

**Factor risk**: ~30%
- Momentum factor
- Low vol factor
- Quality factor

**Idiosyncratic risk**: ~30%
- Stock-specific
- Diversified across 50-70 positions

### Risk Decomposition

```
Total volatility: 9.27%

Components:
  Market beta:        3.7% (40%)
  Factor exposure:    2.8% (30%)
  Stock-specific:     2.8% (30%)

Total: sqrt(3.7² + 2.8² + 2.8²) ≈ 5.2%
(Residual from correlation effects)
```

## Stress Testing

### Scenario Analysis

**What if 2008 crisis repeats?**
- Market drops 50% over 6 months
- Volatility spikes to 40%+

Expected behavior:
1. Vol targeting scales down to 0.5x (minimum)
2. Gross exposure: 100% → 50%
3. Kill switch triggered (likely 8% DD)
4. Portfolio flattened

Outcome: Max loss ~8-10% (vs 50% market loss)

**What if 2020 COVID crash repeats?**
- Market drops 30% in 1 month
- Volatility spikes to 60%

Expected:
1. Vol scale → 0.5x
2. 3-sigma kill switch likely triggers
3. Flatten on -3% day

Outcome: Max loss ~5-8%

## Future Enhancements

**Not in current frozen strategy** (would require revalidation):

1. **Dynamic kill switch thresholds**
   - Adjust based on market vol regime
   - Tighter in low vol, looser in high vol

2. **Partial flattening**
   - Instead of 100% → 0%, go to 50%
   - Gradual de-risking

3. **Automatic recovery**
   - Resume trading when conditions improve
   - Phased re-entry

4. **Factor exposure limits**
   - Explicit momentum, value limits
   - Risk parity across factors

5. **Tail risk hedges**
   - Long volatility positions
   - Put options

---

**See Also**:
- [PORTFOLIO_LOGIC.md](PORTFOLIO_LOGIC.md) - Portfolio construction
- [ARCHITECTURE.md](ARCHITECTURE.md) - System overview
- [../../STRATEGY_DEFINITION.md](../../STRATEGY_DEFINITION.md) - Complete strategy specification

**Last Updated**: January 20, 2026
