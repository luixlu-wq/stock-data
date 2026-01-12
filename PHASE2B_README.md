

# Phase 2B: Portfolio Monetization Engineering

## ChatGPT's Critical Insight

**Phase 2A Results**:
- Temperature = 0.05: Gross Sharpe **2.47**, Net Sharpe **0.10**
- Temperature = 0.10: Gross Sharpe **-0.17**, Net Sharpe **-2.55**

**Interpretation**:
> "You have real, fragile, high-turnover cross-sectional alpha that survives ONLY under sharp ranking. This is not a failure - it's a specific class of alpha that **cannot be traded naïvely** and must be monetized via **portfolio engineering**."

## What Changed: ML → Portfolio Engineering

**Before Phase 2B** (Wrong approach):
- ❌ Add more features
- ❌ Change architecture
- ❌ Tune more hyperparameters
- ❌ Try different models

**Phase 2B** (Correct approach):
- ✅ **Freeze the model** (it's extracting real alpha)
- ✅ Reduce trading costs (not signal strength)
- ✅ Portfolio engineering techniques

## 🔒 FROZEN Settings (No More ML Changes)

- Model architecture: LSTM
- Features: 14 (from Phase 1)
- Sequence length: 90 days
- Temperature: 0.05
- Rank weight: 0.7

**Why freeze?** The model already found alpha (Gross Sharpe 2.47). Problem is **monetization**, not **prediction**.

## Phase 2B Techniques

### 1. Rank-Change Trading Filter
**Problem**: Trading every small prediction change creates excessive turnover

**Solution**: Only trade when rank changes significantly
```python
# Only trade if rank changed by K percentiles
if |rank_t - rank_{t-1}| >= K:
    trade()
```

**Parameters**:
- K = 10 percentiles (default)
- Expected reduction: 30-40% turnover

### 2. Position EWMA Smoothing
**Problem**: Discrete position changes cause churn

**Solution**: Smooth position transitions using EWMA
```python
pos_t = 0.85 * pos_{t-1} + 0.15 * target_t
```

**Parameters**:
- alpha = 0.15 (15% new signal, 85% previous position)
- Reduces position flipping without losing signal

### 3. Cross-Sectional Normalization
**Problem**: Absolute prediction values create correlated trades

**Solution**: Z-score predictions within each day
```python
# Normalize predictions per trading day
pred_normalized = (pred - mean_per_day) / std_per_day
```

**Benefits**:
- Forces dollar neutrality
- Reduces market beta exposure
- More stable rankings

## Phase 2B Experiments

The script tests 5 configurations:

| Configuration | Rank Filter | EWMA | Z-Score | Expected Result |
|---------------|-------------|------|---------|-----------------|
| Baseline | ❌ | ❌ | ❌ | Gross 2.47, Net 0.10 |
| Rank Filter Only | ✅ K=10 | ❌ | ❌ | Lower turnover |
| Position Smoothing | ❌ | ✅ α=0.15 | ❌ | Smoother trades |
| Z-Score Only | ❌ | ❌ | ✅ | Better neutrality |
| **ALL TECHNIQUES** | ✅ | ✅ | ✅ | **Best Net Sharpe** |

## Expected Outcomes (ChatGPT's Targets)

If the alpha is real (evidence says yes):

| Metric | Target |
|--------|--------|
| Gross Sharpe | 1.8-2.3 (down from 2.47 but stable) |
| Net Sharpe | **0.4-0.6** (UP from 0.10) |
| Turnover | 60-75% (down from ~100%) |
| Costs | <5% (down from ~9%) |
| Max Drawdown | <-6% |

**That's tradeable.**

## How to Run

```bash
python phase2b_monetization.py
```

**Time**: ~5-10 minutes (no training, just backtesting)

**Output**:
- Comparison table of all 5 configurations
- Best configuration identified
- Final verdict on tradeability

## Success Criteria

### ✅ TRADEABLE (Target)
- Net Sharpe ≥ 0.4
- Turnover ≤ 80%
- Max Drawdown ≤ -8%

### ✓ Profitable but weak
- Net Sharpe > 0
- May need more refinements

### ❌ Still unprofitable
- Net Sharpe < 0 after all techniques
- May need different data sources or approach

## What Makes This Different

**Most ML projects**:
1. Train model
2. Get poor results
3. Add more features/layers/data
4. Repeat forever

**Professional quant approach** (what we're doing):
1. Find alpha signal (✅ Gross Sharpe 2.47)
2. **Freeze model**
3. Engineer portfolio to monetize
4. Ship when Net Sharpe > threshold

We're past the hard part. Now it's execution engineering.

## After Phase 2B

### If Net Sharpe ≥ 0.4:
- ✅ **Strategy is complete and tradeable**
- Next: Paper trading, live testing
- Consider: Risk limits, execution algorithms

### If 0 < Net Sharpe < 0.4:
- Partially successful
- Consider: Different holding periods, sector neutrality
- May still be usable in ensemble

### If Net Sharpe < 0:
- Portfolio engineering insufficient
- Alpha too fragile for this market
- Learn from experience, try different signal

Ready to run:
```bash
python phase2b_monetization.py
```
