# Phase 2A: Temperature Tuning (ChatGPT's Recommendation)

## Why This Approach (Not My Original Phase 2)

ChatGPT correctly identified that my Phase 2 plan was **overcomplicated**. Instead of adding 7 new features and changing everything at once, they recommend:

**Focus on the highest-leverage change first: Rank loss temperature**

## What ChatGPT Diagnosed

Phase 1 results showed:
- ✓ Turnover reduced 25% (rank loss worked!)
- ✗ Gross Sharpe dropped 0.71 → 0.55 (smoothed too much)
- ✗ Net Sharpe still negative

**Root cause**: Temperature = 1.0 is too high, causing over-smoothing

## Phase 2A: Simple, Focused Changes

### 1. Temperature Grid Search
Test: `[0.05, 0.1, 0.2, 0.3, 0.5, 1.0]`

**Temperature Effects**:
- **Low (0.05-0.1)**: Sharper rankings, may increase turnover slightly
- **Medium (0.2-0.3)**: Balance between sharpness and stability
- **High (0.5-1.0)**: Smoother, lower turnover, but loses signal

### 2. Sequence Length: 60 → 90 Days
- Captures 2-6 month horizons (equity alpha sweet spot)
- More stable than 60, less data-hungry than 120
- Should improve IC stability

### 3. Keep Everything Else Unchanged
- Same 14 features from Phase 1
- Same rank weight (0.7)
- Same preprocessing

## Success Criteria (ChatGPT's Targets)

**Minimum acceptable**:
- Gross Sharpe ≥ 0.6 (vs Phase 1's 0.55)
- Turnover ≤ 100% (vs Phase 1's 95%)
- Net Sharpe closer to 0 (vs Phase 1's -1.04)

**Strong success**:
- Gross Sharpe ≥ 0.7
- Net Sharpe > 0
- Turnover < 90%

## How to Run

```bash
python phase2a_temperature_experiment.py
```

**Time**: ~3-6 hours (trains 6 models: one per temperature)

**Output**:
- Comparison table of all temperatures
- Best temperature identified
- Results saved to `data/processed/phase2a_temperature_results.json`

## What Happens Next

### If Phase 2A succeeds (Gross Sharpe ≥ 0.6):
→ Proceed to **Phase 2B**: Add cross-sectional z-scoring
- Normalize features per day (cross-sectional)
- Add relative momentum (stock return - market median)

### If Phase 2A shows limited improvement:
→ Skip to **Phase 2B** anyway
- Temperature tuning alone may be insufficient
- Cross-sectional features are the next highest-leverage change

### If Phase 2B still fails (Net Sharpe < 0):
→ Deeper changes needed:
- Different architecture (Transformer, GRU)
- Ensemble models
- Check data quality issues

## Why This Approach is Better

**My original Phase 2**:
- Changed 5 things at once (features, sequence, temperature, weight)
- Can't tell what helped vs hurt
- Overcomplicated

**ChatGPT's Phase 2A**:
- Changes 2 things (temperature + sequence length)
- Clear attribution of what works
- Fast iteration
- Highest signal-to-effort ratio

## Expected Temperature Results

Based on theory:

| Temperature | Expected Gross Sharpe | Expected Turnover | Best For |
|-------------|----------------------|-------------------|----------|
| 0.05 | 0.65-0.75 | 100-120% | High signal, can handle turnover |
| 0.1 | 0.60-0.70 | 90-110% | Balanced |
| 0.2 | 0.55-0.65 | 80-100% | Stable rankings |
| 0.3 | 0.50-0.60 | 70-90% | Conservative |
| 0.5 | 0.45-0.55 | 60-80% | Very smooth |
| 1.0 | 0.40-0.50 | 50-70% | Phase 1 baseline |

**Prediction**: Optimal temperature will be 0.1-0.2

Ready to run:
```bash
python phase2a_temperature_experiment.py
```

This will take several hours. You can monitor progress in `logs/phase2a_temperature_experiment.log`.
