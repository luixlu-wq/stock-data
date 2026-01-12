# Phase 2 Quick Reference

## What's New
- **21 features** (was 14): Added momentum, vol skew, microstructure
- **120-day sequences** (was 60): Capture longer trends
- **Sharper rank loss** (temp=0.5): More stable predictions
- **Stronger rank weight** (0.8): Emphasize relative rankings

## Files Created
- `src/data/preprocessor_v3.py` - 21-feature preprocessor
- `phase2_train.py` - Training script
- `phase2_evaluate.py` - Evaluation & comparison
- `PHASE2_README.md` - Full documentation

## How to Run

### Train Phase 2
```bash
python phase2_train.py
```
**Time**: ~30-50 minutes (longer sequences)

### Evaluate Phase 2
```bash
python phase2_evaluate.py
```

## Expected Results

Target improvements over Phase 1:

| Metric | Baseline | Phase 1 | Phase 2 Goal |
|--------|----------|---------|--------------|
| Net Sharpe | -1.74 | -1.04 | **>0** |
| Turnover | 120% | 95% | **60-80%** |
| Gross Sharpe | 0.71 | 0.55 | **0.6-0.8** |

## Key Improvements

1. **Momentum Features (3)**
   - mom_5d, mom_20d, mom_60d
   - Captures trend continuation
   - Reduces false mean-reversion signals

2. **Advanced Volatility (2)**
   - vol_skew: Crash risk detection
   - vol_regime: High/low vol identification

3. **Microstructure (2)**
   - amihud_illiq: Liquidity measure
   - volume_momentum: Accumulation patterns

4. **Rank Loss Tuning**
   - Temperature 0.5 (sharper, was 1.0)
   - Weight 0.8 (stronger, was 0.7)

5. **Longer Context**
   - 120-day lookback (was 60)
   - Better long-term pattern capture

## What If Phase 2 Fails?

If net Sharpe is still <0 after Phase 2:

**Next Steps:**
1. Try different architectures (Transformer, GRU)
2. Ensemble multiple models
3. Add alternative data sources
4. Check for data quality issues
5. Consider market regime modeling

The key insight: We've now tried features, loss functions, and sequence lengths. If still unprofitable, the issue is likely architectural or data-related.
