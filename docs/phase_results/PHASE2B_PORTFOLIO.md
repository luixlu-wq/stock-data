# Phase 2B: Portfolio Construction

**Test 5 portfolio strategies to find optimal construction**

## Objective

Find best way to convert predictions into portfolio positions.

## Strategies Tested

### S0: Naive 130/70
- Long top K, Short bottom K (no filter)
- **Sharpe**: 0.51 (destroyed by shorts)

### S1: Long-Only
- Long top K only
- **Sharpe**: 3.14

### S2: FilterNegative (WINNER)
- Long top K, Short bottom K **WHERE y_pred < 0**
- **Sharpe**: 2.20 (net of costs)

### S3: Dynamic Threshold
- Short only if pred < -0.5%
- **Sharpe**: 2.15

### S4: Scaled by Confidence
- Weight by prediction magnitude
- **Sharpe**: 2.05

## Winning Strategy: S2_FilterNegative

### Performance (Net of Costs)

| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | 2.20 |
| **Annual Return** | 22.4% |
| **Volatility** | 10.2% |
| **Max Drawdown** | -6.3% |
| **Average Turnover** | 49.1% |
| **Average Cost** | 9.8 bps |

### Attribution

**Long Book**:
- Sharpe: 3.14
- Contribution: ~60%

**Short Book** (with filter):
- Sharpe: 0.61
- Contribution: ~20%

**Diversification**: ~20%

### Why It Works

**Problem**: Naive shorts (S0) had Sharpe -1.69
- During bull markets, bottom K stocks still rise
- Shorting rising stocks destroys value

**Solution**: Filter to only negative predictions
- Only short stocks expected to decline
- Short book improves: -1.69 → +0.61

### Configuration

```
K = 38 (number of longs/shorts)
Long Exposure = 65%
Short Exposure = 35%
EWMA α = 0.15 (turnover reduction)
```

## Transaction Costs

### Cost Model

```
Spread: 5 bps
Commission: 1 bps
Market Impact: 20 * turnover bps

Total: ~15 bps per day (at 49% turnover)
```

### Impact

- Gross Sharpe: 2.53
- Net Sharpe: 2.20
- Cost drag: ~12% of returns

## Status

✅ **Complete** - S2_FilterNegative selected as production strategy

---

**Next**: [PHASE3_RISK_ANALYSIS.md](PHASE3_RISK_ANALYSIS.md) - Risk management
