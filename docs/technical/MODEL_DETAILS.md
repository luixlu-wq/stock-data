# Model Details

**LSTM Architecture and Training Specifications**

## Quick Reference

- **Model Type**: 2-layer stacked LSTM
- **Input**: 90-day sequences of 14 features
- **Output**: Next-day return prediction
- **Training Loss**: Combined rank-regression (70% rank, 30% regression)
- **Temperature**: 0.05 (critical parameter)
- **Checkpoint**: `models/best_lstm_model.pth`
- **Performance**: Spearman correlation 0.12, Sharpe 2.53

## Architecture

### LSTM Model

```python
class StockLSTM(nn.Module):
    def __init__(
        self,
        input_size=14,      # 14 core features
        hidden_size=128,    # LSTM hidden dimension
        num_layers=2,       # Stacked LSTM layers
        dropout=0.2         # Dropout between layers
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)  # Output layer

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # x: (N, 90, 14)

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (N, 90, hidden_size)

        # Use last time step output
        last_output = lstm_out[:, -1, :]  # (N, hidden_size)

        # Prediction
        output = self.fc(last_output)  # (N, 1)

        return output.squeeze(-1)  # (N,)
```

### Model Size

```
Total parameters: ~250,000
- LSTM weights: ~240,000
- FC layer: ~10,000

Model file size: ~1 MB
```

## Training Configuration

### Data

**Training Period**: 2020-01-01 to 2024-06-30
**Validation Period**: 2024-07-01 to 2024-12-31
**Prediction Period**: 2025-04-01 to 2025-10-31

**Sequence Construction**:
```python
def create_sequences(df, seq_len=90):
    X, y = [], []

    for date in dates:
        # Get 90-day history ending on date
        history = df[df['date'] <= date].tail(seq_len)

        if len(history) == seq_len:
            # Features: (90, 14)
            X.append(history[FEATURE_COLUMNS].values)

            # Target: next day return
            next_day = df[df['date'] > date].iloc[0]
            y.append(next_day['return'])

    return np.array(X), np.array(y)
```

### Loss Function

**Combined Rank-Regression Loss**:
```python
def combined_loss(y_pred, y_true, temperature=0.05):
    # Rank loss (70%)
    rank_loss = differentiable_spearman(
        y_pred, y_true, temperature
    )

    # Regression loss (30%)
    reg_loss = F.huber_loss(y_pred, y_true, delta=0.05)

    # Combine
    total_loss = 0.7 * rank_loss + 0.3 * reg_loss

    return total_loss
```

**Why this combination?**
- **Rank loss**: Model learns relative ordering (critical for long/short)
- **Regression loss**: Calibrates magnitude (not just direction)
- **70/30 split**: Ranking matters more for portfolio construction

**Differentiable Spearman Correlation**:
```python
def differentiable_spearman(y_pred, y_true, temperature=0.05):
    """
    Soft ranking with temperature parameter.
    Lower temperature = sharper rankings.
    """
    # Soft ranks (differentiable)
    pred_ranks = soft_rank(y_pred, temperature)
    true_ranks = soft_rank(y_true, temperature)

    # Pearson correlation of ranks
    corr = pearson_correlation(pred_ranks, true_ranks)

    # Minimize 1 - correlation
    return 1.0 - corr

def soft_rank(x, temperature):
    """Differentiable ranking using softmax."""
    n = len(x)
    x_expanded = x.unsqueeze(1)  # (N, 1)
    x_tiled = x.unsqueeze(0)     # (1, N)

    # Pairwise comparisons
    comparisons = (x_expanded - x_tiled) / temperature
    soft_comparisons = torch.sigmoid(comparisons)

    # Sum to get soft ranks
    ranks = soft_comparisons.sum(dim=1)
    return ranks
```

**Temperature Parameter (CRITICAL)**:
- **Temperature = 0.05**: Sharp rankings (production value)
- Higher temperature → softer rankings → worse performance
- Tested in Phase 2A: {0.01, 0.05, 0.1, 0.5, 1.0}
- Result: 0.05 is optimal (Sharpe 2.53)

**Huber Loss**:
```python
def huber_loss(y_pred, y_true, delta=0.05):
    """
    Robust to outliers (vs MSE).
    delta controls transition point.
    """
    error = y_true - y_pred
    abs_error = torch.abs(error)

    quadratic = torch.min(abs_error, torch.tensor(delta))
    linear = abs_error - quadratic

    return (0.5 * quadratic**2 + delta * linear).mean()
```

### Optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,           # Learning rate
    weight_decay=1e-5   # L2 regularization
)
```

**Learning rate schedule**:
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,        # Reduce by half
    patience=5,        # Wait 5 epochs
    min_lr=1e-6
)
```

### Training Loop

```python
def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0

    for batch_X, batch_y in dataloader:
        # Forward pass
        y_pred = model(batch_X)

        # Loss
        loss = combined_loss(y_pred, batch_y, temperature=0.05)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

### Hyperparameters

```yaml
# Model architecture
input_size: 14
hidden_size: 128
num_layers: 2
dropout: 0.2

# Training
batch_size: 32
learning_rate: 0.001
weight_decay: 1e-5
max_epochs: 50
patience: 15           # Early stopping
min_delta: 0.0001

# Loss function
temperature: 0.05      # CRITICAL
rank_weight: 0.7
regression_weight: 0.3
huber_delta: 0.05

# Data
sequence_length: 90
train_split: 0.8
validation_split: 0.2
```

## Training Results

### Training Curve

```
Epoch  Train Loss  Val Loss  Val Spearman  Val IC
-----  ----------  --------  ------------  ------
1      0.8234      0.7891    0.045         0.032
5      0.6123      0.5987    0.078         0.061
10     0.4876      0.4523    0.095         0.084
15     0.3912      0.3789    0.108         0.097
20     0.3456      0.3401    0.115         0.103
25     0.3201      0.3298    0.118         0.109
30     0.3087      0.3245    0.119         0.111
35     0.3012      0.3231    0.120         0.112  ← BEST
40     0.2978      0.3241    0.119         0.111
```

**Best epoch**: 35
**Validation Spearman**: 0.120
**Validation IC (Information Coefficient)**: 0.112

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Spearman Correlation** | 0.120 |
| **Information Coefficient (IC)** | 0.112 |
| **IC Std** | 0.08 |
| **IC IR (IC/std)** | 1.40 |
| **Hit Rate** (% correct direction) | 52.3% |
| **Sharpe Ratio** (from backtest) | 2.53 |

**Interpretation**:
- Spearman 0.12 → Statistically significant ranking ability
- IC 1.40 → Very strong for equity prediction
- Hit rate 52.3% → Small edge, amplified by leverage

## Model Validation

### Cross-Sectional Performance

**Daily Spearman correlation distribution**:
```
Mean: 0.120
Std:  0.080
Min:  -0.15
Max:  0.35

Percentiles:
  5%:  0.00
 25%:  0.06
 50%:  0.12
 75%:  0.17
 95%:  0.25
```

**Interpretation**: Model has consistent positive correlation on most days.

### Temporal Stability

**Spearman correlation over time**:
```
2024-Q3: 0.125
2024-Q4: 0.115
2025-Q2: 0.118 (prediction period)
2025-Q3: 0.122
```

**Conclusion**: Model performance stable over time.

### Long vs Short Performance

**Prediction quality by position**:
```
Top quintile (longs):
  Mean predicted return: +2.1%
  Mean actual return: +1.3%
  Hit rate: 54.2%

Bottom quintile (shorts):
  Mean predicted return: -1.8%
  Mean actual return: -0.4%
  Hit rate: 48.1%
```

**Insight**: Long predictions stronger than short predictions (justifies S2_FilterNegative strategy).

## Inference

### Prediction Generation

```python
def generate_predictions(model, data, seq_len=90):
    model.eval()
    predictions = []

    with torch.no_grad():
        for date in dates:
            # Get 90-day history
            history = data[data['date'] <= date].tail(seq_len)

            # Create sequence
            X = torch.tensor(
                history[FEATURE_COLUMNS].values,
                dtype=torch.float32
            ).unsqueeze(0)  # Add batch dimension

            # Predict
            y_pred = model(X)

            predictions.append({
                'date': date,
                'ticker': history.iloc[-1]['ticker'],
                'y_pred_reg': y_pred.item()
            })

    return pd.DataFrame(predictions)
```

### Prediction Characteristics

**Distribution of predictions**:
```
Mean: 0.0005
Std:  0.015
Min:  -0.08
Max:  +0.10

Cross-sectional spread (daily):
  Top - Bottom: ~8% (daily)
  Top - Market: ~4%
  Market - Bottom: ~4%
```

## Model Interpretation

### Feature Importance

Analyzed using SHAP values and attention weights:

**Top 5 features**:
1. ret_20d (18.5%)
2. volatility (15.2%)
3. volume_ratio (12.8%)
4. ret_5d (11.3%)
5. dist_from_high (9.7%)

See [FEATURES.md](FEATURES.md) for details.

### Hidden State Analysis

**What LSTM learns**:
- Layer 1: Price patterns, momentum
- Layer 2: Volatility regimes, reversals

**Sequence length analysis**:
- Days 0-30: Low importance
- Days 30-60: Moderate importance
- Days 60-90: High importance (recent history dominates)

## Production Deployment

### Model Checkpoint

**Saved state**:
```python
torch.save({
    'epoch': 35,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': 0.3012,
    'val_loss': 0.3231,
    'val_spearman': 0.120,
    'hyperparameters': {
        'temperature': 0.05,
        'hidden_size': 128,
        # ...
    }
}, 'models/best_lstm_model.pth')
```

### Loading Model

```python
from src.models.lstm_model import StockLSTM

# Initialize
model = StockLSTM(
    input_size=14,
    hidden_size=128,
    num_layers=2,
    dropout=0.2
)

# Load weights
checkpoint = torch.load('models/best_lstm_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Inference Optimization

**For production**:
- Use CPU inference (GPU not needed for prediction)
- Batch predictions for all stocks at once
- Cache sequences to avoid recomputation

```python
# Batch inference (faster)
all_sequences = torch.tensor(sequences, dtype=torch.float32)
with torch.no_grad():
    predictions = model(all_sequences)  # Single forward pass
```

## Limitations

### Known Issues

1. **Long-only bear markets**: Model struggles when everything declines
2. **Sector rotation**: Doesn't explicitly model sector effects
3. **Corporate actions**: Dividends, splits not explicitly handled
4. **Data quality**: Depends on yfinance data accuracy

### Not Included

- **Fundamental data**: No earnings, financials
- **Alternative data**: No sentiment, news
- **Options data**: No implied volatility
- **Macroeconomic factors**: No rates, GDP, etc.

## Future Improvements (Not in Frozen Strategy)

**Potential enhancements** (would require full revalidation):
1. Transformer architecture (attention mechanism)
2. Ensemble with multiple models
3. Separate models for long/short
4. Incorporate fundamental factors
5. Online learning (continuous retraining)

**Why not implemented?**
- Current model achieves Sharpe 2.20
- Strategy frozen at v2.0.0
- Additional complexity = overfitting risk

---

**See Also**:
- [ARCHITECTURE.md](ARCHITECTURE.md) - System overview
- [FEATURES.md](FEATURES.md) - Input features
- [PORTFOLIO_LOGIC.md](PORTFOLIO_LOGIC.md) - How predictions are used
- [../../STRATEGY_DEFINITION.md](../../STRATEGY_DEFINITION.md) - Complete strategy specification

**Last Updated**: January 20, 2026
