# Data Pipeline Overview

## 📥 **Data Flow**

```
Yahoo Finance API
       ↓
Raw Stock Data (7 columns)
       ↓
Feature Engineering
       ↓
Processed Data (46 columns)
       ↓
Normalization
       ↓
Sequence Creation (60-day windows)
       ↓
LSTM Model (41 features)
       ↓
Predictions (1-day ahead returns)
```

---

## 1️⃣ **Raw Data** (7 columns)

**Source**: Yahoo Finance (free, unlimited)
**Tickers**: 189 S&P 500 stocks
**Time Range**: 2010-01-01 to 2025-12-31 (15 years)
**Frequency**: Daily

### Columns:
```python
['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
```

**Sample**:
```
date                        open      high       low     close    volume  ticker
2010-01-04 00:00:00-05:00  19.95    20.10     19.65     19.89   3815561    AAPL
```

---

## 2️⃣ **Feature Engineering** (→ 46 columns)

### **Added Features (39 new columns)**:

#### **Technical Indicators (16)**
```python
# Moving Averages (5)
SMA_10, SMA_20, SMA_50        # Simple moving averages
EMA_12, EMA_26                # Exponential moving averages

# Momentum Indicators (4)
RSI_14                        # Relative Strength Index
MACD, MACD_signal, MACD_diff  # Moving Average Convergence Divergence

# Volatility Indicators (4)
BB_upper, BB_middle, BB_lower # Bollinger Bands
ATR_14                        # Average True Range

# Volume Indicators (2)
Volume_SMA_20                 # Volume moving average
OBV                           # On-Balance Volume
```

#### **Price Momentum & Volatility (4)**
```python
momentum_5         # 5-day price change %
momentum_10        # 10-day price change %
volatility_10      # 10-day rolling std of returns
volatility_20      # 20-day rolling std of returns
```

#### **Market-Wide Features (3)**
```python
market_return         # Daily average return across all stocks
vs_market            # Stock return - market return
market_correlation   # 20-day rolling correlation with market
```

#### **Time-Based Features (7)**
```python
day_of_week          # 0=Monday, 4=Friday
month                # 1-12
quarter              # 1-4
is_month_start       # Boolean (0/1)
is_month_end         # Boolean (0/1)
is_quarter_start     # Boolean (0/1)
is_quarter_end       # Boolean (0/1)
```

#### **Volume & Price Relationships (7)**
```python
returns              # Daily return %
volume_ratio         # Current volume / 20-day avg
volume_change        # Volume % change from previous day
price_volume_trend   # close * volume
price_volume_corr    # 20-day correlation between price & volume
high_low_range       # (high - low) / close
gap                  # (open - prev_close) / prev_close
```

#### **Target Variables (3)**
```python
target_return        # 1-day ahead return % (REGRESSION TARGET)
target_class         # 1=UP, 0=DOWN (CLASSIFICATION TARGET)
target_price         # Next day closing price (reference only)
```

---

## 3️⃣ **Data Split** (Temporal)

**Critical**: Time-based split to avoid data leakage!

```python
TRAIN:      2010-01-01 → 2023-12-31  (14 years, 640,126 rows)
VALIDATION: 2024-01-01 → 2024-12-31  (1 year,   47,628 rows)
TEST:       2025-01-01 → 2025-12-31  (1 year,   46,116 rows)
```

**Why temporal?**
- Can't train on future data!
- Mimics real trading scenario
- Prevents look-ahead bias

---

## 4️⃣ **Normalization**

**Method**: StandardScaler (z-score normalization)
**Formula**: `(x - mean) / std`

```python
# Fit scaler ONLY on training data
scaler.fit(train_features)

# Transform all splits
train_normalized = scaler.transform(train_features)
val_normalized = scaler.transform(val_features)
test_normalized = scaler.transform(test_features)
```

**Features Normalized (41)**:
- All columns EXCEPT: date, ticker, target_return, target_class, target_price

**Target NOT Normalized**:
- `target_return` stays as raw percentage (e.g., 0.015 = 1.5% gain)
- Allows interpretable predictions

---

## 5️⃣ **Sequence Creation**

**Window Size**: 60 days
**Prediction Horizon**: 1 day ahead

```python
# For each stock separately:
for i in range(len(stock_data) - 60):
    # Input: 60 days of historical data
    X[i] = stock_data[i : i+60]  # Shape: (60, 41)

    # Target: Next day return
    y[i] = target_return[i+60]    # Shape: (1,)
```

**Example**:
```
Input:  Days 1-60   (60 days × 41 features)
Output: Day 61      (1-day return: 0.0123 = 1.23%)
```

**Resulting Shapes**:
```python
TRAIN:      X=(628,786, 60, 41),  y=(628,786,)
VALIDATION: X=(36,288, 60, 41),   y=(36,288,)
TEST:       X=(35,532, 60, 41),   y=(35,532,)
```

---

## 6️⃣ **Model Input**

**What the LSTM sees**:
```python
# Batch of sequences
Input shape:  (batch_size, 60, 41)
             ↓
# 60 time steps, each with 41 features:
# - 16 technical indicators
# - 4 momentum/volatility metrics
# - 3 market features
# - 7 time features
# - 7 volume/price features
# - 4 price metrics (open, high, low, close normalized)
             ↓
# LSTM processes sequence
             ↓
# Output: Predicted next-day return
Output shape: (batch_size, 1)
Example:      [0.0123] = predicting +1.23% return tomorrow
```

---

## 7️⃣ **Prediction Interpretation**

### **Regression Output**:
```python
prediction = 0.0123  # Model output
```

**Interpretation**:
- Predicts stock will go **UP 1.23%** tomorrow
- If current price = $100, predicted price = $101.23
- If prediction > 0 → BUY signal
- If prediction < 0 → SELL/SHORT signal

### **Classification Output** (if using multitask):
```python
class_prediction = 1  # Model output
```

**Interpretation**:
- 0 = DOWN (negative return expected)
- 1 = UP (positive return expected)

---

## 📊 **Data Statistics**

### **Training Set**
```
Rows:        640,126 sequences
Stocks:      189 tickers
Date Range:  2010-2023 (14 years)
Target Mean: 0.000711 (0.071% avg daily return)
Target Std:  0.018141 (1.81% volatility)
```

### **Validation Set**
```
Rows:        47,628 sequences
Date Range:  2024 (1 year)
Target Mean: 0.000743
Target Std:  0.018455
```

### **Test Set**
```
Rows:        46,116 sequences
Date Range:  2025 (1 year)
Target Mean: 0.000785
Target Std:  0.019956
```

---

## 🔧 **Key Design Decisions**

### **1. Why 1-Day Prediction Horizon?**
- More predictable than 5-day (less noise)
- Practical for daily trading
- Balances signal vs noise

### **2. Why 60-Day Sequence Length?**
- ~3 months of trading history
- Captures seasonal patterns
- Not too long (avoids overfitting)

### **3. Why Normalize Features but Not Targets?**
- Features: Different scales (price=$100, volume=1M, RSI=0-100)
- Targets: Want interpretable % returns

### **4. Why Separate Stocks in Sequences?**
- Each stock has unique patterns
- Prevents mixing AAPL day 30 with MSFT day 31
- Maintains temporal integrity

---

## 📁 **File Locations**

```
data/
├── raw/
│   └── stocks_raw.parquet          # 7 columns, 744,080 rows
├── processed/
│   ├── train.parquet                # 46 columns, 640,126 rows
│   ├── validation.parquet           # 46 columns, 47,628 rows
│   └── test.parquet                 # 46 columns, 46,116 rows
└── embeddings/
    └── stock_embeddings.parquet     # For Qdrant (optional)
```

---

## 🎯 **Summary**

**Pipeline**: Raw (7) → Features (46) → Normalized (41) → Sequences (60×41) → Predictions (1)

**Total Features Used**: 41 normalized features per time step
**Sequence Length**: 60 days
**Prediction**: Next day's return (%)

**Training Data**: 628K sequences from 189 stocks over 14 years
**Model**: LSTM with 192 hidden units, 2 layers, attention mechanism

This comprehensive feature set gives the model the best chance to learn patterns in stock price movements!
