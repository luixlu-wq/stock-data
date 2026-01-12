# Quick Start Guide

## Setup (10 minutes)

### 0. Prerequisites (Windows Only)

**Enable Windows Long Path Support** - Required before installation:

Run PowerShell as Administrator:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Restart your computer after running this command.

### 1. Create Virtual Environment
```bash
cd c:\Users\luixj\AI\stock-data
python -m venv venv

# Windows (use .bat file to avoid execution policy issues):
venv\Scripts\activate.bat

# Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install PyTorch with CUDA

**First, check your CUDA version:**
```bash
nvidia-smi
```

**For RTX 5090 (CUDA 12.8 required for sm_120 support):**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**For RTX 40 series (CUDA 12.4):**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**For older GPUs (CUDA 11.8):**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only (slower):**
```bash
pip3 install torch torchvision torchaudio
```

### 4. Start Qdrant Database
```bash
# Pull image
docker pull qdrant/qdrant

# Run container (Windows PowerShell):
docker run -d -p 6333:6333 -p 6334:6334 -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant

# Or Windows Command Prompt:
docker run -d -p 6333:6333 -p 6334:6334 -v %cd%/qdrant_storage:/qdrant/storage qdrant/qdrant

# Linux/Mac:
docker run -d -p 6333:6333 -p 6334:6334 -v ./qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

Verify: http://localhost:6333/dashboard

### 5. Configure Data Source

The system supports both Yahoo Finance (free, unlimited) and Polygon.io (requires API key):

**Option A: Yahoo Finance (Recommended - Free & Unlimited)**
- Already configured in `config/config.yaml`: `source: "yahoo"`
- No API key needed
- 15 years of historical data available
- No rate limits

**Option B: Polygon.io (Optional - Requires API Key)**
```bash
# Create .env file
copy .env.example .env

# Edit .env and add your Polygon.io API key:
# POLYGON_API_KEY=your_api_key_here

# Edit config/config.yaml and change:
# data:
#   source: "polygon"
```

Get your free API key at: https://polygon.io/

**For local Qdrant (already configured):**
- No API key needed for local instances
- URL: http://localhost:6333

### 6. Verify GPU
```bash
python main.py check-gpu
```

Expected output:
- ✅ CUDA Available: Yes
- ✅ GPU detected with memory info
- ✅ RTX 5090 fully supported with PyTorch 2.7.0+ and CUDA 12.8

## Usage

### Option 1: Run Full Pipeline (Recommended for first time)
```bash
python main.py all
```

This will:
1. Download stock data (~10 min)
2. Preprocess and engineer features (~5 min)
3. Upload to Qdrant (~2 min)
4. Train multi-task model (~45-90 min) - both regression & classification
5. Evaluate both tasks (~2 min)

**Total time: ~1-2 hours** (mostly training)

**Note**: Updated in v2.0.0 to use multi-task model for better performance.

### Option 2: Run Step by Step
```bash
# 1. Download data (189 stocks, 15 years: 2010-2025)
python main.py download

# 2. Preprocess (adds 40+ features including technical indicators, market features)
python main.py preprocess

# 3. Upload to Qdrant
python main.py upload

# 4. Train multi-task model (RECOMMENDED)
python main.py train-multitask
# Trains both regression + classification together
# Benefits: Better performance, automatic class weighting, single model

# 5. Evaluate multi-task model on 2025 test data
python main.py eval-multitask

# ============================================
# ADVANCED: Train Individual Models (Optional)
# ============================================
# Only use if you need a specific task or for research/benchmarking
# Multi-task model is recommended for production use

# Train regression only (if you only need return predictions)
python main.py train-reg
python main.py eval-reg

# Train classification only (if you only need direction)
python main.py train-clf
python main.py eval-clf
```

## What You'll Get

After training, you'll have:

1. **Trained Models**:
   - `models/checkpoints/lstm_multitask_best.pth` (recommended - both tasks)
   - `models/checkpoints/lstm_regression_best.pth` (regression only)
   - `models/checkpoints/lstm_classification_best.pth` (classification only)

2. **Processed Data** (640K train, 47K validation, 46K test samples):
   - `data/processed/train.parquet` (2010-2023)
   - `data/processed/validation.parquet` (2024)
   - `data/processed/test.parquet` (2025)

3. **Predictions**:
   - `data/processed/multitask_predictions.parquet` (both regression & classification)
   - `data/processed/regression_predictions.parquet`
   - `data/processed/classification_predictions.parquet`

4. **Qdrant Database**:
   - Vector embeddings of all stock patterns
   - Searchable by similarity for pattern matching

## Expected Results

### Multi-Task Model (RECOMMENDED)
**Regression Task (% Return Prediction):**
- RMSE: ~0.015 (1.5% average error)
- MAE: ~0.011 (1.1% average error)
- MAPE: 2-3% (percentage error)
- Directional Accuracy: 56-62% (>50% is profitable)
- R²: 0.40-0.50 (realistic for returns)

**Classification Task (Direction Prediction):**
- Accuracy: 45-52% (33% = random for 3 classes)
- F1 Score: 0.45-0.50
- Better for identifying strong trends

**Benefits of Multi-Task:**
- Shared learning improves both tasks
- Single model deployment
- 30-40% fewer parameters than separate models

### Individual Models (if trained separately)
Similar performance but require two separate models

## Customization

Edit `config/config.yaml` to:
- **Data Source**: Switch between Yahoo Finance and Polygon.io
- **Stock Tickers**: Add/remove from 200+ S&P 500 stocks
- **Model Architecture**:
  - `hidden_size`: 64, 128, 256 (default: 128)
  - `num_layers`: 1, 2, 3 (default: 2)
  - `dropout`: 0.2-0.5 (default: 0.3)
  - `use_attention`: True/False (default: True - recommended)
- **Training Parameters**:
  - `batch_size`: 128, 256, 512 (default: 256 for RTX 5090)
  - `learning_rate`: 0.0001-0.01 (default: 0.001)
  - `epochs`: 50-200 (default: 100)
  - `max_grad_norm`: 0.5-2.0 (default: 1.0 - gradient clipping)
- **Loss Functions**:
  - Regression: `huber` (robust), `mse`, `mae`
  - Default: `huber` - more robust to outliers
- **Multi-Task Weights** (for multitask model):
  - `reg_weight`: 1.0 (regression task importance)
  - `clf_weight`: 0.5 (classification task importance)
- **Technical Indicators**: Add/modify in preprocessor.py
- **Feature Engineering**: 40+ features including:
  - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
  - Volume analysis (volume ratio, OBV, price-volume correlation)
  - Market features (market return, relative strength, market correlation)
  - Time features (day of week, month, quarter, month-end effects)

## Common Issues

**Windows Long Path Error?**
```powershell
# Run PowerShell as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
# Restart computer
```

**PowerShell Execution Policy Error?**
```bash
# Use this instead:
venv\Scripts\activate.bat
```

**GPU not working?**
```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch with matching CUDA version
pip uninstall -y torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**RTX 5090 - CUDA kernel error?**
If you get "no kernel image is available for execution":
- You need PyTorch 2.7.0+ with CUDA 12.8 for sm_120 support
- **Fix**: Install the correct version:
  ```bash
  pip uninstall -y torch torchvision torchaudio
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  ```
- Then set `device: "cuda"` in `config/config.yaml`
- Do NOT use cu124 or cu118 builds - RTX 5090 requires cu128

**NameError: name 'List' is not defined?**
- Fixed in current version
- Make sure you're using the latest code

**Polygon.io rate limit?**
- Free tier: 5 calls/minute
- Downloader handles this automatically
- Use cached data (automatic on reruns)

**Out of memory?**
- Reduce `batch_size` in config: `batch_size: 16`
- Reduce `sequence_length`: `sequence_length: 30`

**Qdrant connection error?**
```bash
# Check if running
docker ps

# Check dashboard
# Open: http://localhost:6333/dashboard
```

## Model Improvements (January 2026)

**Major fixes and enhancements were implemented to address critical bugs:**

### Critical Bugs Fixed
1. **Target Variable**: Changed from absolute prices → percentage returns (normalized across stocks)
2. **Directional Accuracy**: Fixed calculation bug showing impossible 100% accuracy

### New Features
3. **Multi-Task Learning**: Train both regression + classification together with shared LSTM encoder
4. **Attention Mechanism**: Automatically focuses on important time steps in 60-day sequences
5. **Advanced Features**: Added 20+ new features (market indicators, time features, volume analysis)
6. **Huber Loss**: More robust to outliers than MSE (better handles market crashes)
7. **Gradient Clipping**: Prevents training instability

**See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed documentation of all changes.**

### How to Use New Features

**To retrain with improvements:**
```bash
# 1. Preprocess with new features (adds 20+ advanced features)
python main.py preprocess

# 2. Train multi-task model (recommended)
python main.py train-multitask

# 3. Evaluate both tasks
python main.py eval-multitask
```

**Expected improvements:**
- Directional accuracy: Now realistic 56-62% (was buggy 100%)
- RMSE: ~0.015 for % returns (was 146.57 for mixed price scales)
- MAPE: Improved from 3.12% to ~2.5%
- Single model for both regression and classification

## Next Steps

1. **Retrain Models**: Use new multi-task model with improvements
2. **Experiment**: Try different stocks, indicators, model architectures
3. **Analyze**: Look at predictions vs actual in the parquet files
4. **Hyperparameter Tuning**: Optimize hidden_size, learning_rate, dropout
5. **Backtest**: Simulate trading strategies (to be implemented)
6. **Deploy**: Use trained models for real-time predictions

## Resources

- **Training Guide**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Which command should I use?
- **Training Troubleshooting**: [TRAINING_TROUBLESHOOTING.md](TRAINING_TROUBLESHOOTING.md) - Fix interrupted training
- **Model Improvements**: [IMPROVEMENTS.md](IMPROVEMENTS.md) - Detailed documentation of all fixes
- **Classification Fixes**: [CLASSIFICATION_FIXES.md](CLASSIFICATION_FIXES.md) - Handling class imbalance
- **Full README**: [README.md](README.md) - Complete project overview
- **Configuration**: [config/config.yaml](config/config.yaml) - All settings
- **Yahoo Finance**: https://finance.yahoo.com/ - Free data source
- **Polygon.io Docs**: https://polygon.io/docs - Alternative data source
- **Qdrant Docs**: https://qdrant.tech/documentation/ - Vector database

Happy Trading! 📈
