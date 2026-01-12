# Project Context & Conversation Summary

**Date Created**: 2026-01-01
**Project**: Stock Data Prediction with PyTorch & Qdrant
**Location**: `c:\Users\luixj\AI\stock-data\`

## What Was Built

A complete ML pipeline for stock prediction with the following specifications:

### Requirements (From User)
1. Python project in `/work/ai/stock` (adjusted to `c:\Users\luixj\AI\stock-data\`)
2. Download US stock data
3. Save data in Qdrant DB (running in Docker locally)
4. Use PyTorch with GPU for training
5. Create models for prediction
6. Use data until end of 2024 for training
7. Use 2025 data for testing/verification
8. Use Polygon.io free tier (5 API calls/min) for data

### Key Decisions Made

1. **Data Source**: Polygon.io free tier (chose over yfinance for better data quality)
2. **Storage**: Hybrid approach
   - Raw data: Parquet files (fast sequential reads for training)
   - Embeddings: Qdrant (for similarity search)
3. **Data Split**: Temporal split to avoid data leakage
   - Train: 2020-01-01 to 2024-09-30
   - Validation: 2024-10-01 to 2024-12-31
   - Test: 2025-01-01 onwards
4. **Models**: Both approaches implemented
   - Regression: Predict next day closing price
   - Classification: Predict direction (UP/DOWN/NEUTRAL)

### Architecture

```
Data Flow:
Polygon.io → Raw Parquet → Feature Engineering → Train/Val/Test Split
                                ↓
                          Embeddings → Qdrant
                                ↓
                    PyTorch LSTM Models (GPU)
                                ↓
                    Predictions & Evaluation
```

### Files Created (21 total)

#### Core
- `main.py` - CLI orchestration script
- `requirements.txt` - Dependencies
- `README.md` - Full documentation
- `QUICKSTART.md` - Quick start guide
- `config/config.yaml` - Configuration
- `.env.example` - Environment template

#### Source Code (14 modules)
- `src/data/downloader.py` - Polygon.io with rate limiting
- `src/data/preprocessor.py` - Feature engineering
- `src/database/qdrant_client.py` - Vector DB integration
- `src/database/embeddings.py` - Embedding generation
- `src/models/lstm_model.py` - LSTM architectures
- `src/models/dataset.py` - PyTorch datasets
- `src/models/trainer.py` - Training pipeline
- `src/utils/config_loader.py` - Config management
- `src/utils/logger.py` - Logging
- `src/utils/gpu_check.py` - GPU verification
- `src/utils/metrics.py` - Evaluation metrics

### Technical Features

1. **Rate Limiting**: Smart handling of Polygon.io 5 calls/min limit
2. **Caching**: Local caching to avoid re-downloading
3. **GPU Support**: Full CUDA acceleration
4. **Technical Indicators**: 20+ indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
5. **Temporal Split**: Proper train/val/test split by date
6. **Early Stopping**: Prevents overfitting
7. **Checkpointing**: Save best models
8. **Comprehensive Metrics**: RMSE, MAE, accuracy, F1, confusion matrix, etc.

### Current State

✅ Project structure created
✅ All code modules implemented
✅ Configuration files set up
✅ Documentation written
⏳ **Next**: User needs to setup environment and run pipeline

### Setup Commands (For Reference)

```bash
# 1. Setup
cd c:\Users\luixj\AI\stock-data
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Start Qdrant
docker run -d -p 6333:6333 -p 6334:6334 -v ./qdrant_storage:/qdrant/storage:z qdrant/qdrant

# 3. Configure
copy .env.example .env
# Edit .env and add: POLYGON_API_KEY=your_key_here

# 4. Run
python main.py check-gpu
python main.py all
```

### Important Configuration Details

**Stock Tickers** (config.yaml):
- Default: AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, BRK.B, JPM, V, JNJ, WMT, PG, MA, HD, DIS, NFLX, ADBE, CRM, INTC

**Model Defaults**:
- Sequence length: 60 days
- Hidden size: 128
- LSTM layers: 2
- Dropout: 0.3
- Batch size: 32
- Learning rate: 0.001
- Max epochs: 100
- Early stopping patience: 15

**Classification Thresholds**:
- UP: > +0.5% change
- DOWN: < -0.5% change
- NEUTRAL: between -0.5% and +0.5%

### Known Considerations

1. **Polygon.io Free Tier**: 5 calls/min limit
   - Downloading 20 stocks takes ~10 minutes
   - Automatic rate limiting implemented
   - Caching prevents re-downloads

2. **GPU Memory**:
   - Default settings work on 8GB VRAM
   - Reduce batch_size if OOM errors occur

3. **Training Time**:
   - Each model: 30-60 minutes on GPU
   - Both models: 1-2 hours total

### Questions Asked During Development

1. **Model goal?** → Both regression and classification
2. **Data source?** → Polygon.io (over yfinance and Alpha Vantage)
3. **Time granularity?** → Daily data
4. **Qdrant usage?** → Store all data with embeddings
5. **Storage strategy?** → Hybrid (Parquet + Qdrant)

### Future Enhancement Ideas

- Transformer architecture
- Attention mechanisms
- Real-time prediction API
- Backtesting with trading simulation
- Web dashboard
- Multi-stock portfolio optimization
- Hyperparameter tuning with Optuna

## How to Resume This Project

When you come back to this project or start a new conversation:

1. **Provide this context file** to Claude
2. **Mention what you want to do**, e.g.:
   - "I want to add more technical indicators"
   - "I want to improve the model architecture"
   - "I'm getting an error when running X"
   - "I want to add backtesting functionality"

3. **Share relevant files** if making changes:
   - The file you're modifying
   - Error messages
   - Configuration settings

## Current Issues / TODOs

- [ ] User needs to get Polygon.io API key
- [ ] User needs to setup virtual environment
- [ ] User needs to start Qdrant Docker container
- [ ] User needs to install PyTorch with CUDA
- [ ] User needs to run the pipeline for first time

## Contact / Reference

- Project location: `c:\Users\luixj\AI\stock-data\`
- Polygon.io signup: https://polygon.io/
- Qdrant docs: https://qdrant.tech/documentation/
- PyTorch CUDA install: https://pytorch.org/get-started/locally/

---

**Note**: This file serves as a summary of our conversation. When resuming work on this project, share this file with Claude to quickly restore context.
