# Stock Data Prediction Project

A comprehensive machine learning project for downloading US stock data, storing it in Qdrant vector database, and training PyTorch LSTM models (both regression and classification) with GPU support.

## Features

- **Data Collection**: Download historical stock data from Polygon.io with intelligent rate limiting
- **Feature Engineering**: Calculate 20+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- **Vector Database**: Store embeddings in Qdrant for similarity search
- **Temporal Split**: Train on data until 2024, test on 2025 data (no data leakage)
- **Dual Models**:
  - Regression: Predict next day's closing price
  - Classification: Predict direction (UP/DOWN/NEUTRAL)
- **GPU Support**: Full PyTorch training with CUDA acceleration
- **Comprehensive Metrics**: RMSE, MAE, directional accuracy, confusion matrix, and more

## Project Structure

```
stock-data/
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   ├── raw/                     # Raw downloaded data
│   ├── processed/               # Preprocessed data (train/val/test)
│   └── embeddings/              # Generated embeddings
├── models/
│   └── checkpoints/             # Saved model checkpoints
├── logs/                        # Application logs
├── src/
│   ├── data/
│   │   ├── downloader.py        # Polygon.io data downloader
│   │   └── preprocessor.py      # Feature engineering & preprocessing
│   ├── database/
│   │   ├── qdrant_client.py     # Qdrant vector database client
│   │   └── embeddings.py        # Embedding generation
│   ├── models/
│   │   ├── lstm_model.py        # LSTM architectures
│   │   ├── dataset.py           # PyTorch datasets
│   │   └── trainer.py           # Training pipeline
│   └── utils/
│       ├── config_loader.py     # Configuration management
│       ├── logger.py            # Logging utilities
│       ├── metrics.py           # Evaluation metrics
│       └── gpu_check.py         # GPU verification
├── main.py                      # Main orchestration script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Prerequisites

1. **Python 3.8+** (Python 3.13 recommended)
2. **CUDA-capable GPU** (recommended)
   - NVIDIA GPU with CUDA support
   - CUDA Toolkit 12.1+ installed
   - Note: RTX 50-series GPUs work but may show compatibility warnings
3. **Docker** (for Qdrant)
4. **Polygon.io API Key**
   - Sign up at https://polygon.io/
   - Free tier: 5 API calls/minute
5. **Windows Long Path Support** (Windows only)
   - Required for pip installations
   - See installation section for setup instructions

## Installation

### 1. Enable Windows Long Path Support (Windows Only)

**Required before installation to avoid pip errors.**

Run PowerShell as Administrator:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Then restart your computer.

Alternatively, use Registry Editor:
1. Press `Win + R`, type `regedit`, press Enter
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Set `LongPathsEnabled` to `1`
4. Restart your computer

### 2. Clone and Setup Environment

```bash
# Navigate to project directory
cd c:\Users\luixj\AI\stock-data

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell (if execution policy allows):
venv\Scripts\Activate.ps1
# Windows Command Prompt or PowerShell (recommended):
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note for PowerShell users**: If you get an execution policy error, use `activate.bat` instead of `Activate.ps1`, or run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install PyTorch with CUDA Support

**Check your CUDA version first:**
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

**For CPU only (not recommended):**
```bash
pip3 install torch torchvision torchaudio
```

**Note for RTX 5090 Users**:
- RTX 5090 has compute capability sm_120, which is fully supported by PyTorch 2.7.0+ with CUDA 12.8
- **IMPORTANT**: You MUST use the cu128 index (NOT cu124 or cu118)
- If you see error "CUDA error: no kernel image is available", you're using the wrong PyTorch build
- Verify with: `python main.py check-gpu` - should show "PyTorch Version: 2.7.0+cu128" or higher

### 4. Start Qdrant Database

```bash
# Pull Qdrant image
docker pull qdrant/qdrant

# Run Qdrant container (Windows PowerShell - use backticks for line continuation)
docker run -d -p 6333:6333 -p 6334:6334 `
    -v ${PWD}/qdrant_storage:/qdrant/storage `
    qdrant/qdrant

# Or on Windows Command Prompt (single line):
docker run -d -p 6333:6333 -p 6334:6334 -v %cd%/qdrant_storage:/qdrant/storage qdrant/qdrant

# Linux/Mac:
docker run -d -p 6333:6333 -p 6334:6334 \
    -v ./qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

Verify Qdrant is running:
- Open browser: http://localhost:6333/dashboard
- Or check: `docker ps`

### 5. Configure API Keys

```bash
# Copy example env file
# Windows:
copy .env.example .env
# Linux/Mac:
cp .env.example .env

# Edit .env and add your Polygon.io API key
# POLYGON_API_KEY=your_api_key_here

# For local Qdrant (default):
# QDRANT_URL=http://localhost:6333
# No QDRANT_API_KEY needed for local Docker instances
```

**Note**: Local Qdrant instances don't require an API key. Only set `QDRANT_API_KEY` if using Qdrant Cloud or if you've enabled authentication on your Docker container.

### 6. Verify GPU Setup

```bash
python main.py check-gpu
```

This will show:
- CUDA availability
- GPU information (model, memory, compute capability)
- PyTorch and CUDA versions

## Configuration

Edit `config/config.yaml` to customize:

- **Stock tickers**: Add/remove tickers to download
- **Date ranges**: Adjust training/validation/test periods
- **Model architecture**: LSTM layers, hidden size, dropout
- **Training parameters**: Batch size, learning rate, epochs
- **Technical indicators**: Add/remove features

## Usage

### Step-by-Step Workflow

#### 1. Download Stock Data

```bash
python main.py download
```

This will:
- Download data for all tickers in config
- Respect Polygon.io rate limits (5 calls/min)
- Cache data locally
- Save to `data/raw/stocks_raw.parquet`

**Expected time**: ~10 minutes for 20 stocks × 5 years

#### 2. Preprocess Data

```bash
python main.py preprocess
```

This will:
- Calculate 20+ technical indicators
- Create regression and classification labels
- Split data temporally:
  - Train: until Sept 30, 2024
  - Validation: Oct-Dec 2024
  - Test: 2025
- Normalize features
- Save to `data/processed/`

#### 3. Upload to Qdrant

```bash
python main.py upload
```

This will:
- Generate embeddings from features
- Create Qdrant collection
- Upload train/val/test embeddings
- Store metadata (ticker, date, price)

#### 4. Train Regression Model

```bash
python main.py train-reg
```

This will:
- Create LSTM regression model
- Train on GPU (if available)
- Use early stopping
- Save best checkpoint to `models/checkpoints/lstm_regression_best.pth`

**Expected time**: 30-60 minutes (depends on GPU)

#### 5. Train Classification Model

```bash
python main.py train-clf
```

This will:
- Create LSTM classification model
- Train to predict UP/DOWN/NEUTRAL
- Save best checkpoint

#### 6. Evaluate Models

```bash
# Evaluate regression model on 2025 test data
python main.py eval-reg

# Evaluate classification model
python main.py eval-clf
```

This will:
- Load best checkpoint
- Make predictions on 2025 test set
- Calculate metrics:
  - Regression: RMSE, MAE, MAPE, R², directional accuracy
  - Classification: Accuracy, precision, recall, F1, confusion matrix
- Save predictions to `data/processed/`

### Run Full Pipeline

```bash
python main.py all
```

Runs all steps in sequence: download → preprocess → upload → train both models → evaluate both models

## Model Architecture

### LSTM Regression Model

```
Input: (batch_size, 60, num_features)
  ↓
LSTM (2 layers, 128 hidden units, dropout=0.3)
  ↓
Fully Connected (128 → 64 → 1)
  ↓
Output: Predicted price
```

### LSTM Classification Model

```
Input: (batch_size, 60, num_features)
  ↓
LSTM (2 layers, 128 hidden units, dropout=0.3)
  ↓
Fully Connected (128 → 64 → 3)
  ↓
Softmax
  ↓
Output: Class probabilities [UP, DOWN, NEUTRAL]
```

## Data Split Strategy

**Temporal Split (No Data Leakage)**:

- **Training**: 2020-01-01 to 2024-09-30
- **Validation**: 2024-10-01 to 2024-12-31
- **Test**: 2025-01-01 onwards

This ensures:
- Model never sees future data during training
- Realistic evaluation on unseen 2025 data
- Mimics real-world trading scenario

## Technical Indicators

The preprocessor calculates:

1. **Moving Averages**: SMA_10, SMA_20, SMA_50, EMA_12, EMA_26
2. **Momentum**: RSI_14, MACD, MACD_signal, MACD_diff
3. **Volatility**: Bollinger Bands (upper/middle/lower), ATR_14
4. **Volume**: Volume_SMA_20, OBV (On-Balance Volume)
5. **Price Action**: Returns, momentum, volatility measures

## Qdrant Vector Database

### Purpose
- Store feature embeddings for similarity search
- Find similar stock patterns across history
- Enable pattern-based analysis

### Collection Schema
```python
{
  "vector": [64 dimensions],
  "payload": {
    "ticker": "AAPL",
    "date": "2024-09-30",
    "close": 150.25,
    "target_class": 1,  # UP
    "dataset": "train"
  }
}
```

### Similarity Search Example

```python
from src.database.qdrant_client import StockQdrantClient
from src.database.embeddings import EmbeddingGenerator

# Initialize clients
qdrant = StockQdrantClient(config)
embedder = EmbeddingGenerator(config)

# Generate embedding for current pattern
current_embedding = embedder.simple_embedding(current_features)

# Search for similar patterns
similar = qdrant.search_similar(
    query_vector=current_embedding,
    limit=10,
    filter_dict={'ticker': 'AAPL'}  # Optional filter
)

# Returns 10 most similar historical patterns
```

## Training Configuration

Default settings in `config/config.yaml`:

```yaml
model:
  architecture:
    hidden_size: 128
    num_layers: 2
    dropout: 0.3

  training:
    batch_size: 32
    epochs: 100
    learning_rate: 0.001

    early_stopping:
      patience: 15
      min_delta: 0.0001
```

## GPU Optimization

The project is optimized for GPU training:

- **DataLoader**: Uses `pin_memory=True` for faster CPU→GPU transfer
- **Mixed Precision**: Can be enabled for faster training (future enhancement)
- **Batch Processing**: Configurable batch size
- **Device Handling**: Automatic GPU detection

## Evaluation Metrics

### Regression Metrics

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination
- **Directional Accuracy**: Did we predict the right direction?

### Classification Metrics

- **Accuracy**: Overall accuracy
- **Precision/Recall/F1**: Per class and macro average
- **Confusion Matrix**: Breakdown of predictions
- **Directional Accuracy**: Practical trading metric

## Troubleshooting

### Windows Long Path Error

If you see errors about file paths being too long during `pip install`:

```powershell
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Restart your computer after enabling.

### PowerShell Execution Policy Error

If `venv\Scripts\Activate.ps1` fails:

**Solution 1** (Recommended): Use the batch file instead:
```powershell
venv\Scripts\activate.bat
```

**Solution 2**: Enable PowerShell scripts:
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### GPU Not Detected

```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with correct CUDA version
pip uninstall -y torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### RTX 5090 Runtime Error

If you see error: "CUDA error: no kernel image is available for execution on the device":

**Problem**: RTX 5090 (sm_120) requires PyTorch 2.7.0+ with CUDA 12.8. Other CUDA versions won't work.

**Solution - Install PyTorch with CUDA 12.8**:
```bash
# Uninstall current PyTorch
pip uninstall -y torch torchvision torchaudio

# Install with CUDA 12.8 support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Verify it works**:
```bash
python main.py check-gpu
```

Expected output:
- PyTorch Version: 2.7.0+cu128 or higher (e.g., 2.9.1+cu128)
- CUDA Version: 12.8
- No compatibility warnings

**Then enable GPU in config**:
Edit [config/config.yaml](config/config.yaml):
```yaml
model:
  device: "cuda"  # Make sure this is set to cuda, not cpu
```

### Missing `List` Import Error

If you see `NameError: name 'List' is not defined`:
- This has been fixed in the codebase
- Make sure you have the latest version of [src/database/embeddings.py](src/database/embeddings.py)

### Polygon.io Rate Limit

The downloader automatically handles rate limiting. If you hit limits:
- Use cached data (automatic)
- Upgrade to paid Polygon.io plan
- Reduce number of tickers in config

### Qdrant Connection Error

```bash
# Check if Qdrant is running
docker ps

# Restart Qdrant
docker restart <qdrant_container_id>

# Check logs
docker logs <qdrant_container_id>
```

### Out of Memory

If GPU runs out of memory:
1. Reduce batch size in config: `batch_size: 16`
2. Reduce sequence length: `sequence_length: 30`
3. Reduce model size: `hidden_size: 64`

## Future Enhancements

- [ ] Transformer model architecture
- [ ] Attention mechanism
- [ ] Multi-stock portfolio optimization
- [ ] Real-time prediction API
- [ ] Backtesting framework with trading simulation
- [ ] Web dashboard for monitoring
- [ ] Hyperparameter tuning with Optuna
- [ ] Model ensemble methods

## License

MIT License

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/yourusername/stock-data)
- Documentation: This README

## Acknowledgments

- **Polygon.io**: Stock data provider
- **Qdrant**: Vector database
- **PyTorch**: Deep learning framework
- **TA-Lib**: Technical analysis library

---

**Disclaimer**: This project is for educational and research purposes only. Past performance does not guarantee future results. Always do your own research before making investment decisions.
