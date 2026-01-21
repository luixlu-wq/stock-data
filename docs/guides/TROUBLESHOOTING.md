# Troubleshooting Guide

**Common issues and solutions for the stock trading system**

## Installation Issues

### Problem: `pip install` fails with "No module named 'pip'"`

**Solution:**
```bash
# Reinstall pip
python -m ensurepip --upgrade

# Or download get-pip.py
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

### Problem: Virtual environment activation fails

**Windows PowerShell**:
```powershell
# If you get "execution policy" error
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
venv\Scripts\activate
```

**Linux/Mac**:
```bash
# Make sure you have right permissions
chmod +x venv/bin/activate
source venv/bin/activate
```

### Problem: Package installation fails with "Microsoft Visual C++ required"

**Windows Solution:**
```
Download and install:
Microsoft C++ Build Tools
https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### Problem: PyTorch installation issues

**Solution:**
```bash
# Uninstall existing torch
pip uninstall torch

# Install CPU-only version (smaller, faster for inference)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or install with CUDA if you have NVIDIA GPU
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Data Issues

### Problem: "File not found: phase1_predictions.parquet"

**Check if file exists:**
```bash
ls -l data/processed/phase1_predictions.parquet
```

**If missing**, you need to generate predictions:
```bash
# This requires running the full training pipeline
python scripts/phase1/phase1_runner.py
```

**⚠️ Note**: Training takes significant time and resources. If you just cloned the repo, make sure you pulled LFS files:
```bash
git lfs pull
```

### Problem: "KeyError: 'y_true_reg'" or similar column errors

**Cause**: Predictions file schema mismatch

**Solution:**
```python
import pandas as pd

# Check actual columns
df = pd.read_parquet('data/processed/phase1_predictions.parquet')
print(df.columns.tolist())

# Common column name issues:
# 'y_actual' vs 'y_true_reg'
# 'prediction' vs 'y_pred_reg'
# 'ticker_symbol' vs 'ticker'
```

### Problem: Empty or corrupt parquet file

**Solution:**
```python
import pandas as pd

# Try to read and diagnose
df = pd.read_parquet('data/processed/phase1_predictions.parquet')
print(f"Rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Null values:\n{df.isnull().sum()}")

# If corrupt, regenerate from source
```

## Paper Trading Issues

### Problem: "No data for date YYYY-MM-DD"

**Cause**: Trying to run paper trading for a date outside prediction range (2025-04-01 to 2025-10-31)

**Solution:**
```python
import pandas as pd

# Check available dates
df = pd.read_parquet('data/processed/phase1_predictions.parquet')
available_dates = sorted(df['date'].unique())
print(f"First date: {available_dates[0]}")
print(f"Last date: {available_dates[-1]}")
print(f"Total dates: {len(available_dates)}")

# Reset progress to start from first date
import json
progress_file = 'data/processed/phase4/paper_trading_progress.json'
with open(progress_file, 'w') as f:
    json.dump({
        'start_calendar_date': '2026-01-20',
        'start_historical_date': '2025-04-01',
        'last_historical_date': '2025-04-01',
        'days_completed': 0
    }, f, indent=2)
```

### Problem: Paper trading runs but shows no P&L

**Check**:
1. Are positions being constructed?
2. Are returns available for the date?

**Debug:**
```python
import pandas as pd

df = pd.read_parquet('data/processed/phase1_predictions.parquet')
day = df[df['date'] == '2025-04-01'].copy()

print(f"Stocks on 2025-04-01: {len(day)}")
print(f"Predictions:\n{day['y_pred_reg'].describe()}")
print(f"Actuals:\n{day['y_true_reg'].describe()}")
print(f"Null actuals: {day['y_true_reg'].isnull().sum()}")
```

### Problem: Kill switches always triggering

**Cause**: Insufficient trading history for risk calculations

**Solution:**
Kill switches need historical data:
- 3-sigma loss: Needs 20+ days
- Sharpe < 0: Needs 60+ days
- DD tracking: Needs cumulative returns

**During early paper trading**, kill switches may trigger frequently. This is expected in the first 60 days.

## Automation Issues

### Problem: Qdrant connection failed

**Check if Qdrant is running:**
```bash
# Check Docker containers
docker ps

# Check if port 6333 is in use
netstat -an | grep 6333  # Linux/Mac
netstat -an | findstr 6333  # Windows
```

**Start Qdrant:**
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

**If Docker not installed:**
```bash
# Install Docker Desktop (Windows/Mac)
# Or install Docker Engine (Linux)
https://docs.docker.com/get-docker/
```

### Problem: Windows Task Scheduler task not running

**Debug steps:**

1. **Check task exists:**
   ```powershell
   Get-ScheduledTask -TaskName "DailyPaperTrading"
   ```

2. **Check last run result:**
   ```powershell
   Get-ScheduledTaskInfo -TaskName "DailyPaperTrading"
   ```

3. **Run manually to see errors:**
   ```powershell
   Start-ScheduledTask -TaskName "DailyPaperTrading"
   ```

4. **Check logs:**
   ```bash
   cat logs/automation/daily_automation_YYYYMMDD.log
   ```

**Common issues:**
- Task not running with highest privileges
- Python path incorrect in task
- Working directory not set correctly

**Solution - Recreate task:**
```powershell
# Delete old task
Unregister-ScheduledTask -TaskName "DailyPaperTrading" -Confirm:$false

# Run setup script as Administrator
.\scripts\automation\setup_daily_task.ps1
```

### Problem: "qdrant_client" module not found

**Solution:**
```bash
pip install qdrant-client
```

### Problem: Automation runs but doesn't save to Qdrant

**Check collections:**
```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# List collections
collections = client.get_collections()
print(f"Collections: {collections}")

# Check collection sizes
for coll in ['stock_recommendations', 'trading_results', 'performance_metrics']:
    try:
        info = client.get_collection(coll)
        print(f"{coll}: {info.points_count} points")
    except:
        print(f"{coll}: Does not exist")
```

**Recreate collections if needed:**
```bash
python scripts/automation/daily_paper_trading_qdrant.py
# This will auto-create collections on first run
```

## Model Issues

### Problem: "Model file not found: models/best_lstm_model.pth"

**Check if file exists:**
```bash
ls -l models/best_lstm_model.pth
```

**If missing**, you need to train the model:
```bash
python scripts/phase1/phase1_runner.py
```

### Problem: "LSTM forward pass error" or dimension mismatches

**Cause**: Model architecture mismatch

**Check model architecture:**
```python
import torch
from src.models.lstm_model import StockLSTM

model = StockLSTM(input_size=14, hidden_size=64, num_layers=2)
try:
    model.load_state_dict(torch.load('models/best_lstm_model.pth'))
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Model load failed: {e}")
```

**Solution**: Re-train model with correct architecture parameters.

## Performance Issues

### Problem: Paper trading runs very slowly

**Likely causes:**
1. Reading large parquet file repeatedly
2. Inefficient data filtering
3. Complex calculations

**Solutions:**
```python
# Cache predictions in memory if running multiple days
import pandas as pd

class CachedPredictor:
    def __init__(self):
        self.df = pd.read_parquet('data/processed/phase1_predictions.parquet')

    def get_day(self, date):
        return self.df[self.df['date'] == date].copy()
```

### Problem: Out of memory errors

**Solutions:**
1. Use chunked reading for large files
2. Filter data early to reduce memory usage
3. Use categorical dtypes for ticker symbols
4. Clear unused DataFrames

```python
import pandas as pd

# Read only needed columns
df = pd.read_parquet(
    'data/processed/phase1_predictions.parquet',
    columns=['date', 'ticker', 'y_pred_reg', 'y_true_reg', 'close']
)

# Convert to categorical (saves memory)
df['ticker'] = df['ticker'].astype('category')
```

## Query and Reporting Issues

### Problem: query_qdrant.py shows no results

**Check collections have data:**
```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# Check points count
info = client.get_collection("stock_recommendations")
print(f"Total points: {info.points_count}")

# Try scrolling
results = client.scroll(
    collection_name="stock_recommendations",
    limit=10
)
print(f"Results: {len(results[0])}")
```

**If empty**: Run daily automation at least once to populate data.

### Problem: Unicode errors when generating reports

**Error**: `UnicodeEncodeError: 'charmap' codec can't encode character`

**Solution**: Always specify UTF-8 encoding:
```python
# Wrong
with open('report.txt', 'w') as f:
    f.write(text)

# Correct
with open('report.txt', 'w', encoding='utf-8') as f:
    f.write(text)
```

## Configuration Issues

### Problem: "Config file not found: config/config.yaml"

**Check file exists:**
```bash
ls -l config/config.yaml
```

**If missing**, create from template or use defaults in code.

### Problem: Environment variables not loading

**Check .env file:**
```bash
cat .env
```

**Make sure python-dotenv is installed:**
```bash
pip install python-dotenv
```

**Load in code:**
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('API_KEY', 'default_value')
```

## Getting More Help

### Enable Debug Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Check System Info

```python
import sys
import platform
import torch
import pandas

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"Pandas: {pandas.__version__}")
```

### Still Stuck?

1. Check logs in `logs/` directory
2. Review code comments in relevant scripts
3. Check phase results documentation for context
4. Review [PROJECT_ORGANIZATION.md](../../PROJECT_ORGANIZATION.md) for file locations

---

**Last Updated**: January 20, 2026
