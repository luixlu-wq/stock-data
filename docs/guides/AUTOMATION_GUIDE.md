# Automated Paper Trading with Qdrant

Complete automation system for daily paper trading with vector database storage.

---

## Overview

This automation system:
1. **Runs daily paper trading** automatically at 4:15 PM EST
2. **Saves stock recommendations** to Qdrant vector database
3. **Stores trading results** with full specifications
4. **Tracks performance metrics** over time
5. **Enables similarity search** for stocks and patterns

---

## Prerequisites

### 1. Install Qdrant

**Option A: Docker** (Recommended)
```bash
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

**Option B: Standalone**
Download from: https://github.com/qdrant/qdrant/releases

### 2. Install Python Dependencies

```bash
pip install qdrant-client
```

Already installed ✅

---

## Setup Instructions

### Step 1: Start Qdrant Server

**Using Docker**:
```bash
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant-paper-trading qdrant/qdrant
```

**Verify Qdrant is running**:
```bash
curl http://localhost:6333/
```

Or open in browser: http://localhost:6333/dashboard

### Step 2: Setup Windows Task Scheduler

**Option A: PowerShell** (Recommended)

1. Open PowerShell as **Administrator**
2. Navigate to project directory:
   ```powershell
   cd c:\Users\luixj\AI\stock-data
   ```
3. Run setup script:
   ```powershell
   .\scripts\automation\setup_daily_task.ps1
   ```

**Option B: Batch File**

1. Open Command Prompt as **Administrator**
2. Navigate to project directory
3. Run:
   ```cmd
   scripts\automation\setup_daily_task.bat
   ```

**Option C: Manual Setup**

1. Open Task Scheduler (taskschd.msc)
2. Create Basic Task
3. Name: "DailyPaperTrading"
4. Trigger: Daily at 4:15 PM
5. Action: Start a program
   - Program: `c:\Users\luixj\AI\stock-data\venv\Scripts\python.exe`
   - Arguments: `scripts\automation\daily_paper_trading_qdrant.py`
   - Start in: `c:\Users\luixj\AI\stock-data`

### Step 3: Test the Automation

**Run manually**:
```bash
cd c:\Users\luixj\AI\stock-data
venv\Scripts\activate
python scripts/automation/daily_paper_trading_qdrant.py
```

**Or via Task Scheduler**:
```powershell
Start-ScheduledTask -TaskName "DailyPaperTrading"
```

---

## What Gets Saved to Qdrant

### Collection 1: Stock Recommendations
- **Ticker symbol**
- **Position** (LONG/SHORT)
- **Weight** (portfolio allocation %)
- **Prediction** (expected return)
- **Actual return** (realized return)
- **Close price**
- **Date** (historical + calendar)
- **Saved timestamp**

**Vector embedding**: [weight, prediction, actual_return, price, position_flag, ...]

### Collection 2: Trading Results
- **Daily PnL** (net and vol-scaled)
- **Turnover**
- **Vol scale factor**
- **Kill switch status**
- **Long/Short PnL**
- **Number of positions**
- **Date** (historical + calendar)

**Vector embedding**: [pnl_net, pnl_scaled, turnover, vol_scale, ks_flag]

### Collection 3: Performance Metrics
- **Sharpe ratio** (raw and scaled)
- **Annual return**
- **Volatility**
- **Max drawdown**
- **Long/Short Sharpe**
- **Avg turnover**
- **Kill switch %**
- **Deployment ready flag**

**Vector embedding**: [sharpe, return, vol, maxdd, long_sharpe, short_sharpe, turnover, ks_pct]

---

## Querying the Database

### View Latest Recommendations
```bash
python scripts/automation/query_qdrant.py --type recommendations --limit 10
```

### View Trading Results
```bash
python scripts/automation/query_qdrant.py --type results --limit 5
```

### View Performance Metrics
```bash
python scripts/automation/query_qdrant.py --type performance
```

### View Recommendations for Specific Date
```bash
python scripts/automation/query_qdrant.py --date 2025-04-01
```

### Search Similar Stocks
```bash
python scripts/automation/query_qdrant.py --search AAPL
```

### Database Statistics
```bash
python scripts/automation/query_qdrant.py --type stats
```

---

## Daily Workflow (Automated)

**Time**: 4:15 PM EST (automatically)

1. **Task Scheduler triggers** the automation script
2. **Script loads** next historical date from progress tracker
3. **Runs paper trading** for that date
4. **Extracts stock recommendations** (38 longs, 15-38 shorts)
5. **Saves to Qdrant**:
   - All stock recommendations with specifications
   - Daily trading results
   - Updated performance metrics
6. **Updates progress** tracker
7. **Logs results** to `logs/automation/`

**No manual intervention required!**

---

## Monitoring

### Check Task Status
```powershell
Get-ScheduledTask -TaskName "DailyPaperTrading"
```

### View Task History
```powershell
Get-ScheduledTaskInfo -TaskName "DailyPaperTrading"
```

### Check Logs
```bash
# View today's automation log
cat logs/automation/daily_automation_$(date +%Y%m%d).log

# Or on Windows
type logs\automation\daily_automation_%date:~-4,4%%date:~-10,2%%date:~-7,2%.log
```

### Check Qdrant Dashboard
Open browser: http://localhost:6333/dashboard

---

## Stopping/Starting Automation

### Disable Daily Task (Temporarily)
```powershell
Disable-ScheduledTask -TaskName "DailyPaperTrading"
```

### Enable Daily Task
```powershell
Enable-ScheduledTask -TaskName "DailyPaperTrading"
```

### Delete Daily Task (Permanently)
```powershell
Unregister-ScheduledTask -TaskName "DailyPaperTrading" -Confirm:$false
```

### Stop Qdrant Server
```bash
docker stop qdrant-paper-trading
```

### Start Qdrant Server
```bash
docker start qdrant-paper-trading
```

---

## Troubleshooting

### Task Not Running?

1. **Check if task exists**:
   ```powershell
   Get-ScheduledTask -TaskName "DailyPaperTrading"
   ```

2. **Check last run result**:
   ```powershell
   Get-ScheduledTaskInfo -TaskName "DailyPaperTrading"
   ```

3. **Check logs**:
   ```bash
   ls logs/automation/
   ```

4. **Run manually to see errors**:
   ```bash
   python scripts/automation/daily_paper_trading_qdrant.py
   ```

### Qdrant Connection Failed?

1. **Check if Qdrant is running**:
   ```bash
   curl http://localhost:6333/
   ```

2. **Start Qdrant**:
   ```bash
   docker start qdrant-paper-trading
   # Or start new instance:
   docker run -d -p 6333:6333 qdrant/qdrant
   ```

3. **Check Qdrant logs**:
   ```bash
   docker logs qdrant-paper-trading
   ```

### Python Environment Issues?

1. **Verify venv is activated** in the task
2. **Check Python path** in task settings
3. **Install missing dependencies**:
   ```bash
   pip install qdrant-client
   ```

---

## Advanced Usage

### Custom Schedule

Edit the task to run at different times:

```powershell
$Trigger = New-ScheduledTaskTrigger -Daily -At "3:00 PM"
Set-ScheduledTask -TaskName "DailyPaperTrading" -Trigger $Trigger
```

### Run on Weekdays Only

```powershell
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "4:15 PM"
Set-ScheduledTask -TaskName "DailyPaperTrading" -Trigger $Trigger
```

### Multiple Instances

To run multiple strategies in parallel, create separate tasks:

```powershell
# Copy the script and modify for strategy variations
# Create task with different name
Register-ScheduledTask -TaskName "DailyPaperTrading_Strategy2" ...
```

---

## Data Backup

### Backup Qdrant Data

```bash
# Stop Qdrant
docker stop qdrant-paper-trading

# Backup storage
cp -r qdrant_storage qdrant_storage_backup_$(date +%Y%m%d)

# Restart Qdrant
docker start qdrant-paper-trading
```

### Export Data from Qdrant

```python
# Custom export script
from qdrant_client import QdrantClient
import json

client = QdrantClient(host="localhost", port=6333)
results = client.scroll(collection_name="stock_recommendations", limit=10000)

with open('recommendations_backup.json', 'w') as f:
    json.dump([p.payload for p in results[0]], f, indent=2)
```

---

## Performance

### Expected Resource Usage

- **Qdrant Memory**: ~100-200 MB (for 60 days of data)
- **Disk Space**: ~50-100 MB (for 60 days)
- **Daily Execution Time**: ~5-10 seconds
- **Network**: Local only (no internet required after initial setup)

### Scaling

The system can handle:
- **Thousands of stocks** (current: ~189)
- **Years of data** (60+ days is typical for paper trading)
- **Real-time queries** (< 100ms for similarity search)

---

## Next Steps After 60 Days

After completing 60 days of automated paper trading:

1. **Review cumulative metrics**:
   ```bash
   python scripts/automation/query_qdrant.py --type performance
   ```

2. **Analyze all recommendations**:
   ```bash
   python scripts/automation/query_qdrant.py --type recommendations --limit 1000
   ```

3. **If Sharpe > 1.0**:
   - Proceed to Phase 5 (live trading)
   - Keep automation running for comparison

4. **Export final report**:
   - Use query script to extract all data
   - Generate comprehensive performance report

---

## Benefits of Qdrant Integration

1. **Vector Similarity Search**: Find stocks with similar characteristics
2. **Historical Queries**: Fast retrieval of past recommendations
3. **Pattern Detection**: Identify recurring patterns in recommendations
4. **Performance Analytics**: Track metrics over time
5. **Data Persistence**: All data safely stored in database
6. **Scalability**: Handles large datasets efficiently

---

## Summary

✅ **Fully Automated**: Runs daily without intervention
✅ **Data Persistence**: All data saved to Qdrant
✅ **Easy Queries**: Simple scripts to view any data
✅ **Reliable**: Windows Task Scheduler ensures execution
✅ **Scalable**: Can handle years of trading data
✅ **Insightful**: Similarity search reveals patterns

**Your paper trading is now fully automated and professionally tracked!** 🚀

---

## Quick Reference

| Task | Command |
|------|---------|
| Setup automation | `.\scripts\automation\setup_daily_task.ps1` |
| Run manually | `python scripts/automation/daily_paper_trading_qdrant.py` |
| View recommendations | `python scripts/automation/query_qdrant.py --type recommendations` |
| View results | `python scripts/automation/query_qdrant.py --type results` |
| View performance | `python scripts/automation/query_qdrant.py --type performance` |
| Search similar | `python scripts/automation/query_qdrant.py --search AAPL` |
| Start Qdrant | `docker start qdrant-paper-trading` |
| Stop Qdrant | `docker stop qdrant-paper-trading` |
| Check task status | `Get-ScheduledTask -TaskName "DailyPaperTrading"` |
| Disable task | `Disable-ScheduledTask -TaskName "DailyPaperTrading"` |

---

**Everything is set up and ready to run automatically!** 🎯
