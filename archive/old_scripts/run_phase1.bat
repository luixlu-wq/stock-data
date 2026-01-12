@echo off
echo ======================================================================
echo PHASE 1: RANK LOSS + SIMPLIFIED FEATURES
echo ======================================================================
echo.
echo This script will:
echo   1. Train model with rank loss and 14 simplified features
echo   2. Evaluate on test set
echo   3. Compare against baseline (Phase 0)
echo.
echo Expected time: 30-60 minutes
echo.
echo WARNING: Do NOT interrupt training once it starts!
echo ======================================================================
echo.
pause

echo.
echo ======================================================================
echo STEP 1: TRAINING PHASE 1 MODEL
echo ======================================================================
echo.
python phase1_train.py
if errorlevel 1 (
    echo.
    echo ERROR: Training failed!
    echo Check logs/phase1_training.log for details
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo STEP 2: EVALUATING AND COMPARING
echo ======================================================================
echo.
python phase1_evaluate.py
if errorlevel 1 (
    echo.
    echo ERROR: Evaluation failed!
    echo Check logs/phase1_evaluation.log for details
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo PHASE 1 COMPLETE!
echo ======================================================================
echo.
echo Results saved to:
echo   - data/processed/phase1_predictions.parquet
echo   - data/processed/phase1_backtest_results.parquet
echo   - data/processed/phase1_metrics.json
echo.
echo Check logs for detailed comparison against baseline.
echo.
pause
