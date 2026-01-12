@echo off
echo ======================================================================
echo FRESH TRAINING SCRIPT - Multi-Task Model
echo ======================================================================
echo.
echo This script will:
echo   1. Delete the old broken checkpoint
echo   2. Start fresh training (45-90 minutes)
echo   3. Monitor progress
echo.
echo IMPORTANT: Do NOT close this window during training!
echo.
pause

echo.
echo [1/3] Deleting old checkpoint...
if exist models\checkpoints\lstm_multitask_best.pth (
    del models\checkpoints\lstm_multitask_best.pth
    echo ✓ Old checkpoint deleted
) else (
    echo ℹ No old checkpoint found (OK)
)

echo.
echo [2/3] Starting training...
echo ⚠️  DO NOT CLOSE THIS WINDOW!
echo Expected time: 45-90 minutes
echo.
echo You will see:
echo   - Training header with configuration
echo   - Epoch progress (e.g., "Epoch [1/100]...")
echo   - Milestones every 10 epochs
echo   - Completion message when done
echo.
echo If epochs don't appear, training might have failed - check logs\stock_data.log
echo.
echo ======================================================================
echo STARTING TRAINING... (watch for epoch output below)
echo ======================================================================
echo.

python main.py train-multitask

if errorlevel 1 (
    echo.
    echo ======================================================================
    echo ❌ TRAINING FAILED
    echo ======================================================================
    echo Check logs\stock_data.log for errors
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo ✅ TRAINING COMPLETED SUCCESSFULLY!
echo ======================================================================
echo.
echo [3/3] Now evaluate the model...
echo.

python main.py eval-multitask

echo.
echo ======================================================================
echo ✅ ALL DONE!
echo ======================================================================
echo.
echo Trained model saved to: models\checkpoints\lstm_multitask_best.pth
echo Predictions saved to: data\processed\multitask_predictions.parquet
echo.
echo Check the results above. Expected:
echo   - Classification Accuracy: 45-52%%
echo   - Regression RMSE: ~0.015
echo   - All 3 classes should be predicted (DOWN, UP, NEUTRAL)
echo.
pause
