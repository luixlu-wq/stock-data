#!/bin/bash

echo "======================================================================"
echo "FRESH TRAINING SCRIPT - Multi-Task Model"
echo "======================================================================"
echo ""
echo "This script will:"
echo "  1. Delete the old broken checkpoint"
echo "  2. Start fresh training (45-90 minutes)"
echo "  3. Monitor progress"
echo ""
echo "IMPORTANT: Do NOT interrupt this script during training!"
echo ""
read -p "Press Enter to continue..."

echo ""
echo "[1/3] Deleting old checkpoint..."
if [ -f models/checkpoints/lstm_multitask_best.pth ]; then
    rm models/checkpoints/lstm_multitask_best.pth
    echo "✓ Old checkpoint deleted"
else
    echo "ℹ No old checkpoint found (OK)"
fi

echo ""
echo "[2/3] Starting training..."
echo "⚠️  DO NOT INTERRUPT!"
echo "Expected time: 45-90 minutes"
echo ""
echo "TIP: Open another terminal and run:"
echo "  tail -f logs/stock_data.log"
echo ""
echo "======================================================================"
echo "TRAINING IN PROGRESS..."
echo "======================================================================"
echo ""

python main.py train-multitask

if [ $? -ne 0 ]; then
    echo ""
    echo "======================================================================"
    echo "❌ TRAINING FAILED"
    echo "======================================================================"
    echo "Check logs/stock_data.log for errors"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✅ TRAINING COMPLETED SUCCESSFULLY!"
echo "======================================================================"
echo ""
echo "[3/3] Now evaluate the model..."
echo ""

python main.py eval-multitask

echo ""
echo "======================================================================"
echo "✅ ALL DONE!"
echo "======================================================================"
echo ""
echo "Trained model saved to: models/checkpoints/lstm_multitask_best.pth"
echo "Predictions saved to: data/processed/multitask_predictions.parquet"
echo ""
echo "Check the results above. Expected:"
echo "  - Classification Accuracy: 45-52%"
echo "  - Regression RMSE: ~0.015"
echo "  - All 3 classes should be predicted (DOWN, UP, NEUTRAL)"
echo ""
