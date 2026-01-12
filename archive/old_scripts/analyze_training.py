import torch
import pandas as pd
import numpy as np

# Load checkpoint
cp = torch.load('models/checkpoints/lstm_regression_best.pth', map_location='cpu')
h = cp['history']

print('='*70)
print('TRAINING PROGRESSION ANALYSIS')
print('='*70)
print(f'\nBest val loss: {cp["val_loss"]:.8f}')
print(f'Total epochs: {len(h["train_loss"])}')

print(f'\nFirst 10 epochs:')
for i in range(min(10, len(h['train_loss']))):
    print(f'  Epoch {i+1:2d}: Train={h["train_loss"][i]:.6f}, Val={h["val_loss"][i]:.6f}')

print(f'\nLast 5 epochs:')
for i in range(max(0, len(h['train_loss'])-5), len(h['train_loss'])):
    print(f'  Epoch {i+1:2d}: Train={h["train_loss"][i]:.6f}, Val={h["val_loss"][i]:.6f}')

# Analyze predictions
print('\n' + '='*70)
print('PREDICTION ANALYSIS')
print('='*70)
df = pd.read_parquet('data/processed/regression_predictions.parquet')

print(f'\nPredictions:')
print(f'  Mean: {df["y_pred"].mean():.6f}')
print(f'  Std:  {df["y_pred"].std():.6f}')
print(f'  Range: [{df["y_pred"].min():.6f}, {df["y_pred"].max():.6f}]')

print(f'\nTrue values:')
print(f'  Mean: {df["y_true"].mean():.6f}')
print(f'  Std:  {df["y_true"].std():.6f}')
print(f'  Range: [{df["y_true"].min():.6f}, {df["y_true"].max():.6f}]')

print(f'\nVariance ratio: {(df["y_pred"].std() / df["y_true"].std()):.4f}')
print(f'Correlation: {np.corrcoef(df["y_true"], df["y_pred"])[0,1]:.4f}')

# Check if loss is suspiciously low
print(f'\n' + '='*70)
print('DIAGNOSIS')
print('='*70)

if cp["val_loss"] < 0.001:
    print('⚠️  ISSUE: Validation loss is extremely low (<0.001)')
    print('   This indicates the model learned to output near-zero predictions')
    print('   to minimize Huber loss on small-scale targets (std~0.018)')

if df["y_pred"].std() / df["y_true"].std() < 0.15:
    print('⚠️  ISSUE: Severe underfitting detected')
    print(f'   Predictions only {(df["y_pred"].std() / df["y_true"].std()*100):.1f}% of actual variance')
    print('   Model is too conservative')
