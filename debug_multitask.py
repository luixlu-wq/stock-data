"""
Debug script to investigate why multitask model never predicts UP class.
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.preprocessor import StockPreprocessor
from src.models.lstm_model import create_model
from src.utils.config_loader import ConfigLoader

# Load config
config = ConfigLoader('config/config.yaml')

# Load test data
test_df = pd.read_parquet('data/processed/test.parquet')
print(f"Test dataframe shape: {test_df.shape}")
print(f"\nTest dataframe class distribution:")
print(test_df['target_class'].value_counts().sort_index())

# Create sequences
preprocessor = StockPreprocessor(config)
preprocessor.feature_columns = preprocessor.get_feature_columns(test_df)

sequence_length = config.get('data.sequence_length', 60)
X_test, y_test_reg, y_test_clf, metadata = preprocessor.create_sequences(test_df, sequence_length)

print(f"\nTest sequences shape: {X_test.shape}")
print(f"y_test_clf shape: {y_test_clf.shape}")
print(f"\nSequence-level class distribution:")
unique, counts = np.unique(y_test_clf, return_counts=True)
for cls, count in zip(unique, counts):
    cls_name = ['DOWN', 'UP', 'NEUTRAL'][int(cls)]
    print(f"{cls_name} ({int(cls)}): {count} ({count/len(y_test_clf)*100:.1f}%)")

# Load model
input_size = X_test.shape[2]
model_config = config.get('model.architecture')

model = create_model(
    model_type="multitask",
    input_size=input_size,
    config=model_config
)

# Load checkpoint
checkpoint_path = Path("models/checkpoints/lstm_multitask_best.pth")

if not checkpoint_path.exists():
    print(f"\nERROR: Checkpoint not found: {checkpoint_path}")
    exit(1)

checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"\nModel loaded from: {checkpoint_path}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test on first 100 samples
print("\n" + "="*70)
print("TESTING ON FIRST 100 SAMPLES")
print("="*70)

X_sample = torch.FloatTensor(X_test[:100])
y_clf_sample = y_test_clf[:100]

with torch.no_grad():
    reg_out, clf_logits = model(X_sample)

# Get predictions
clf_probs = torch.softmax(clf_logits, dim=1)
clf_preds = clf_logits.argmax(axis=1).numpy()

print(f"\nTrue labels (first 20): {y_clf_sample[:20]}")
print(f"Predictions (first 20): {clf_preds[:20]}")

print(f"\nPrediction distribution:")
unique_pred, counts_pred = np.unique(clf_preds, return_counts=True)
for cls, count in zip(unique_pred, counts_pred):
    cls_name = ['DOWN', 'UP', 'NEUTRAL'][int(cls)]
    print(f"{cls_name} ({int(cls)}): {count} ({count/len(clf_preds)*100:.1f}%)")

print(f"\nClassification logits statistics:")
print(f"DOWN (class 0) logits  - mean: {clf_logits[:, 0].mean():.4f}, std: {clf_logits[:, 0].std():.4f}")
print(f"UP (class 1) logits    - mean: {clf_logits[:, 1].mean():.4f}, std: {clf_logits[:, 1].std():.4f}")
print(f"NEUTRAL (class 2) logits - mean: {clf_logits[:, 2].mean():.4f}, std: {clf_logits[:, 2].std():.4f}")

print(f"\nSample probabilities (first 5 samples):")
for i in range(5):
    true_label = ['DOWN', 'UP', 'NEUTRAL'][int(y_clf_sample[i])]
    pred_label = ['DOWN', 'UP', 'NEUTRAL'][int(clf_preds[i])]
    probs = clf_probs[i].numpy()
    print(f"Sample {i}: True={true_label}, Pred={pred_label}")
    print(f"  Probs: DOWN={probs[0]:.3f}, UP={probs[1]:.3f}, NEUTRAL={probs[2]:.3f}")

# Check regression outputs too
print(f"\nRegression outputs:")
print(f"Mean: {reg_out.mean():.6f}, Std: {reg_out.std():.6f}")
print(f"Min: {reg_out.min():.6f}, Max: {reg_out.max():.6f}")
