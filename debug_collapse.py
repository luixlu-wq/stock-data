import torch
import numpy as np

# Load the latest checkpoint
checkpoint = torch.load('models/checkpoints/lstm_regression_best.pth')

print('='*70)
print('CHECKPOINT INSPECTION')
print('='*70)

# Show training history
history = checkpoint.get('history', {})
print(f"\nEpochs trained: {len(history.get('train_loss', []))}")
print(f"Best epoch: {checkpoint.get('epoch', 'N/A')}")
best_loss = checkpoint.get('best_loss', float('inf'))
if isinstance(best_loss, (int, float)) and best_loss != float('inf'):
    print(f"Best val loss: {best_loss:.8f}")
else:
    print(f"Best val loss: {best_loss}")

# Show last few training losses
if history.get('train_loss'):
    print(f"\nLast 5 train losses:")
    for i, loss in enumerate(history['train_loss'][-5:], 1):
        print(f"  Epoch {i}: {loss:.8f}")

if history.get('val_loss'):
    print(f"\nLast 5 val losses:")
    for i, loss in enumerate(history['val_loss'][-5:], 1):
        print(f"  Epoch {i}: {loss:.8f}")

# Check model state
model_state = checkpoint.get('model_state_dict', {})
print(f"\nModel state keys: {len(model_state.keys())}")

# Check if weights are actually changing
lstm_weights = [k for k in model_state.keys() if 'lstm' in k and 'weight' in k]
print(f"\nLSTM weight layers: {len(lstm_weights)}")

if lstm_weights:
    first_weight = model_state[lstm_weights[0]]
    print(f"\nFirst LSTM weight stats:")
    print(f"  Shape: {first_weight.shape}")
    print(f"  Mean: {first_weight.mean():.8f}")
    print(f"  Std: {first_weight.std():.8f}")
    print(f"  Min: {first_weight.min():.8f}")
    print(f"  Max: {first_weight.max():.8f}")

# Check output layer
output_weights = [k for k in model_state.keys() if 'fc' in k and 'weight' in k]
if output_weights:
    fc_weight = model_state[output_weights[-1]]
    print(f"\nOutput layer weight stats:")
    print(f"  Shape: {fc_weight.shape}")
    print(f"  Mean: {fc_weight.mean():.8f}")
    print(f"  Std: {fc_weight.std():.8f}")
    print(f"  Min: {fc_weight.min():.8f}")
    print(f"  Max: {fc_weight.max():.8f}")

    # Check if output layer bias exists
    fc_bias_key = output_weights[-1].replace('weight', 'bias')
    if fc_bias_key in model_state:
        fc_bias = model_state[fc_bias_key]
        print(f"\nOutput layer bias:")
        print(f"  Value: {fc_bias.item():.8f}")

print('\n' + '='*70)
print('POSSIBLE ISSUES:')
print('='*70)

# Check for vanishing gradients
if lstm_weights:
    first_weight = model_state[lstm_weights[0]]
    if first_weight.std() < 0.01:
        print("⚠️  LSTM weights have very low variance - possible vanishing gradients")
    if abs(first_weight.mean()) > 1.0:
        print("⚠️  LSTM weights have high mean - possible exploding gradients")

# Check output layer
if output_weights:
    fc_weight = model_state[output_weights[-1]]
    if fc_weight.std() < 0.01:
        print("⚠️  Output layer has very low variance")

    fc_bias_key = output_weights[-1].replace('weight', 'bias')
    if fc_bias_key in model_state:
        fc_bias = model_state[fc_bias_key]
        if abs(fc_bias.item()) < 1e-6:
            print("⚠️  Output bias is near zero - model may be outputting zeros")

print('\n' + '='*70)
