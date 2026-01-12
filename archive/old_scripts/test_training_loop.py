"""
Quick diagnostic to test if training loop is working.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys

print("=" * 70)
print("TRAINING LOOP DIAGNOSTIC")
print("=" * 70)

# Create dummy data
print("\n[1/5] Creating dummy data...")
X = torch.randn(1000, 60, 41)  # 1000 samples, 60 time steps, 41 features
y_reg = torch.randn(1000, 1)   # Regression target
y_clf = torch.randint(0, 3, (1000,))  # Classification target (3 classes)

dataset = TensorDataset(X, y_reg, y_clf)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
print(f"Created dataset: {len(dataset)} samples, batch_size=256")

# Create simple model
print("\n[2/5] Creating model...")
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(41, 128, 2, batch_first=True)
        self.reg_head = nn.Linear(128, 1)
        self.clf_head = nn.Linear(128, 3)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        features = h[-1]
        return self.reg_head(features), self.clf_head(features)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = SimpleModel().to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# Setup training
print("\n[3/5] Setting up training...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
reg_loss_fn = nn.HuberLoss()
clf_loss_fn = nn.CrossEntropyLoss()
print("Optimizer: Adam (lr=0.003)")
print("Losses: Huber (regression), CrossEntropy (classification)")

# Train for 5 epochs
print("\n[4/5] Training for 5 epochs...")
print("-" * 70)

model.train()
for epoch in range(5):
    total_loss = 0
    num_batches = 0

    for batch_X, batch_y_reg, batch_y_clf in dataloader:
        batch_X = batch_X.to(device)
        batch_y_reg = batch_y_reg.to(device)
        batch_y_clf = batch_y_clf.to(device)

        optimizer.zero_grad()

        reg_pred, clf_pred = model(batch_X)

        reg_loss = reg_loss_fn(reg_pred, batch_y_reg)
        clf_loss = clf_loss_fn(clf_pred, batch_y_clf)

        loss = reg_loss + 0.5 * clf_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    progress = (epoch + 1) / 5 * 100

    # This should print to console
    msg = f"Epoch [{epoch+1:3d}/5] ({progress:5.1f}%) - Loss: {avg_loss:.4f}"
    print(msg)
    sys.stdout.flush()  # Force flush

print("-" * 70)

print("\n[5/5] Testing model output...")
model.eval()
with torch.no_grad():
    test_x = torch.randn(10, 60, 41).to(device)
    reg_out, clf_out = model(test_x)
    print(f"Regression output shape: {reg_out.shape}")
    print(f"Classification output shape: {clf_out.shape}")
    print(f"Regression values (first 3): {reg_out[:3, 0].cpu().numpy()}")
    print(f"Classification predictions: {clf_out.argmax(dim=1).cpu().numpy()}")

print("\n" + "=" * 70)
print("✅ DIAGNOSTIC COMPLETE - Training loop works!")
print("=" * 70)
