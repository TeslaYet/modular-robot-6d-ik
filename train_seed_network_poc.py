#!/usr/bin/env python3
"""
Train simple MLP to predict IK seed (Δq) from target pose.
Proof of Concept with 64-64 architecture.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch not installed!")
    print("Install with: pip install torch")
    exit(1)


class IKSeedDataset(Dataset):
    """PyTorch dataset for IK seed prediction."""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class IKSeedNet(nn.Module):
    """Simple MLP for IK seed prediction (64-64 architecture)."""
    def __init__(self, input_dim, output_dim):
        super(IKSeedNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cpu'):
    """Train the IK seed network."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    print(f"\n🎓 Training Started")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print("="*70)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                predictions = model(features)
                loss = criterion(predictions, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss = {train_loss:.6f}, "
                  f"Val Loss = {val_loss:.6f}")
    
    print("\n✅ Training Complete!")
    return train_losses, val_losses


def plot_training_curves(train_losses, val_losses, save_path='training_curves_poc.png'):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Courbes d\'Apprentissage - IK Seed Network PoC', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add min validation loss annotation
    min_val_idx = np.argmin(val_losses)
    min_val_loss = val_losses[min_val_idx]
    plt.plot(min_val_idx + 1, min_val_loss, 'go', markersize=10)
    plt.annotate(f'Min: {min_val_loss:.6f}', 
                xy=(min_val_idx + 1, min_val_loss),
                xytext=(min_val_idx + 1 + 5, min_val_loss + 0.001),
                fontsize=10, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved: {save_path}")
    plt.show()


def main():
    print("🧠 ENTRAÎNEMENT RÉSEAU IK SEED - POC")
    print("="*70)
    
    # Load dataset
    dataset_path = 'ik_seed_dataset_poc.npz'
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("Run: python generate_seed_dataset_poc.py first")
        return
    
    print(f"📂 Loading dataset: {dataset_path}")
    data = np.load(dataset_path)
    
    features = data['features']
    labels = data['labels']
    feature_mean = data['feature_mean']
    feature_std = data['feature_std']
    
    print(f"   Samples: {len(features)}")
    print(f"   Input dim: {features.shape[1]}")
    print(f"   Output dim: {labels.shape[1]}")
    
    # Normalize features
    features_norm = (features - feature_mean) / feature_std
    
    # Create dataset
    dataset = IKSeedDataset(features_norm, labels)
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n📊 Split:")
    print(f"   Train: {train_size} samples")
    print(f"   Val:   {val_size} samples")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # Create model
    input_dim = features.shape[1]
    output_dim = labels.shape[1]
    model = IKSeedNet(input_dim, output_dim)
    
    print(f"\n🏗️  Model Architecture:")
    print(f"   Input:  {input_dim}")
    print(f"   Hidden: 64 → 64")
    print(f"   Output: {output_dim}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total params: {total_params:,}")
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"\n🚀 GPU detected! Using CUDA")
    else:
        print(f"\n💻 Using CPU")
    
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        epochs=50, lr=1e-3, device=device
    )
    
    # Save model
    model_path = 'ik_seed_model_poc.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'input_dim': input_dim,
        'output_dim': output_dim,
    }, model_path)
    
    print(f"\n💾 Model saved: {model_path}")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    
    # Quick evaluation on validation set
    model.eval()
    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            preds = model(features).cpu().numpy()
            val_predictions.append(preds)
            val_targets.append(labels.numpy())
    
    val_predictions = np.vstack(val_predictions)
    val_targets = np.vstack(val_targets)
    
    # Compute prediction errors
    pred_errors = np.abs(val_predictions - val_targets)
    mean_error_per_joint = np.mean(pred_errors, axis=0)
    mean_error_overall = np.mean(pred_errors)
    
    print(f"\n📊 Validation Set Performance:")
    print(f"   Mean Δq error: {mean_error_overall:.2f}° (across all joints)")
    print(f"   Per-joint errors: {np.round(mean_error_per_joint, 2)}")
    
    if mean_error_overall < 20:
        print(f"\n✅ SUCCESS: Mean error {mean_error_overall:.2f}° < 20°")
        print("   MLP predictions are within acceptable range")
        print("   → Proceed to hybrid solver testing")
    elif mean_error_overall < 40:
        print(f"\n⚠️  MARGINAL: Mean error {mean_error_overall:.2f}° (20-40°)")
        print("   MLP provides rough initialization")
        print("   → Test if DLS refinement still helps")
    else:
        print(f"\n❌ POOR: Mean error {mean_error_overall:.2f}° > 40°")
        print("   MLP needs improvement:")
        print("   - More data (10k-20k samples)")
        print("   - Larger model (128-128)")
        print("   - More epochs (100+)")
    
    print("\n" + "="*70)
    print("✅ POC TRAINING COMPLETE")
    print("="*70)
    print("\nFichiers générés:")
    print(f"  - {model_path} (modèle entraîné)")
    print(f"  - training_curves_poc.png (courbes)")
    print(f"  - dataset_poc_statistics.png (dataset viz)")
    
    print("\nProchaine étape:")
    print("  python test_hybrid_solver_poc.py")


if __name__ == "__main__":
    main()

