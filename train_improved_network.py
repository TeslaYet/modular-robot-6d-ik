#!/usr/bin/env python3
"""
Train improved MLP: 128-128 architecture with 10k samples.
Target: <15° prediction error for better hybrid performance.
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


class IKSeedNetImproved(nn.Module):
    """Improved MLP for IK seed prediction (128-128 architecture)."""
    def __init__(self, input_dim, output_dim):
        super(IKSeedNetImproved, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, device='cpu'):
    """Train the improved IK seed network."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\n🎓 Training Started")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr} (with scheduler)")
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
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }, 'ik_seed_model_improved_best.pth')
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train = {train_loss:.4f}, "
                  f"Val = {val_loss:.4f}, "
                  f"Best = {best_val_loss:.4f}")
    
    print("\n✅ Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    return train_losses, val_losses


def plot_training_curves(train_losses, val_losses, save_path='training_curves_improved.png'):
    """Plot training curves with improvements."""
    plt.figure(figsize=(12, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.7)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    # Mark best epoch
    best_epoch = np.argmin(val_losses) + 1
    best_val = min(val_losses)
    plt.plot(best_epoch, best_val, 'g*', markersize=15, label=f'Best (epoch {best_epoch})')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Courbes d\'Apprentissage - IK Seed Network Amélioré (128-128, 10k samples)', 
             fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved: {save_path}")
    plt.show()


def main():
    if not TORCH_AVAILABLE:
        print("❌ PyTorch required")
        return
    
    print("🧠 ENTRAÎNEMENT RÉSEAU AMÉLIORÉ - 128-128 MLP")
    print("="*70)
    
    # Load dataset
    dataset_path = 'ik_seed_dataset_10k.npz'
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("Run: python generate_improved_dataset.py first")
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
    
    # Normalize
    features_norm = (features - feature_mean) / feature_std
    
    # Create dataset
    dataset = IKSeedDataset(features_norm, labels)
    
    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n📊 Split:")
    print(f"   Train: {train_size} samples")
    print(f"   Val:   {val_size} samples")
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    # Create improved model
    input_dim = features.shape[1]
    output_dim = labels.shape[1]
    model = IKSeedNetImproved(input_dim, output_dim)
    
    print(f"\n🏗️  Model Architecture:")
    print(f"   Input:  {input_dim}")
    print(f"   Hidden: 128 → Dropout(0.1) → 128 → Dropout(0.1)")
    print(f"   Output: {output_dim}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total params: {total_params:,}")
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"\n🚀 GPU detected! Using CUDA")
    else:
        print(f"\n💻 Using CPU (consider GPU for faster training)")
    
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        epochs=100, lr=1e-3, device=device
    )
    
    # Load best model for final evaluation
    best_checkpoint = torch.load('ik_seed_model_improved_best.pth', 
                                map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Save final model with normalization params
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'best_epoch': best_checkpoint['epoch'],
        'best_val_loss': best_checkpoint['val_loss'],
    }, 'ik_seed_model_improved.pth')
    
    print(f"\n💾 Models saved:")
    print(f"   - ik_seed_model_improved.pth (final)")
    print(f"   - ik_seed_model_improved_best.pth (best checkpoint)")
    
    # Plot curves
    plot_training_curves(train_losses, val_losses)
    
    # Evaluate on validation set
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
    
    # Compute errors
    pred_errors = np.abs(val_predictions - val_targets)
    mean_error = np.mean(pred_errors)
    per_joint = np.mean(pred_errors, axis=0)
    
    print(f"\n📊 Validation Performance:")
    print(f"   Mean Δq error: {mean_error:.2f}°")
    print(f"   Per-joint: {np.round(per_joint, 2)}")
    
    if mean_error < 15:
        print(f"\n🎉 SUCCESS!")
        print(f"   Mean error {mean_error:.2f}° < 15°")
        print(f"   → Hybrid with 200-300 DLS iters should work well")
        print(f"   → Expected: 3-5× speedup + <5mm accuracy")
    elif mean_error < 20:
        print(f"\n✅ GOOD!")
        print(f"   Mean error {mean_error:.2f}° < 20°")
        print(f"   → Hybrid with 300-500 DLS iters should work")
        print(f"   → Expected: 2-3× speedup + <5mm accuracy")
    else:
        print(f"\n⚠️  Needs more improvement: {mean_error:.2f}°")
        print(f"   Try: 20k samples or 128-128-128 architecture")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print("\nProchaine étape:")
    print("  python test_hybrid_solver_improved.py")


if __name__ == "__main__":
    main()

