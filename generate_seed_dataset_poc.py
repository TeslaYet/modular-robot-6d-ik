#!/usr/bin/env python3
"""
Proof of Concept: Generate IK training dataset from SET_D
Creates 5k samples of (target_pose, DH, q_current) → Δq labels
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import importlib.util
import time

# Load dh_utils (2).py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DH_UTILS2_PATH = os.path.join(THIS_DIR, "dh_utils (2).py")
spec = importlib.util.spec_from_file_location("dh2", DH_UTILS2_PATH)
dh2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dh2)
sys.modules['dh_utils'] = dh2

from module_catalog import get_module_catalog
from dls_ik_baseline import (
    forward_kinematics,
    inverse_kinematics_dls,
    euler_to_rotation_matrix,
)


def quaternion_from_rotation_matrix(R):
    """Convert rotation matrix to quaternion [w, x, y, z]."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])


def encode_dh_params(config):
    """Flatten DH parameters to vector."""
    features = []
    for joint in config:
        features.extend([
            joint['d'],
            joint['a'],
            joint['alpha'],
            1.0 if joint['type'] == 'rot360' else 0.0,  # Joint type encoding
        ])
    return np.array(features)


def generate_sample(config, q_current=None):
    """
    Generate a single training sample.
    
    Returns:
        features: [target_pos(3), target_quat(4), DH_flat(4*n), q_current(n)]
        label: Δq = q_dls - q_current
        metadata: errors, time, etc.
    """
    n_dof = len(config)
    
    # Random target joint angles
    q_target_random = np.random.uniform(-60, 60, n_dof)
    for i, joint in enumerate(config):
        if joint['type'] == 'rot180':
            q_target_random[i] = np.clip(q_target_random[i], -90, 90)
    
    # Forward kinematics to get target pose
    x_target, R_target = forward_kinematics(config, q_target_random)
    
    # Current joint state (neutral or small perturbation)
    if q_current is None:
        q_current = np.random.uniform(-10, 10, n_dof)
    
    # Solve IK with DLS (ground truth)
    start_time = time.time()
    try:
        q_dls = inverse_kinematics_dls(
            config, x_target, R_target,
            q_init=q_current,
            max_iter=1000,
            lam=0.01,
            pos_tol=1e-4,
            ori_tol=1e-4,
        )
        solve_time = time.time() - start_time
        
        # Verify solution quality
        x_check, R_check = forward_kinematics(config, q_dls)
        pos_error = np.linalg.norm(x_target - x_check) * 1000  # mm
        
        # Label: delta from current to solution
        delta_q = q_dls - q_current
        
        # Features
        target_pos_norm = x_target  # Will normalize later
        target_quat = quaternion_from_rotation_matrix(R_target)
        dh_flat = encode_dh_params(config)
        
        features = np.hstack([
            target_pos_norm,    # 3
            target_quat,        # 4
            dh_flat,           # 4*n_dof
            q_current,         # n_dof
        ])
        
        return {
            'features': features,
            'label': delta_q,
            'pos_error_mm': pos_error,
            'solve_time_ms': solve_time * 1000,
            'success': True,
        }
    
    except Exception as e:
        return {'success': False, 'error': str(e)}


def generate_dataset(config, n_samples=5000, save_path='ik_seed_dataset_poc.npz'):
    """Generate full dataset and save to disk."""
    print(f"🔄 Generating {n_samples} training samples from SET_D")
    print(f"Expected time: ~{n_samples * 0.5 / 60:.1f} minutes")
    print("="*70)
    
    features_list = []
    labels_list = []
    metadata_list = []
    
    failed = 0
    
    # Use tqdm for progress bar
    try:
        from tqdm import tqdm
        iterator = tqdm(range(n_samples), desc="Generating samples")
    except ImportError:
        print("Install tqdm for progress bar: pip install tqdm")
        iterator = range(n_samples)
    
    for i in iterator:
        sample = generate_sample(config)
        
        if sample['success']:
            features_list.append(sample['features'])
            labels_list.append(sample['label'])
            metadata_list.append({
                'pos_error_mm': sample['pos_error_mm'],
                'solve_time_ms': sample['solve_time_ms'],
            })
        else:
            failed += 1
    
    # Convert to arrays
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    # Compute normalization statistics
    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0) + 1e-8
    label_mean = np.mean(labels, axis=0)
    label_std = np.std(labels, axis=0) + 1e-8
    
    # Save dataset
    np.savez(
        save_path,
        features=features,
        labels=labels,
        feature_mean=feature_mean,
        feature_std=feature_std,
        label_mean=label_mean,
        label_std=label_std,
        metadata=metadata_list,
    )
    
    print(f"\n{'='*70}")
    print(f"✅ Dataset Generated Successfully")
    print(f"{'='*70}")
    print(f"Samples: {len(features)}")
    print(f"Failed: {failed}")
    print(f"Success rate: {len(features)/(len(features)+failed)*100:.1f}%")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Label dimension: {labels.shape[1]}")
    print(f"\nSaved to: {save_path}")
    
    # Dataset statistics
    print(f"\n📊 Dataset Statistics:")
    pos_errors = [m['pos_error_mm'] for m in metadata_list]
    solve_times = [m['solve_time_ms'] for m in metadata_list]
    
    print(f"DLS Solution Quality:")
    print(f"  Mean position error: {np.mean(pos_errors):.3f} mm")
    print(f"  Max position error:  {np.max(pos_errors):.3f} mm")
    print(f"DLS Solve Time:")
    print(f"  Mean: {np.mean(solve_times):.1f} ms")
    print(f"  Max:  {np.max(solve_times):.1f} ms")
    
    print(f"\nLabel Statistics (Δq in degrees):")
    print(f"  Mean |Δq|: {np.mean(np.abs(labels)):.2f}°")
    print(f"  Max |Δq|:  {np.max(np.abs(labels)):.2f}°")
    
    return features, labels, feature_mean, feature_std


def visualize_dataset(save_path='ik_seed_dataset_poc.npz'):
    """Visualize generated dataset statistics."""
    data = np.load(save_path, allow_pickle=True)
    features = data['features']
    labels = data['labels']
    metadata = data['metadata']
    
    fig = plt.figure(figsize=(16, 10))
    
    # Target positions in 3D
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    target_positions = features[:, 0:3]
    ax1.scatter(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2],
               c='blue', s=5, alpha=0.3)
    ax1.scatter(0, 0, 0, c='red', s=100, marker='s', label='Base')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Distribution Positions Cibles', fontweight='bold')
    ax1.legend()
    ax1.set_box_aspect([1, 1, 1])
    
    # Δq distribution per joint
    ax2 = fig.add_subplot(2, 3, 2)
    n_joints = labels.shape[1]
    for i in range(n_joints):
        ax2.hist(labels[:, i], bins=30, alpha=0.5, label=f'Joint {i+1}')
    ax2.set_xlabel('Δq (degrés)')
    ax2.set_ylabel('Fréquence')
    ax2.set_title('Distribution Δq par Articulation', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # DLS quality
    ax3 = fig.add_subplot(2, 3, 3)
    pos_errors = [m['pos_error_mm'] for m in metadata]
    ax3.hist(pos_errors, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(pos_errors), color='red', linestyle='--', linewidth=2,
                label=f'Moyenne: {np.mean(pos_errors):.3f}mm')
    ax3.set_xlabel('Erreur Position DLS (mm)')
    ax3.set_ylabel('Fréquence')
    ax3.set_title('Qualité Solutions DLS (Ground Truth)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Solve time distribution
    ax4 = fig.add_subplot(2, 3, 4)
    solve_times = [m['solve_time_ms'] for m in metadata]
    ax4.hist(solve_times, bins=30, color='orange', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(solve_times), color='red', linestyle='--', linewidth=2,
                label=f'Moyenne: {np.mean(solve_times):.1f}ms')
    ax4.set_xlabel('Temps Résolution DLS (ms)')
    ax4.set_ylabel('Fréquence')
    ax4.set_title('Temps de Calcul DLS', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Feature correlation heatmap (sample)
    ax5 = fig.add_subplot(2, 3, 5)
    # Show correlation between first few features
    sample_features = features[:, :10]  # First 10 features
    corr = np.corrcoef(sample_features.T)
    im = ax5.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax5.set_title('Corrélation Features (échantillon)', fontweight='bold')
    plt.colorbar(im, ax=ax5)
    
    # Summary panel
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    summary = f"DATASET POC - SET_D\n"
    summary += "="*50 + "\n\n"
    summary += f"Échantillons: {len(features)}\n"
    summary += f"Dimension features: {features.shape[1]}\n"
    summary += f"Dimension labels: {labels.shape[1]} (DDL)\n\n"
    
    summary += "QUALITÉ LABELS (DLS):\n"
    summary += f"  Err pos moyenne: {np.mean(pos_errors):.3f} mm\n"
    summary += f"  Err pos max: {np.max(pos_errors):.3f} mm\n"
    summary += f"  Temps moy: {np.mean(solve_times):.1f} ms\n\n"
    
    summary += "DISTRIBUTION Δq:\n"
    summary += f"  Moyenne |Δq|: {np.mean(np.abs(labels)):.2f}°\n"
    summary += f"  Max |Δq|: {np.max(np.abs(labels)):.2f}°\n\n"
    
    summary += "PRÊT POUR:\n"
    summary += "  ✅ Entraînement MLP\n"
    summary += "  ✅ Validation croisée\n"
    summary += "  ✅ Test hybride MLP+DLS\n"
    
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Dataset PoC - Statistiques et Distribution', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dataset_poc_statistics.png', dpi=150, bbox_inches='tight')
    print("\n💾 Saved visualization: dataset_poc_statistics.png")
    plt.show()


def main():
    print("🧠 GÉNÉRATION DATASET POC - APPRENTISSAGE IK")
    print("="*70)
    
    # Get SET_D configuration
    catalog = get_module_catalog()
    config = catalog['SET_D_EXTENDED_REACH'].config
    
    print(f"\nConfiguration: SET_D Extended Reach")
    print(f"DDL: {len(config)}")
    print(f"Modules: {[j['type'] for j in config]}")
    
    # Generate samples
    n_samples = 5000
    save_path = 'ik_seed_dataset_poc.npz'
    
    print(f"\n🎲 Génération {n_samples} échantillons...")
    print("Chaque échantillon:")
    print("  1. q_random → FK → target_pose")
    print("  2. DLS(target_pose, q_current) → q_solution")
    print("  3. Label: Δq = q_solution - q_current")
    print()
    
    start_time = time.time()
    features, labels, feat_mean, feat_std = generate_dataset(
        config, 
        n_samples=n_samples,
        save_path=save_path
    )
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Temps total: {total_time/60:.1f} minutes")
    print(f"   Temps par échantillon: {total_time/n_samples*1000:.1f} ms")
    
    # Visualize dataset
    print(f"\n🎨 Génération visualisation dataset...")
    visualize_dataset(save_path)
    
    print("\n" + "="*70)
    print("✅ DATASET POC PRÊT")
    print("="*70)
    print(f"\nFichier généré: {save_path}")
    print(f"Taille: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
    print("\nProchaine étape: python train_seed_network_poc.py")


if __name__ == "__main__":
    # Check if tqdm is available, install if not
    try:
        import tqdm
    except ImportError:
        print("⚠️  tqdm non installé. Installation recommandée:")
        print("   pip install tqdm")
        print("\nContinuation sans barre de progression...\n")
    
    main()

