#!/usr/bin/env python3
"""
Test Hybrid IK Solver: MLP warm-start + DLS refinement
Compare against pure DLS baseline.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import importlib.util

# Load dh_utils
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("dh2", os.path.join(THIS_DIR, "dh_utils (2).py"))
dh2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dh2)
sys.modules['dh_utils'] = dh2

from module_catalog import get_module_catalog
from dls_ik_baseline import (
    forward_kinematics,
    inverse_kinematics_dls,
    euler_to_rotation_matrix,
    rotation_error_cross,
)
from generate_seed_dataset_poc import quaternion_from_rotation_matrix, encode_dh_params

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class IKSeedNet(nn.Module):
    """Simple MLP for IK seed prediction."""
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


def load_model(model_path='ik_seed_model_poc.pth'):
    """Load trained model and normalization parameters."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = IKSeedNet(checkpoint['input_dim'], checkpoint['output_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['feature_mean'], checkpoint['feature_std']


def predict_seed(model, target_pos, target_R, config, q_current, feat_mean, feat_std):
    """Use MLP to predict initial joint angles."""
    # Prepare features
    target_quat = quaternion_from_rotation_matrix(target_R)
    dh_flat = encode_dh_params(config)
    
    features = np.hstack([
        target_pos,
        target_quat,
        dh_flat,
        q_current,
    ])
    
    # Normalize
    features_norm = (features - feat_mean) / feat_std
    
    # Predict
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features_norm).unsqueeze(0)
        delta_q_pred = model(features_tensor).squeeze(0).numpy()
    
    # Apply delta
    q_seed = q_current + delta_q_pred
    
    return q_seed, delta_q_pred


def hybrid_ik_solver(model, feat_mean, feat_std, config, target_pos, target_R, 
                    q_current=None, max_iter_dls=500):
    """
    Hybrid solver: MLP warm-start + DLS refinement.
    
    Returns:
        q_solution, time_mlp_ms, time_dls_ms, iterations_used
    """
    if q_current is None:
        q_current = np.zeros(len(config))
    
    # Step 1: MLP prediction
    start_mlp = time.time()
    q_seed, delta_pred = predict_seed(model, target_pos, target_R, config, 
                                      q_current, feat_mean, feat_std)
    time_mlp = (time.time() - start_mlp) * 1000  # ms
    
    # Step 2: DLS refinement from MLP seed
    start_dls = time.time()
    q_solution = inverse_kinematics_dls(
        config, target_pos, target_R,
        q_init=q_seed,
        max_iter=max_iter_dls,
        lam=0.01,
        pos_tol=1e-4,
        ori_tol=1e-4,
    )
    time_dls = (time.time() - start_dls) * 1000  # ms
    
    return q_solution, time_mlp, time_dls


def compare_solvers(model, feat_mean, feat_std, config, n_tests=50):
    """Compare hybrid solver vs pure DLS."""
    print(f"\n⚖️  COMPARAISON SOLVEURS - {n_tests} Tests")
    print("="*70)
    
    results = {
        'pure_dls': [],
        'hybrid': [],
    }
    
    for i in range(n_tests):
        # Generate random target
        q_rand = np.random.uniform(-60, 60, len(config))
        for j, joint in enumerate(config):
            if joint['type'] == 'rot180':
                q_rand[j] = np.clip(q_rand[j], -90, 90)
        
        x_target, R_target = forward_kinematics(config, q_rand)
        q_current = np.zeros(len(config))
        
        # Pure DLS (from zeros)
        start = time.time()
        q_dls = inverse_kinematics_dls(config, x_target, R_target, 
                                      q_init=q_current, max_iter=1000, lam=0.01)
        time_dls = (time.time() - start) * 1000
        
        x_check_dls, R_check_dls = forward_kinematics(config, q_dls)
        err_pos_dls = np.linalg.norm(x_target - x_check_dls) * 1000
        err_ori_dls = np.linalg.norm(rotation_error_cross(R_check_dls, R_target)) * 180 / np.pi
        
        results['pure_dls'].append({
            'time_ms': time_dls,
            'pos_err': err_pos_dls,
            'ori_err': err_ori_dls,
        })
        
        # Hybrid (MLP + DLS refinement with max 100 iters)
        q_hybrid, time_mlp, time_dls_refine = hybrid_ik_solver(
            model, feat_mean, feat_std, config, x_target, R_target, 
            q_current, max_iter_dls=500
        )
        
        x_check_hyb, R_check_hyb = forward_kinematics(config, q_hybrid)
        err_pos_hyb = np.linalg.norm(x_target - x_check_hyb) * 1000
        err_ori_hyb = np.linalg.norm(rotation_error_cross(R_check_hyb, R_target)) * 180 / np.pi
        
        results['hybrid'].append({
            'time_ms': time_mlp + time_dls_refine,
            'time_mlp_ms': time_mlp,
            'time_dls_ms': time_dls_refine,
            'pos_err': err_pos_hyb,
            'ori_err': err_ori_hyb,
        })
        
        # Print first 5
        if i < 5:
            print(f"Test {i+1}:")
            print(f"  Pure DLS:  {time_dls:.1f}ms, Pos={err_pos_dls:.2f}mm")
            print(f"  Hybrid:    {time_mlp + time_dls_refine:.1f}ms " +
                  f"(MLP:{time_mlp:.1f}ms + DLS:{time_dls_refine:.1f}ms), Pos={err_pos_hyb:.2f}mm")
    
    return results


def analyze_comparison(results):
    """Analyze and visualize comparison results."""
    # Extract metrics
    dls_times = [r['time_ms'] for r in results['pure_dls']]
    dls_pos_errs = [r['pos_err'] for r in results['pure_dls']]
    
    hyb_times = [r['time_ms'] for r in results['hybrid']]
    hyb_pos_errs = [r['pos_err'] for r in results['hybrid']]
    hyb_mlp_times = [r['time_mlp_ms'] for r in results['hybrid']]
    hyb_dls_times = [r['time_dls_ms'] for r in results['hybrid']]
    
    # Statistics
    print(f"\n{'='*70}")
    print("📊 STATISTIQUES COMPARATIVES")
    print(f"{'='*70}")
    
    print(f"\nTEMPS DE CALCUL:")
    print(f"  Pure DLS (1000 iter):")
    print(f"    Moyenne: {np.mean(dls_times):.1f} ms")
    print(f"    Médiane: {np.median(dls_times):.1f} ms")
    
    print(f"\n  Hybrid (MLP + 100 iter DLS):")
    print(f"    Moyenne totale: {np.mean(hyb_times):.1f} ms")
    print(f"    - MLP:  {np.mean(hyb_mlp_times):.1f} ms")
    print(f"    - DLS:  {np.mean(hyb_dls_times):.1f} ms")
    
    speedup = np.mean(dls_times) / np.mean(hyb_times)
    print(f"\n  ⚡ SPEEDUP: {speedup:.1f}× plus rapide")
    
    print(f"\nPRÉCISION POSITION:")
    print(f"  Pure DLS:  {np.mean(dls_pos_errs):.3f} mm (max: {np.max(dls_pos_errs):.3f}mm)")
    print(f"  Hybrid:    {np.mean(hyb_pos_errs):.3f} mm (max: {np.max(hyb_pos_errs):.3f}mm)")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Time comparison
    ax1 = axes[0, 0]
    ax1.boxplot([dls_times, hyb_times], labels=['Pure DLS\n(1000 iter)', 'Hybrid\n(MLP + 100 iter)'])
    ax1.set_ylabel('Temps (ms)', fontsize=11)
    ax1.set_title('Comparaison Temps de Calcul', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add speedup annotation
    ax1.text(1.5, max(dls_times)*0.9, f'{speedup:.1f}× plus rapide', 
            fontsize=12, fontweight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Accuracy comparison
    ax2 = axes[0, 1]
    ax2.boxplot([dls_pos_errs, hyb_pos_errs], labels=['Pure DLS', 'Hybrid'])
    ax2.set_ylabel('Erreur Position (mm)', fontsize=11)
    ax2.set_title('Comparaison Précision', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Time breakdown for hybrid
    ax3 = axes[1, 0]
    mlp_mean = np.mean(hyb_mlp_times)
    dls_mean = np.mean(hyb_dls_times)
    ax3.bar(['MLP\nInference', 'DLS\nRefinement'], [mlp_mean, dls_mean], 
           color=['orange', 'blue'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Temps (ms)', fontsize=11)
    ax3.set_title('Décomposition Temps Hybrid', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for i, (label, val) in enumerate(zip(['MLP', 'DLS'], [mlp_mean, dls_mean])):
        ax3.text(i, val + 2, f'{val:.1f}ms', ha='center', fontsize=10, fontweight='bold')
    
    # Scatter: time vs accuracy
    ax4 = axes[1, 1]
    ax4.scatter(dls_times, dls_pos_errs, alpha=0.5, s=50, color='blue', label='Pure DLS')
    ax4.scatter(hyb_times, hyb_pos_errs, alpha=0.5, s=50, color='green', label='Hybrid')
    ax4.set_xlabel('Temps (ms)', fontsize=11)
    ax4.set_ylabel('Erreur Position (mm)', fontsize=11)
    ax4.set_title('Temps vs Précision', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Validation Hybrid Solver PoC - MLP + DLS vs Pure DLS',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('hybrid_solver_comparison_poc.png', dpi=150, bbox_inches='tight')
    print("\n💾 Saved: hybrid_solver_comparison_poc.png")
    plt.show()
    
    return speedup, np.mean(hyb_pos_errs)


def main():
    if not TORCH_AVAILABLE:
        print("❌ PyTorch required. Install: pip install torch")
        return
    
    print("🧪 TEST HYBRID SOLVER POC")
    print("MLP Warm-Start + DLS Refinement vs Pure DLS")
    print("="*70)
    
    # Load model
    model_path = 'ik_seed_model_poc.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Run: python train_seed_network_poc.py first")
        return
    
    print(f"📂 Loading model: {model_path}")
    model, feat_mean, feat_std = load_model(model_path)
    print("✅ Model loaded")
    
    # Get config
    catalog = get_module_catalog()
    config = catalog['SET_D_EXTENDED_REACH'].config
    
    print(f"\nConfiguration: SET_D Extended Reach")
    print(f"DDL: {len(config)}")
    
    # Run comparison
    results = compare_solvers(model, feat_mean, feat_std, config, n_tests=50)
    
    # Analyze
    speedup, avg_error = analyze_comparison(results)
    
    # Final verdict
    print(f"\n{'='*70}")
    print("🎯 RÉSULTAT POC")
    print(f"{'='*70}")
    
    if speedup > 5.0 and avg_error < 5.0:
        print(f"\n🎉 SUCCESS!")
        print(f"   ⚡ Speedup: {speedup:.1f}× plus rapide")
        print(f"   🎯 Précision: {avg_error:.3f}mm (maintenue)")
        print(f"\n✅ PoC VALIDÉ - Procéder implémentation complète:")
        print("   1. Générer 50k samples (Set A+D+E)")
        print("   2. Entraîner modèle 128-128-128")
        print("   3. Intégrer avec vision temps-réel")
    elif speedup > 3.0:
        print(f"\n✅ PROMISING")
        print(f"   Speedup: {speedup:.1f}×")
        print(f"   Précision: {avg_error:.3f}mm")
        print(f"\n💡 Amélio possibles:")
        print("   - Plus de données (10k-20k)")
        print("   - Modèle plus grand (128-128)")
    else:
        print(f"\n⚠️  MARGINAL")
        print(f"   Speedup: {speedup:.1f}× (espéré: 5-10×)")
        print(f"\n🔧 Debug requis:")
        print("   - Vérifier qualité prédictions MLP")
        print("   - Augmenter taille dataset")
        print("   - Tuner architecture")
    
    print(f"\n{'='*70}")
    print("Fichiers générés:")
    print("  - hybrid_solver_comparison_poc.png")


if __name__ == "__main__":
    main()

