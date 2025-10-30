#!/usr/bin/env python3
"""
FK→IK→FK Round-Trip Validation for SET_D (Extended Reach)
Tests solver reliability with 100 random joint configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import importlib.util

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
    rotation_error_cross,
    euler_to_rotation_matrix,
)


def roundtrip_test_single(config, q_original):
    """
    Single roundtrip test: FK(q_orig) → target → IK(target) → FK(q_solved) → compare
    Returns position error and orientation error.
    """
    # Forward kinematics from original q
    x_target, R_target = forward_kinematics(config, q_original)
    
    # Solve IK to reach that target
    q_solved = inverse_kinematics_dls(
        config, x_target, R_target,
        q_init=None,
        max_iter=1000,
        lam=0.01,
        pos_tol=1e-4,
        ori_tol=1e-4,
    )
    
    # Forward kinematics from solved q
    x_check, R_check = forward_kinematics(config, q_solved)
    
    # Compute errors
    pos_error = np.linalg.norm(x_target - x_check)  # meters
    ori_error = np.linalg.norm(rotation_error_cross(R_check, R_target))  # radians equivalent
    
    return {
        'q_original': q_original,
        'q_solved': q_solved,
        'x_target': x_target,
        'x_check': x_check,
        'pos_error_mm': pos_error * 1000,
        'ori_error_deg': ori_error * 180 / np.pi,
        'success': pos_error < 0.01 and ori_error < 0.01,  # 10mm, ~0.5° thresholds
    }


def run_roundtrip_tests(config, n_tests=100):
    """Run n_tests FK→IK→FK roundtrip tests with random joint angles."""
    print(f"🔄 Running {n_tests} FK→IK→FK Roundtrip Tests")
    print("="*70)
    
    results = []
    
    for i in range(n_tests):
        # Generate random joint angles (realistic range)
        q_rand = np.random.uniform(-60, 60, len(config))
        
        # Clamp rot180 joints to their limits
        for j, joint in enumerate(config):
            if joint['type'] == 'rot180':
                q_rand[j] = np.clip(q_rand[j], -90, 90)
        
        try:
            result = roundtrip_test_single(config, q_rand)
            results.append(result)
            
            # Print first 5 and last 5
            if i < 5 or i >= n_tests - 5:
                print(f"Test {i+1:3d}: Pos={result['pos_error_mm']:6.3f}mm, "
                      f"Ori={result['ori_error_deg']:8.5f}°")
        
        except Exception as e:
            print(f"Test {i+1:3d}: ❌ Failed - {e}")
    
    return results


def analyze_results(results):
    """Analyze and print statistics from roundtrip tests."""
    if not results:
        print("No successful results to analyze")
        return
    
    pos_errors = [r['pos_error_mm'] for r in results]
    ori_errors = [r['ori_error_deg'] for r in results]
    successes = [r for r in results if r['success']]
    
    print(f"\n{'='*70}")
    print(f"STATISTICAL ANALYSIS - {len(results)} Tests")
    print(f"{'='*70}")
    
    print(f"\nPosition Errors (mm):")
    print(f"  Mean:   {np.mean(pos_errors):8.3f} mm")
    print(f"  Median: {np.median(pos_errors):8.3f} mm")
    print(f"  Std:    {np.std(pos_errors):8.3f} mm")
    print(f"  Min:    {np.min(pos_errors):8.3f} mm")
    print(f"  Max:    {np.max(pos_errors):8.3f} mm")
    print(f"  95th %: {np.percentile(pos_errors, 95):8.3f} mm")
    
    print(f"\nOrientation Errors (degrees):")
    print(f"  Mean:   {np.mean(ori_errors):8.5f}°")
    print(f"  Median: {np.median(ori_errors):8.5f}°")
    print(f"  Std:    {np.std(ori_errors):8.5f}°")
    print(f"  Min:    {np.min(ori_errors):8.5f}°")
    print(f"  Max:    {np.max(ori_errors):8.5f}°")
    print(f"  95th %: {np.percentile(ori_errors, 95):8.5f}°")
    
    print(f"\nSuccess Rate:")
    print(f"  {len(successes)}/{len(results)} ({len(successes)/len(results)*100:.1f}%)")
    print(f"  Criteria: Pos < 10mm AND Ori < 0.5°")
    
    # Error distribution
    pos_ranges = [
        (0, 0.1, "< 0.1mm (Sub-micron)"),
        (0.1, 1, "0.1-1mm (Excellent)"),
        (1, 5, "1-5mm (Good)"),
        (5, 10, "5-10mm (Acceptable)"),
        (10, np.inf, "> 10mm (Poor)")
    ]
    
    print(f"\nPosition Error Distribution:")
    for low, high, label in pos_ranges:
        count = sum(1 for e in pos_errors if low <= e < high)
        pct = count / len(pos_errors) * 100
        bar = '█' * int(pct / 2)
        print(f"  {label:25s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    return {
        'pos_mean': np.mean(pos_errors),
        'ori_mean': np.mean(ori_errors),
        'success_rate': len(successes) / len(results),
    }


def visualize_results(results, config_name="SET_D", save_image=False):
    """Create comprehensive visualization of roundtrip test results."""
    if not results:
        return
    
    pos_errors = np.array([r['pos_error_mm'] for r in results])
    ori_errors = np.array([r['ori_error_deg'] for r in results])
    
    fig = plt.figure(figsize=(18, 10))
    
    # Subplot 1: Position error histogram
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.hist(pos_errors, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(pos_errors), color='red', linestyle='--', linewidth=2, 
                label=f'Moyenne: {np.mean(pos_errors):.3f}mm')
    ax1.axvline(np.median(pos_errors), color='green', linestyle='--', linewidth=2,
                label=f'Médiane: {np.median(pos_errors):.3f}mm')
    ax1.set_xlabel('Erreur Position (mm)', fontsize=10)
    ax1.set_ylabel('Fréquence', fontsize=10)
    ax1.set_title('Distribution Erreurs Position', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Orientation error histogram
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(ori_errors, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(ori_errors), color='red', linestyle='--', linewidth=2,
                label=f'Moyenne: {np.mean(ori_errors):.5f}°')
    ax2.set_xlabel('Erreur Orientation (°)', fontsize=10)
    ax2.set_ylabel('Fréquence', fontsize=10)
    ax2.set_title('Distribution Erreurs Orientation', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Scatter plot (position vs orientation error)
    ax3 = fig.add_subplot(2, 3, 3)
    scatter = ax3.scatter(pos_errors, ori_errors, c=pos_errors, cmap='viridis', 
                         alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Erreur Position (mm)', fontsize=10)
    ax3.set_ylabel('Erreur Orientation (°)', fontsize=10)
    ax3.set_title('Corrélation Position-Orientation', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Erreur Pos (mm)')
    
    # Subplot 4: Cumulative distribution (position)
    ax4 = fig.add_subplot(2, 3, 4)
    sorted_pos = np.sort(pos_errors)
    cumulative = np.arange(1, len(sorted_pos) + 1) / len(sorted_pos) * 100
    ax4.plot(sorted_pos, cumulative, color='blue', linewidth=2)
    ax4.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(95, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(np.median(pos_errors), color='green', linestyle='--', linewidth=2,
                label=f'Médiane: {np.median(pos_errors):.3f}mm')
    ax4.axvline(np.percentile(pos_errors, 95), color='orange', linestyle='--', linewidth=2,
                label=f'95e %ile: {np.percentile(pos_errors, 95):.3f}mm')
    ax4.set_xlabel('Erreur Position (mm)', fontsize=10)
    ax4.set_ylabel('Pourcentage Cumulatif (%)', fontsize=10)
    ax4.set_title('Distribution Cumulative Position', fontsize=11, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, np.percentile(pos_errors, 99)])
    
    # Subplot 5: Box plot comparison
    ax5 = fig.add_subplot(2, 3, 5)
    bp = ax5.boxplot([pos_errors, ori_errors*10], labels=['Position (mm)', 'Orientation (°×10)'],
                     patch_artist=True, notch=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax5.set_ylabel('Erreur', fontsize=10)
    ax5.set_title('Box Plot Comparatif', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Subplot 6: Summary statistics panel
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"{config_name} - Validation Aller-Retour FK→IK→FK\n"
    summary_text += "="*60 + "\n\n"
    summary_text += f"Nombre de Tests: {len(results)}\n\n"
    
    summary_text += "POSITION:\n"
    summary_text += f"  Moyenne:  {np.mean(pos_errors):8.3f} mm\n"
    summary_text += f"  Médiane:  {np.median(pos_errors):8.3f} mm\n"
    summary_text += f"  Écart-type: {np.std(pos_errors):6.3f} mm\n"
    summary_text += f"  Maximum:  {np.max(pos_errors):8.3f} mm\n"
    summary_text += f"  95e %ile: {np.percentile(pos_errors, 95):8.3f} mm\n\n"
    
    summary_text += "ORIENTATION:\n"
    summary_text += f"  Moyenne:  {np.mean(ori_errors):8.5f}°\n"
    summary_text += f"  Médiane:  {np.median(ori_errors):8.5f}°\n"
    summary_text += f"  Écart-type: {np.std(ori_errors):6.5f}°\n"
    summary_text += f"  Maximum:  {np.max(ori_errors):8.5f}°\n\n"
    
    # Success criteria
    excellent = sum(1 for e in pos_errors if e < 1.0)
    good = sum(1 for e in pos_errors if 1.0 <= e < 5.0)
    acceptable = sum(1 for e in pos_errors if 5.0 <= e < 10.0)
    
    summary_text += "QUALITÉ:\n"
    summary_text += f"  Excellent (<1mm):   {excellent:3d} ({excellent/len(results)*100:5.1f}%)\n"
    summary_text += f"  Bon (1-5mm):        {good:3d} ({good/len(results)*100:5.1f}%)\n"
    summary_text += f"  Acceptable (5-10mm): {acceptable:3d} ({acceptable/len(results)*100:5.1f}%)\n\n"
    
    # Overall assessment
    if np.mean(pos_errors) < 1.0:
        assessment = "✅ EXCELLENT - Production Ready"
        color = 'green'
    elif np.mean(pos_errors) < 5.0:
        assessment = "✅ GOOD - Suitable for most tasks"
        color = 'darkgreen'
    elif np.mean(pos_errors) < 10.0:
        assessment = "⚠️ ACCEPTABLE - Review use cases"
        color = 'orange'
    else:
        assessment = "❌ POOR - Needs improvement"
        color = 'red'
    
    summary_text += f"ÉVALUATION:\n  {assessment}"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add assessment badge
    ax6.text(0.5, 0.15, assessment.split('-')[0].strip(), 
            transform=ax6.transAxes,
            fontsize=14, fontweight='bold', color=color,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor=color, linewidth=3))
    
    plt.suptitle(f'{config_name} - Validation Fiabilité Solveur IK (100 Tests Aléatoires)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_image:
        filename = f'{config_name}_roundtrip_validation.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n💾 Saved: {filename}")
    
    plt.show()


def visualize_3d_error_distribution(results, config, save_image=False):
    """Visualize errors in 3D workspace."""
    fig = plt.figure(figsize=(16, 8))
    
    # 3D scatter of target positions colored by error
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    x_targets = [r['x_target'][0] for r in results]
    y_targets = [r['x_target'][1] for r in results]
    z_targets = [r['x_target'][2] for r in results]
    pos_errors = [r['pos_error_mm'] for r in results]
    
    scatter = ax1.scatter(x_targets, y_targets, z_targets, 
                         c=pos_errors, cmap='RdYlGn_r', s=50,
                         alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax1.scatter(0, 0, 0, c='black', s=200, marker='s', label='Base')
    ax1.set_xlabel('X (m)', fontsize=10)
    ax1.set_ylabel('Y (m)', fontsize=10)
    ax1.set_zlabel('Z (m)', fontsize=10)
    ax1.set_title('Erreurs Position dans Espace de Travail', fontsize=12, fontweight='bold')
    ax1.legend()
    cbar = plt.colorbar(scatter, ax=ax1, pad=0.1, shrink=0.8)
    cbar.set_label('Erreur Position (mm)', fontsize=9)
    ax1.view_init(elev=20, azim=45)
    ax1.set_box_aspect([1, 1, 1])
    
    # 3D scatter colored by orientation error
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    ori_errors = [r['ori_error_deg'] for r in results]
    
    scatter2 = ax2.scatter(x_targets, y_targets, z_targets,
                          c=ori_errors, cmap='plasma', s=50,
                          alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax2.scatter(0, 0, 0, c='black', s=200, marker='s', label='Base')
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    ax2.set_zlabel('Z (m)', fontsize=10)
    ax2.set_title('Erreurs Orientation dans Espace de Travail', fontsize=12, fontweight='bold')
    ax2.legend()
    cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.1, shrink=0.8)
    cbar2.set_label('Erreur Orientation (°)', fontsize=9)
    ax2.view_init(elev=20, azim=45)
    ax2.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    
    if save_image:
        filename = f'SET_D_roundtrip_3d_workspace.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {filename}")
    
    plt.show()


def visualize_sample_configurations(results, config, n_samples=6, save_image=False):
    """Visualize sample robot configurations from roundtrip tests."""
    fig = plt.figure(figsize=(18, 12))
    
    # Select samples: best, worst, and random
    sorted_by_pos = sorted(results, key=lambda r: r['pos_error_mm'])
    samples = [
        sorted_by_pos[0],  # Best
        sorted_by_pos[len(sorted_by_pos)//4],
        sorted_by_pos[len(sorted_by_pos)//2],
        sorted_by_pos[3*len(sorted_by_pos)//4],
        sorted_by_pos[-1],  # Worst
        results[np.random.randint(len(results))],  # Random
    ]
    
    for idx, result in enumerate(samples[:n_samples]):
        ax = fig.add_subplot(2, 3, idx+1, projection='3d')
        
        # Compute robot link positions
        q = result['q_solved']
        T = np.eye(4)
        positions = [T[:3, 3].copy()]
        
        for i, joint in enumerate(config):
            jt = joint["type"]
            theta = np.deg2rad(q[i]) if "rot" in jt else 0.0
            d = joint["d"] if "rot" in jt else joint["d"] + q[i] / 1000.0
            A = dh2.dh_matrix(theta, d, joint["a"], joint["alpha"])
            T = T @ A
            positions.append(T[:3, 3].copy())
        
        positions = np.array(positions)
        
        # Plot robot
        for i in range(len(positions) - 1):
            ax.plot([positions[i, 0], positions[i+1, 0]],
                   [positions[i, 1], positions[i+1, 1]],
                   [positions[i, 2], positions[i+1, 2]],
                   'b-', linewidth=2)
        
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='blue', s=30)
        ax.scatter(0, 0, 0, c='black', s=100, marker='s')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                  c='red', s=150, marker='*')
        
        # Target vs reached
        ax.scatter(*result['x_target'], c='red', s=80, marker='X', alpha=0.5, label='Cible')
        ax.scatter(*result['x_check'], c='lime', s=60, marker='o', label='Atteint')
        
        # Title with errors
        title = f"Test {idx+1}\n"
        title += f"Pos: {result['pos_error_mm']:.2f}mm, Ori: {result['ori_error_deg']:.4f}°"
        ax.set_title(title, fontsize=9)
        
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        if idx == 0:
            ax.legend(fontsize=7)
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1, 1, 1])
    
    plt.suptitle('SET_D - Échantillons de Configurations (Meilleur → Pire)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_image:
        plt.savefig('SET_D_roundtrip_samples.png', dpi=150, bbox_inches='tight')
        print("💾 Saved: SET_D_roundtrip_samples.png")
    
    plt.show()


def main():
    print("🤖 VALIDATION SET_D - TEST ALLER-RETOUR FK→IK→FK")
    print("Mesure la fiabilité du solveur avec 100 configurations aléatoires\n")
    
    # Get SET_D configuration
    catalog = get_module_catalog()
    module_set = catalog['SET_D_EXTENDED_REACH']
    config = module_set.config
    
    print(f"Configuration testée: {module_set.name}")
    print(f"Modules: {[j['type'] for j in config]}")
    print()
    
    # Run 100 roundtrip tests
    results = run_roundtrip_tests(config, n_tests=100)
    
    # Analyze results
    stats = analyze_results(results)
    
    # Visualizations
    print(f"\n🎨 Génération des visualisations...")
    
    # Visualization 1: Statistical analysis
    visualize_results(results, config_name="SET_D_Extended_Reach", save_image=True)
    
    # Visualization 2: 3D workspace error distribution
    visualize_3d_error_distribution(results, config, save_image=True)
    
    # Visualization 3: Sample configurations
    visualize_sample_configurations(results, config, n_samples=6, save_image=True)
    
    print("\n" + "="*70)
    print("✅ VALIDATION COMPLÈTE")
    print("="*70)
    
    # Final verdict
    if stats['pos_mean'] < 1.0:
        print("\n🎉 RÉSULTAT: EXCELLENT")
        print(f"   Erreur position moyenne {stats['pos_mean']:.3f}mm < 1mm")
        print(f"   Erreur orientation moyenne {stats['ori_mean']:.5f}° ≈ 0°")
        print("\n✅ SET_D est VALIDÉ pour usage production")
        print("✅ Fiabilité solveur confirmée")
    elif stats['pos_mean'] < 5.0:
        print("\n✅ RÉSULTAT: BON")
        print(f"   Erreur position moyenne {stats['pos_mean']:.3f}mm < 5mm")
        print("✅ Approprié pour la plupart des applications")
    else:
        print("\n⚠️ RÉSULTAT: À AMÉLIORER")
        print(f"   Erreur position moyenne {stats['pos_mean']:.3f}mm")
    
    print("\nFichiers générés:")
    print("  - SET_D_Extended_Reach_roundtrip_validation.png")
    print("  - SET_D_roundtrip_3d_workspace.png")
    print("  - SET_D_roundtrip_samples.png")


if __name__ == "__main__":
    main()

