#!/usr/bin/env python3
"""
3D Visualization of Module Catalog IK Results
Shows robot configurations, workspace, target poses, and achieved poses.
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

from module_catalog import (
    get_module_catalog,
    get_workspace_test_poses,
    estimate_reach,
    is_reachable,
    euler_to_rotation_matrix,
)
from dls_ik_baseline import (
    forward_kinematics,
    inverse_kinematics_dls,
    rotation_error_cross,
)


def compute_robot_points(config, q_deg):
    """Compute all link positions for visualization."""
    T = np.eye(4)
    positions = [T[:3, 3].copy()]
    orientations = [T[:3, :3].copy()]
    
    for i, joint in enumerate(config):
        jt = joint["type"]
        theta = np.deg2rad(q_deg[i]) if "rot" in jt else 0.0
        d = joint["d"] if "rot" in jt else joint["d"] + q_deg[i] / 1000.0
        A = dh2.dh_matrix(theta, d, joint["a"], joint["alpha"])
        T = T @ A
        positions.append(T[:3, 3].copy())
        orientations.append(T[:3, :3].copy())
    
    return np.array(positions), orientations


def plot_robot_config(ax, config, q, color='blue', alpha=1.0, label=None):
    """Plot a single robot configuration."""
    positions, orientations = compute_robot_points(config, q)
    
    # Plot links
    for i in range(len(positions) - 1):
        ax.plot([positions[i, 0], positions[i+1, 0]],
                [positions[i, 1], positions[i+1, 1]],
                [positions[i, 2], positions[i+1, 2]],
                color=color, linewidth=2, alpha=alpha)
    
    # Plot joints
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=color, s=30, alpha=alpha, marker='o')
    
    # Plot end effector with larger marker
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
               c=color, s=100, alpha=alpha, marker='*', label=label)
    
    # Plot end effector orientation
    R_end = orientations[-1]
    pos_end = positions[-1]
    axis_len = 0.05
    colors_axes = ['red', 'green', 'blue']
    for i, (axis, col) in enumerate(zip(R_end.T, colors_axes)):
        ax.quiver(pos_end[0], pos_end[1], pos_end[2],
                  axis[0], axis[1], axis[2],
                  length=axis_len, color=col, alpha=alpha*0.7, arrow_length_ratio=0.3)
    
    return positions


def plot_workspace_sphere(ax, reach, alpha=0.1):
    """Plot workspace boundary as transparent sphere."""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = reach * np.outer(np.cos(u), np.sin(v))
    y = reach * np.outer(np.sin(u), np.sin(v))
    z = reach * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, color='cyan', alpha=alpha, linewidth=0)


def visualize_catalog_set(set_id, catalog, save_image=False):
    """Visualize a catalog set with its test results."""
    module_set = catalog[set_id]
    config = module_set.config
    reach = estimate_reach(config)
    
    # Generate test poses
    test_poses = get_workspace_test_poses(reach)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Main 3D view
    ax_main = fig.add_subplot(2, 2, 1, projection='3d')
    ax_main.set_title(f"{module_set.name}\nWorkspace & IK Solutions", fontsize=12, fontweight='bold')
    
    # Plot workspace boundary
    plot_workspace_sphere(ax_main, reach, alpha=0.05)
    
    # Plot base
    ax_main.scatter(0, 0, 0, c='black', s=200, marker='s', label='Base')
    
    # Solve IK for each test pose and visualize
    colors = ['blue', 'green', 'orange', 'purple']
    results_summary = []
    
    for i, (pose, color) in enumerate(zip(test_poses, colors)):
        target_pos = np.array(pose['position'])
        target_euler = pose['euler']
        
        # Check reachability
        if not is_reachable(config, target_pos):
            continue
        
        # Solve IK
        R_target = euler_to_rotation_matrix(*target_euler)
        
        try:
            # Multi-restart
            best_q = None
            best_err = np.inf
            for q_init in [np.zeros(len(config)), 
                          np.random.uniform(-20, 20, len(config))]:
                q = inverse_kinematics_dls(config, target_pos, R_target, 
                                          q_init=q_init, max_iter=1000, lam=0.01)
                x_check, R_check = forward_kinematics(config, q)
                err = np.linalg.norm(target_pos - x_check)
                if err < best_err:
                    best_err = err
                    best_q = q
            
            q_solution = best_q
            
            # Verify
            x_reached, R_reached = forward_kinematics(config, q_solution)
            pos_err = np.linalg.norm(target_pos - x_reached) * 1000
            ori_err = np.linalg.norm(rotation_error_cross(R_reached, R_target)) * 180 / np.pi
            
            # Plot robot in this configuration
            positions = plot_robot_config(ax_main, config, q_solution, 
                                        color=color, alpha=0.6,
                                        label=f"{pose['name'][:15]}")
            
            # Plot target position (red) vs reached position (green)
            ax_main.scatter(*target_pos, c='red', s=150, marker='X', 
                          edgecolors='black', linewidths=2, alpha=0.8)
            ax_main.scatter(*x_reached, c='lime', s=100, marker='o', alpha=0.8)
            
            # Plot target orientation
            for j, (axis, col) in enumerate(zip(R_target.T, ['red', 'green', 'blue'])):
                ax_main.quiver(target_pos[0], target_pos[1], target_pos[2],
                              axis[0], axis[1], axis[2],
                              length=0.08, color=col, alpha=0.5, 
                              arrow_length_ratio=0.2, linewidth=2)
            
            results_summary.append({
                'name': pose['name'],
                'pos_err': pos_err,
                'ori_err': ori_err,
            })
        
        except Exception as e:
            print(f"Failed {pose['name']}: {e}")
    
    # Formatting
    ax_main.set_xlabel('X (m)', fontsize=10)
    ax_main.set_ylabel('Y (m)', fontsize=10)
    ax_main.set_zlabel('Z (m)', fontsize=10)
    ax_main.legend(loc='upper left', fontsize=8)
    ax_main.set_box_aspect([1, 1, 1])
    
    # Set axis limits based on reach
    lim = reach * 0.8
    ax_main.set_xlim([-lim, lim])
    ax_main.set_ylim([-lim, lim])
    ax_main.set_zlim([0, lim*1.5])
    ax_main.view_init(elev=20, azim=45)
    
    # Add grid
    ax_main.grid(True, alpha=0.3)
    
    # Subplot 2: Performance metrics
    ax_perf = fig.add_subplot(2, 2, 2)
    ax_perf.axis('off')
    
    perf_text = f"{module_set.name}\n"
    perf_text += "="*50 + "\n\n"
    perf_text += f"Portée Estimée: {reach:.3f} m\n"
    perf_text += f"DDL: {len(config)}\n\n"
    
    perf_text += "Performance Validée:\n"
    perf_text += "-"*50 + "\n"
    if results_summary:
        avg_pos = np.mean([r['pos_err'] for r in results_summary])
        max_pos = np.max([r['pos_err'] for r in results_summary])
        avg_ori = np.mean([r['ori_err'] for r in results_summary])
        max_ori = np.max([r['ori_err'] for r in results_summary])
        
        perf_text += f"Position:\n"
        perf_text += f"  Moyenne: {avg_pos:.2f} mm\n"
        perf_text += f"  Maximum: {max_pos:.2f} mm\n\n"
        perf_text += f"Orientation:\n"
        perf_text += f"  Moyenne: {avg_ori:.3f}°\n"
        perf_text += f"  Maximum: {max_ori:.3f}°\n\n"
        
        perf_text += "Résultats par Pose:\n"
        perf_text += "-"*50 + "\n"
        for r in results_summary:
            perf_text += f"• {r['name'][:25]}\n"
            perf_text += f"  Pos: {r['pos_err']:.2f}mm, Ori: {r['ori_err']:.2f}°\n"
    
    ax_perf.text(0.1, 0.95, perf_text, transform=ax_perf.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Subplot 3: Top view (XY plane)
    ax_top = fig.add_subplot(2, 2, 3)
    ax_top.set_title("Vue de Dessus (Plan XY)", fontsize=10)
    ax_top.set_xlabel('X (m)')
    ax_top.set_ylabel('Y (m)')
    ax_top.grid(True, alpha=0.3)
    ax_top.set_aspect('equal')
    
    # Draw workspace circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax_top.plot(reach * np.cos(theta), reach * np.sin(theta), 
                'c--', alpha=0.3, linewidth=2, label='Portée Max')
    
    # Plot robot configurations (top view)
    for i, (pose, color) in enumerate(zip(test_poses, colors)):
        target_pos = np.array(pose['position'])
        if not is_reachable(config, target_pos):
            continue
        
        R_target = euler_to_rotation_matrix(*pose['euler'])
        try:
            q = inverse_kinematics_dls(config, target_pos, R_target, max_iter=500, lam=0.01)
            positions, _ = compute_robot_points(config, q)
            
            # Plot links
            ax_top.plot(positions[:, 0], positions[:, 1], color=color, linewidth=2, alpha=0.6)
            ax_top.scatter(positions[-1, 0], positions[-1, 1], 
                          c=color, s=100, marker='*', label=pose['name'][:15])
            
            # Plot target
            ax_top.scatter(target_pos[0], target_pos[1], 
                          c='red', s=80, marker='X', alpha=0.5)
        except:
            pass
    
    ax_top.scatter(0, 0, c='black', s=100, marker='s', label='Base')
    ax_top.legend(fontsize=7, loc='upper right')
    ax_top.set_xlim([-reach, reach])
    ax_top.set_ylim([-reach, reach])
    
    # Subplot 4: Side view (XZ plane)
    ax_side = fig.add_subplot(2, 2, 4)
    ax_side.set_title("Vue de Côté (Plan XZ)", fontsize=10)
    ax_side.set_xlabel('X (m)')
    ax_side.set_ylabel('Z (m)')
    ax_side.grid(True, alpha=0.3)
    ax_side.set_aspect('equal')
    
    # Draw workspace semi-circle
    theta = np.linspace(0, np.pi, 100)
    ax_side.plot(reach * np.cos(theta), reach * np.sin(theta), 
                'c--', alpha=0.3, linewidth=2)
    
    # Plot robot configurations (side view)
    for i, (pose, color) in enumerate(zip(test_poses, colors)):
        target_pos = np.array(pose['position'])
        if not is_reachable(config, target_pos):
            continue
        
        R_target = euler_to_rotation_matrix(*pose['euler'])
        try:
            q = inverse_kinematics_dls(config, target_pos, R_target, max_iter=500, lam=0.01)
            positions, _ = compute_robot_points(config, q)
            
            # Plot links
            ax_side.plot(positions[:, 0], positions[:, 2], color=color, linewidth=2, alpha=0.6)
            ax_side.scatter(positions[-1, 0], positions[-1, 2], 
                           c=color, s=100, marker='*')
            
            # Plot target
            ax_side.scatter(target_pos[0], target_pos[2], 
                           c='red', s=80, marker='X', alpha=0.5)
        except:
            pass
    
    ax_side.scatter(0, 0, c='black', s=100, marker='s')
    ax_side.set_xlim([0, reach])
    ax_side.set_ylim([0, reach*1.2])
    
    plt.tight_layout()
    
    if save_image:
        filename = f"catalog_{set_id}_visualization.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    plt.show()


def visualize_comparison_all_sets(catalog, save_image=False):
    """Compare all catalog sets in a single comprehensive view."""
    fig = plt.figure(figsize=(20, 12))
    
    sets_to_compare = ['SET_D_EXTENDED_REACH', 'SET_A_FULL_6D', 'SET_E_COMPACT']
    bar_colors = ['blue', 'green', 'orange', 'purple']
    
    for idx, set_id in enumerate(sets_to_compare):
        module_set = catalog[set_id]
        config = module_set.config
        reach = estimate_reach(config)
        
        # 3D subplot for each set
        ax = fig.add_subplot(2, 3, idx+1, projection='3d')
        ax.set_title(module_set.name.split('-')[0].strip(), fontsize=11, fontweight='bold')
        
        # Workspace
        plot_workspace_sphere(ax, reach, alpha=0.05)
        ax.scatter(0, 0, 0, c='black', s=100, marker='s')
        
        # Test one representative pose
        test_poses = get_workspace_test_poses(reach)
        target_pos = np.array(test_poses[0]['position'])  # Vertical approach
        target_euler = test_poses[0]['euler']
        R_target = euler_to_rotation_matrix(*target_euler)
        
        try:
            q = inverse_kinematics_dls(config, target_pos, R_target, max_iter=1000, lam=0.01)
            positions = plot_robot_config(ax, config, q, color='blue', alpha=0.8)
            
            # Target vs reached
            x_reached, R_reached = forward_kinematics(config, q)
            ax.scatter(*target_pos, c='red', s=150, marker='X', label='Cible')
            ax.scatter(*x_reached, c='lime', s=100, marker='o', label='Atteint')
            
            # Error
            pos_err = np.linalg.norm(target_pos - x_reached) * 1000
            ax.text2D(0.02, 0.02, f"Err: {pos_err:.2f}mm", 
                     transform=ax.transAxes, fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        except:
            pass
        
        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Y (m)', fontsize=8)
        ax.set_zlabel('Z (m)', fontsize=8)
        ax.legend(fontsize=7)
        lim = reach * 0.7
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([0, lim*1.5])
        ax.view_init(elev=25, azim=45)
        ax.set_box_aspect([1, 1, 1])
        
        # Bar chart of errors for this set (subplot below)
        ax_bar = fig.add_subplot(2, 3, idx+4)
        ax_bar.set_title(f"Erreurs de Position", fontsize=10)
        
        pose_names = []
        pos_errors = []
        
        for pose in test_poses:
            target_pos = np.array(pose['position'])
            if not is_reachable(config, target_pos):
                continue
            
            R_target = euler_to_rotation_matrix(*pose['euler'])
            try:
                q = inverse_kinematics_dls(config, target_pos, R_target, max_iter=1000, lam=0.01)
                x_check, _ = forward_kinematics(config, q)
                err = np.linalg.norm(target_pos - x_check) * 1000
                pose_names.append(pose['name'].split('(')[0].strip()[:12])
                pos_errors.append(err)
            except:
                pass
        
        if pose_names:
            bars = ax_bar.bar(range(len(pose_names)), pos_errors, color=bar_colors[:len(pose_names)])
            ax_bar.set_xticks(range(len(pose_names)))
            ax_bar.set_xticklabels(pose_names, rotation=45, ha='right', fontsize=8)
            ax_bar.set_ylabel('Erreur (mm)', fontsize=9)
            ax_bar.grid(True, alpha=0.3, axis='y')
            ax_bar.set_ylim([0, max(pos_errors) * 1.2])
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, pos_errors)):
                ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=7)
    
    plt.suptitle('Comparaison des Ensembles du Catalogue - Résultats IK 6D Validés',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_image:
        plt.savefig('catalog_comparison_all_sets.png', dpi=150, bbox_inches='tight')
        print("Saved: catalog_comparison_all_sets.png")
    
    plt.show()


def visualize_single_pose_multiple_robots(catalog, save_image=False):
    """Show how different catalog sets solve the same target pose."""
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Common target
    target_pos = np.array([0.35, 0.10, 0.20])
    target_euler = [0, 0, 45]
    R_target = euler_to_rotation_matrix(*target_euler)
    
    sets = ['SET_D_EXTENDED_REACH', 'SET_A_FULL_6D', 'SET_E_COMPACT']
    colors = ['blue', 'green', 'orange']
    
    max_reach = 0
    
    for set_id, color in zip(sets, colors):
        module_set = catalog[set_id]
        config = module_set.config
        reach = estimate_reach(config)
        max_reach = max(max_reach, reach)
        
        # Scale target to this robot's workspace
        scaled_target = target_pos * (reach * 0.65 / np.linalg.norm(target_pos))
        
        if is_reachable(config, scaled_target):
            try:
                q = inverse_kinematics_dls(config, scaled_target, R_target, max_iter=1000, lam=0.01)
                x_reached, R_reached = forward_kinematics(config, q)
                pos_err = np.linalg.norm(scaled_target - x_reached) * 1000
                
                # Plot
                plot_robot_config(ax, config, q, color=color, alpha=0.6,
                                label=f"{module_set.name.split('-')[0]} ({pos_err:.1f}mm)")
                
                # Target marker
                ax.scatter(*scaled_target, c='red', s=100, marker='X', alpha=0.3)
            except Exception as e:
                print(f"Failed {set_id}: {e}")
    
    # Plot workspace boundary of largest robot
    plot_workspace_sphere(ax, max_reach, alpha=0.03)
    
    ax.scatter(0, 0, 0, c='black', s=200, marker='s', label='Base', zorder=100)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Comparaison: Même Cible, Différents Ensembles de Modules',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)
    
    lim = max_reach * 0.7
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([0, lim*1.5])
    
    plt.tight_layout()
    
    if save_image:
        plt.savefig('catalog_same_target_comparison.png', dpi=150, bbox_inches='tight')
        print("Saved: catalog_same_target_comparison.png")
    
    plt.show()


def main():
    print("🎨 VISUALISATION 3D - RÉSULTATS CATALOGUE MODULES\n")
    
    catalog = get_module_catalog()
    
    # Option 1: Visualize each set individually
    print("Génération des visualisations pour chaque ensemble...\n")
    
    for set_id in ['SET_D_EXTENDED_REACH', 'SET_A_FULL_6D', 'SET_E_COMPACT']:
        print(f"Visualisation: {catalog[set_id].name}")
        visualize_catalog_set(set_id, catalog, save_image=True)
    
    # Option 2: Comparison view
    print("\nGénération de la vue comparative...")
    visualize_comparison_all_sets(catalog, save_image=True)
    
    # Option 3: Same target, different robots
    print("\nGénération de la comparaison même cible...")
    visualize_single_pose_multiple_robots(catalog, save_image=True)
    
    print("\n✅ Visualisations générées!")
    print("Fichiers créés:")
    print("  - catalog_*_visualization.png (vues détaillées)")
    print("  - catalog_comparison_all_sets.png (comparaison)")
    print("  - catalog_same_target_comparison.png (même cible)")


if __name__ == "__main__":
    import matplotlib
    # Use Agg backend for headless, or comment out for interactive
    # matplotlib.use('Agg')
    main()

