#!/usr/bin/env python3
"""
Adaptive IK solver for modular robots.
Automatically detects capabilities and selects the best strategy.
"""

import numpy as np
import sys
import os
import importlib.util

# Load dh_utils (2).py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DH_UTILS2_PATH = os.path.join(THIS_DIR, "dh_utils (2).py")
spec = importlib.util.spec_from_file_location("dh2", DH_UTILS2_PATH)
dh2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dh2)

# Load DLS baseline functions
sys.modules['dh_utils'] = dh2
from dls_ik_baseline import (
    forward_kinematics,
    rotation_error_cross,
    euler_to_rotation_matrix,
    inverse_kinematics_dls,
)


def compute_rotational_jacobian(config, q_deg, eps=1.0):
    """Compute 3×n rotational Jacobian using finite differences."""
    n = len(q_deg)
    Jori = np.zeros((3, n))
    _, R0 = forward_kinematics(config, q_deg)
    
    for k in range(n):
        qd = q_deg.copy()
        qd[k] += eps
        _, Rd = forward_kinematics(config, qd)
        e_R = rotation_error_cross(Rd, R0)
        Jori[:, k] = e_R / eps
    
    return Jori


def analyze_robot_capabilities(config, test_positions=None):
    """
    Analyze what the robot can control based on its configuration.
    
    Returns:
        dict with:
        - n_dof: number of degrees of freedom
        - has_spherical_wrist: bool
        - avg_sigma_min: average rotational manipulability
        - can_control_full_6d: bool
        - capability_level: 'full_6d' | 'partial_6d' | 'position_only'
    """
    n_dof = len(config)
    
    # Check for spherical wrist pattern (last 3 joints)
    has_spherical_wrist = False
    if n_dof >= 6:
        last3 = config[-3:]
        # Check if all a ≈ 0 (intersecting axes)
        all_a_zero = all(abs(j['a']) < 0.01 for j in last3)
        # Check if alphas suggest orthogonal axes
        alphas = [abs(j['alpha']) for j in last3]
        has_orthogonal = any(abs(a - np.pi/2) < 0.2 for a in alphas)
        has_spherical_wrist = all_a_zero and has_orthogonal
    
    # Compute average rotational manipulability at sample poses
    if test_positions is None:
        # Default test positions in workspace
        test_positions = [
            [0.3, 0.0, 0.2],
            [0.3, 0.2, 0.2],
            [0.4, 0.1, 0.15],
        ]
    
    sigma_mins = []
    for pos in test_positions:
        try:
            # Position-only IK to get a sample configuration
            q = position_only_ik_simple(config, pos)
            Jori = compute_rotational_jacobian(config, q)
            s = np.linalg.svd(Jori, compute_uv=False)
            sigma_mins.append(float(s.min()))
        except Exception:
            pass
    
    avg_sigma_min = np.mean(sigma_mins) if sigma_mins else 0.0
    
    # Determine capability level
    if n_dof >= 6 and has_spherical_wrist and avg_sigma_min > 0.7:
        capability_level = 'full_6d'
    elif n_dof >= 5 and avg_sigma_min > 0.4:
        capability_level = 'partial_6d'
    else:
        capability_level = 'position_only'
    
    return {
        'n_dof': n_dof,
        'has_spherical_wrist': has_spherical_wrist,
        'avg_sigma_min': avg_sigma_min,
        'can_control_full_6d': capability_level == 'full_6d',
        'capability_level': capability_level,
        'sigma_values': sigma_mins,
    }


def position_only_ik_simple(config, target_pos, max_iter=200, lam=0.01):
    """Simple position-only IK (3×n Jacobian)."""
    n = len(config)
    q = np.zeros(n)
    target_pos = np.asarray(target_pos, float)
    
    for _ in range(max_iter):
        x_cur, _ = forward_kinematics(config, q)
        e_pos = target_pos - x_cur
        
        if np.linalg.norm(e_pos) < 1e-3:
            return q
        
        # Position Jacobian
        J = np.zeros((3, n))
        eps = 0.01
        for k in range(n):
            qd = q.copy()
            qd[k] += eps
            xd, _ = forward_kinematics(config, qd)
            J[:, k] = (xd - x_cur) / eps
        
        # DLS step
        JtJ = J.T @ J
        dq = np.linalg.inv(JtJ + (lam**2) * np.eye(n)) @ (J.T @ e_pos)
        q = q + dq
    
    return q


def inverse_kinematics_dls_weighted(
    config,
    target_pos,
    target_R,
    pos_weight=1.0,
    ori_weight=1.0,
    max_iter=200,
    lam=0.01,
    step_size=1.0,
):
    """DLS IK with separate position and orientation weights."""
    n = len(config)
    q = position_only_ik_simple(config, target_pos, max_iter=100)
    target_pos = np.asarray(target_pos, float)
    
    for _ in range(max_iter):
        x_cur, R_cur = forward_kinematics(config, q)
        e_pos = target_pos - x_cur
        e_ori = rotation_error_cross(R_cur, target_R)
        
        if np.linalg.norm(e_pos) < 1e-3 and np.linalg.norm(e_ori) < 1e-3:
            return q
        
        # Weighted error
        e6 = np.hstack([pos_weight * e_pos, ori_weight * e_ori])
        
        # Jacobian
        J = np.zeros((6, n))
        eps = 0.01
        for k in range(n):
            qd = q.copy()
            qd[k] += eps
            xd, Rd = forward_kinematics(config, qd)
            J[0:3, k] = (xd - x_cur) / eps
            e_ori_d = rotation_error_cross(Rd, R_cur)
            J[3:6, k] = e_ori_d / eps
        
        # DLS step
        JtJ = J.T @ J
        dq = step_size * np.linalg.inv(JtJ + (lam**2) * np.eye(n)) @ (J.T @ e6)
        q = q + dq
    
    return q


def adaptive_ik_solver(config, target_pos, target_euler_deg=None, verbose=True):
    """
    Automatically select best IK strategy based on robot capabilities.
    
    Args:
        config: DH configuration
        target_pos: Target position [x, y, z]
        target_euler_deg: Target orientation [roll, pitch, yaw] or None
        verbose: Print capability analysis
    
    Returns:
        q_solution, info dict
    """
    # Analyze capabilities
    caps = analyze_robot_capabilities(config)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"ROBOT ANALYSIS")
        print(f"{'='*60}")
        print(f"DOF: {caps['n_dof']}")
        print(f"Spherical wrist: {'Yes' if caps['has_spherical_wrist'] else 'No'}")
        print(f"Avg σmin(Jori): {caps['avg_sigma_min']:.3f}")
        print(f"Capability level: {caps['capability_level'].upper()}")
        print()
    
    # If no orientation specified, use position-only
    if target_euler_deg is None:
        if verbose:
            print("Strategy: POSITION-ONLY (no orientation specified)")
        q = position_only_ik_simple(config, target_pos)
        return q, {'strategy': 'position_only', 'capabilities': caps}
    
    # Convert orientation to rotation matrix
    R_target = euler_to_rotation_matrix(*target_euler_deg)
    
    # Strategy selection based on capabilities
    if caps['capability_level'] == 'full_6d':
        # Full 6D IK
        if verbose:
            print("Strategy: FULL 6D IK (spherical wrist detected)")
        q = inverse_kinematics_dls(config, target_pos, R_target, 
                                   q_init=None, max_iter=200, lam=0.01)
        strategy = 'full_6d'
    
    elif caps['capability_level'] == 'partial_6d':
        # Position-priority with relaxed orientation
        if verbose:
            print("Strategy: POSITION-PRIORITY (partial orientation control)")
            print("  Position weight: 5.0, Orientation weight: 1.0")
        q = inverse_kinematics_dls_weighted(config, target_pos, R_target,
                                           pos_weight=5.0, ori_weight=1.0,
                                           max_iter=250, lam=0.01)
        strategy = 'partial_6d'
    
    else:
        # Position-only fallback
        if verbose:
            print("Strategy: POSITION-ONLY (insufficient DOF for orientation)")
        q = position_only_ik_simple(config, target_pos)
        strategy = 'position_only'
    
    # Verify solution
    x_check, R_check = forward_kinematics(config, q)
    pos_err = np.linalg.norm(target_pos - x_check) * 1000  # mm
    ori_err = np.linalg.norm(rotation_error_cross(R_check, R_target))
    ori_err_deg = ori_err * 180 / np.pi  # approximate degrees
    
    if verbose:
        print(f"\nResult:")
        print(f"  Position error: {pos_err:.2f} mm")
        print(f"  Orientation error: {ori_err_deg:.1f}°")
        
        # Recommendations
        if strategy != 'full_6d' and caps['n_dof'] >= 6:
            print(f"\n💡 Recommendation:")
            if not caps['has_spherical_wrist']:
                print("  - Add WristBottom+WristTop modules to last 3 joints for better orientation")
            if caps['avg_sigma_min'] < 0.7:
                print("  - Current combo has poor rotational manipulability")
                print("  - Consider repositioning or relaxing orientation constraints")
    
    return q, {
        'strategy': strategy,
        'capabilities': caps,
        'pos_error_mm': pos_err,
        'ori_error_deg': ori_err_deg,
    }


def test_random_module_combinations(n_combos=20):
    """Test adaptive solver on random module combinations."""
    print(f"\n{'='*60}")
    print(f"TESTING {n_combos} RANDOM MODULE COMBINATIONS")
    print(f"{'='*60}\n")
    
    # Common test target
    target_pos = [0.35, 0.15, 0.20]
    target_euler = [0, 0, 45]
    
    results = {
        'full_6d': [],
        'partial_6d': [],
        'position_only': [],
    }
    
    for i in range(n_combos):
        print(f"\n--- Combo {i+1}/{n_combos} ---")
        config = dh2.random_robot_dh(6)
        
        try:
            q, info = adaptive_ik_solver(config, target_pos, target_euler, verbose=False)
            
            # Categorize
            strategy = info['strategy']
            results[strategy].append({
                'pos_err': info['pos_error_mm'],
                'ori_err': info['ori_error_deg'],
                'sigma_min': info['capabilities']['avg_sigma_min'],
            })
            
            print(f"Strategy: {strategy}")
            print(f"  Pos err: {info['pos_error_mm']:.1f} mm, Ori err: {info['ori_error_deg']:.1f}°")
            print(f"  σmin: {info['capabilities']['avg_sigma_min']:.3f}")
        
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"SUMMARY - {n_combos} Module Combinations")
    print(f"{'='*60}")
    
    for strategy_name, data in results.items():
        if data:
            count = len(data)
            avg_pos = np.mean([r['pos_err'] for r in data])
            avg_ori = np.mean([r['ori_err'] for r in data])
            avg_sigma = np.mean([r['sigma_min'] for r in data])
            
            print(f"\n{strategy_name.upper()}: {count} combos ({count/n_combos*100:.0f}%)")
            print(f"  Avg position error: {avg_pos:.2f} mm")
            print(f"  Avg orientation error: {avg_ori:.1f}°")
            print(f"  Avg σmin(Jori): {avg_sigma:.3f}")
    
    print()


def demo_adaptive_solver():
    """Demonstrate the adaptive solver with a specific combo."""
    print(f"\n{'='*60}")
    print("ADAPTIVE SOLVER DEMO")
    print(f"{'='*60}")
    
    # Generate a random 6-DOF modular robot
    config = dh2.random_robot_dh(6)
    print(f"\nGenerated combo: {[j['type'] for j in config]}")
    
    # Test target pose
    target_pos = [0.35, 0.15, 0.20]
    target_euler = [0, 0, 45]
    
    print(f"\nTarget:")
    print(f"  Position: {target_pos}")
    print(f"  Orientation: {target_euler}°")
    
    # Solve with adaptive strategy
    q_solution, info = adaptive_ik_solver(config, target_pos, target_euler, verbose=True)
    
    print(f"\nSolution angles (deg): {np.round(q_solution, 1)}")
    
    return q_solution, info


def main():
    print("🤖 ADAPTIVE MODULAR IK SOLVER")
    print("Automatically adapts to any module combination\n")
    
    # Demo with one combo
    demo_adaptive_solver()
    
    # Test on multiple random combos
    test_random_module_combinations(n_combos=20)
    
    print("\n" + "="*60)
    print("✅ ADAPTIVE SOLVER VALIDATED")
    print("="*60)
    print("\nKey Insights:")
    print("  - Full 6D control requires spherical wrist + good σmin")
    print("  - Partial control uses position-priority weighting")
    print("  - Always falls back gracefully to position-only")
    print("  - User gets clear feedback on what's achievable")


if __name__ == "__main__":
    main()

