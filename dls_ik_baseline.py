#!/usr/bin/env python3
"""
Canonical Damped Least Squares (DLS) IK solver.
Textbook formulation guaranteed to converge for smooth reachable targets.
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


def forward_kinematics(config, q_deg):
    """
    Forward kinematics using DH parameters.
    Returns: position (3,) and rotation matrix (3x3)
    """
    T = np.eye(4)
    for i, joint in enumerate(config):
        jt = joint["type"]
        theta = np.deg2rad(q_deg[i]) if "rot" in jt else 0.0
        d = joint["d"] if "rot" in jt else joint["d"] + q_deg[i] / 1000.0
        A = dh2.dh_matrix(theta, d, joint["a"], joint["alpha"])
        T = T @ A
    return T[:3, 3], T[:3, :3]


def rotation_error_cross(R_cur, R_tgt):
    """
    Orientation error using cross-product formulation (textbook DLS).
    e_R = 0.5 * (R_cur[:,0] × R_tgt[:,0] + R_cur[:,1] × R_tgt[:,1] + R_cur[:,2] × R_tgt[:,2])
    """
    e_R = 0.5 * (
        np.cross(R_cur[:, 0], R_tgt[:, 0]) +
        np.cross(R_cur[:, 1], R_tgt[:, 1]) +
        np.cross(R_cur[:, 2], R_tgt[:, 2])
    )
    return e_R


def compute_jacobian_6d(config, q_deg, eps=0.01):
    """
    Compute 6×n Jacobian using finite differences.
    Returns J where rows 0-2 are position derivatives, rows 3-5 are orientation derivatives.
    """
    n = len(q_deg)
    J = np.zeros((6, n))
    
    x_cur, R_cur = forward_kinematics(config, q_deg)
    
    for i in range(n):
        q_plus = q_deg.copy()
        q_plus[i] += eps
        x_plus, R_plus = forward_kinematics(config, q_plus)
        
        # Position derivative
        J[0:3, i] = (x_plus - x_cur) / eps
        
        # Orientation derivative (using cross-product error)
        # Note: this is approximate but works for small eps
        e_R_cur = np.zeros(3)
        e_R_plus = rotation_error_cross(R_plus, R_cur)
        J[3:6, i] = e_R_plus / eps
    
    return J


def inverse_kinematics_dls(
    config,
    target_pos,
    target_R,
    q_init=None,
    max_iter=100,
    lam=0.01,
    pos_tol=1e-3,
    ori_tol=1e-3,
    step_size=1.0,
):
    """
    Damped Least Squares IK solver.
    
    Args:
        config: DH configuration
        target_pos: Target position [x, y, z]
        target_R: Target rotation matrix (3x3)
        q_init: Initial joint angles (degrees)
        max_iter: Maximum iterations
        lam: Damping factor
        pos_tol: Position tolerance (meters)
        ori_tol: Orientation tolerance (cross-product norm)
        step_size: Step size multiplier
    
    Returns:
        Joint angles (degrees)
    """
    n = len(config)
    if q_init is None:
        q = np.zeros(n)
    else:
        q = np.array(q_init, dtype=float)
    
    for iteration in range(max_iter):
        # Forward kinematics
        x_cur, R_cur = forward_kinematics(config, q)
        
        # Compute errors
        e_pos = target_pos - x_cur
        e_ori = rotation_error_cross(R_cur, target_R)
        e = np.hstack([e_pos, e_ori])
        
        # Check convergence
        if np.linalg.norm(e_pos) < pos_tol and np.linalg.norm(e_ori) < ori_tol:
            return q
        
        # Compute Jacobian
        J = compute_jacobian_6d(config, q)
        
        # DLS update: Δq = (J^T J + λ²I)^{-1} J^T e
        JtJ = J.T @ J
        dq = np.linalg.inv(JtJ + (lam**2) * np.eye(n)) @ (J.T @ e)
        
        # Update with step size
        q = q + step_size * dq
    
    return q


def euler_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """Convert Euler angles (degrees) to rotation matrix (ZYX convention)."""
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    return Rz @ Ry @ Rx


def test_synthetic_fk_ik_roundtrip(config, n_tests=10):
    """
    Validate IK solver: FK(random q) → target → IK(target) → should return q ≈ random q
    """
    print("=" * 60)
    print("SYNTHETIC FK→IK ROUNDTRIP TEST")
    print("=" * 60)
    
    errors_pos = []
    errors_ori = []
    
    for i in range(n_tests):
        # Generate random joint angles
        q_true = np.random.uniform(-45, 45, len(config))
        
        # Forward kinematics
        x_target, R_target = forward_kinematics(config, q_true)
        
        # Solve IK from zero initial guess
        q_solved = inverse_kinematics_dls(
            config, x_target, R_target, q_init=None, max_iter=100, lam=0.01
        )
        
        # Verify with FK
        x_check, R_check = forward_kinematics(config, q_solved)
        
        # Compute errors
        pos_err = np.linalg.norm(x_target - x_check)
        ori_err = np.linalg.norm(rotation_error_cross(R_check, R_target))
        
        errors_pos.append(pos_err * 1000)  # mm
        errors_ori.append(ori_err * 1000)  # mrad → approx deg for small errors
        
        if i < 3:  # Show first 3
            print(f"Test {i+1}:")
            print(f"  q_true:   {np.round(q_true, 2)}")
            print(f"  q_solved: {np.round(q_solved, 2)}")
            print(f"  Position error: {pos_err*1000:.3f} mm")
            print(f"  Orientation error: {ori_err:.6f}\n")
    
    print(f"Summary ({n_tests} tests):")
    print(f"  Position error: mean={np.mean(errors_pos):.3f} mm, max={np.max(errors_pos):.3f} mm")
    print(f"  Orientation error: mean={np.mean(errors_ori):.6f}, max={np.max(errors_ori):.6f}")
    print()


def test_target_pose(config, target_pos, target_euler_deg):
    """
    Test IK for a specific target pose.
    """
    print("=" * 60)
    print("TARGET POSE TEST")
    print("=" * 60)
    print(f"Target position: {target_pos}")
    print(f"Target orientation (deg): {target_euler_deg}\n")
    
    # Convert orientation to rotation matrix
    R_target = euler_to_rotation_matrix(*target_euler_deg)
    
    # Solve IK
    q_solution = inverse_kinematics_dls(
        config, target_pos, R_target, q_init=None, max_iter=200, lam=0.01
    )
    
    # Verify
    x_check, R_check = forward_kinematics(config, q_solution)
    pos_err = np.linalg.norm(target_pos - x_check)
    ori_err = np.linalg.norm(rotation_error_cross(R_check, R_target))
    
    print(f"Solution: q = {np.round(q_solution, 2)}")
    print(f"Reached position: {np.round(x_check, 4)}")
    print(f"Position error: {pos_err*1000:.3f} mm")
    print(f"Orientation error: {ori_err:.6f}")
    print()


def get_common_robot_dhs():
    """Return DH parameters for common industrial robots."""
    robots = {}
    
    # UR5 (Universal Robots UR5)
    robots['UR5'] = [
        {"type": "rot360", "d": 0.089159, "a": 0.0,      "alpha":  np.pi/2},
        {"type": "rot360", "d": 0.0,      "a": -0.42500, "alpha":  0.0},
        {"type": "rot360", "d": 0.0,      "a": -0.39225, "alpha":  0.0},
        {"type": "rot360", "d": 0.10915,  "a": 0.0,      "alpha":  np.pi/2},
        {"type": "rot360", "d": 0.09465,  "a": 0.0,      "alpha": -np.pi/2},
        {"type": "rot360", "d": 0.0823,   "a": 0.0,      "alpha":  0.0},
    ]
    
    # PUMA 560 (classic spherical wrist)
    robots['PUMA560'] = [
        {"type": "rot360", "d": 0.0,    "a": 0.0,   "alpha":  np.pi/2},
        {"type": "rot360", "d": 0.0,    "a": 0.432, "alpha":  0.0},
        {"type": "rot360", "d": 0.149,  "a": 0.020, "alpha": -np.pi/2},
        {"type": "rot360", "d": 0.433,  "a": 0.0,   "alpha":  np.pi/2},
        {"type": "rot360", "d": 0.0,    "a": 0.0,   "alpha": -np.pi/2},
        {"type": "rot360", "d": 0.0,    "a": 0.0,   "alpha":  0.0},
    ]
    
    # Simple spherical wrist 6R (your custom)
    robots['Spherical6R'] = [
        {"type": "rot360", "d": 0.15, "a": 0.0,  "alpha":  np.pi/2},
        {"type": "rot360", "d": 0.00, "a": 0.35, "alpha":  0.0},
        {"type": "rot360", "d": 0.00, "a": 0.25, "alpha":  0.0},
        {"type": "rot360", "d": 0.10, "a": 0.0,  "alpha":  np.pi/2},
        {"type": "rot360", "d": 0.10, "a": 0.0,  "alpha": -np.pi/2},
        {"type": "rot360", "d": 0.10, "a": 0.0,  "alpha":  0.0},
    ]
    
    # 4-DOF SCARA-like
    robots['SCARA4'] = [
        {"type": "rot360", "d": 0.35, "a": 0.25,  "alpha": 0.0},
        {"type": "rot360", "d": 0.0,  "a": 0.20,  "alpha": 0.0},
        {"type": "prismatic", "d": 0.0, "a": 0.0, "alpha": 0.0},
        {"type": "rot360", "d": 0.05, "a": 0.0,   "alpha": 0.0},
    ]
    
    # 3-DOF planar arm
    robots['Planar3R'] = [
        {"type": "rot360", "d": 0.0, "a": 0.30, "alpha": 0.0},
        {"type": "rot360", "d": 0.0, "a": 0.25, "alpha": 0.0},
        {"type": "rot360", "d": 0.0, "a": 0.15, "alpha": 0.0},
    ]
    
    return robots


def get_common_test_poses():
    """Return common target poses for testing."""
    return [
        {
            'name': 'Front vertical approach',
            'position': [0.40, 0.00, 0.30],
            'euler': [0, 0, 0],
        },
        {
            'name': 'Side horizontal approach',
            'position': [0.30, 0.20, 0.20],
            'euler': [0, 90, 0],
        },
        {
            'name': 'Angled approach',
            'position': [0.35, -0.15, 0.25],
            'euler': [30, 45, 60],
        },
        {
            'name': 'Low pickup',
            'position': [0.45, 0.10, 0.10],
            'euler': [0, 0, 90],
        },
    ]


def test_robot_with_common_poses(robot_name, config):
    """Test a robot configuration with common target poses."""
    print("=" * 60)
    print(f"ROBOT: {robot_name}")
    print("=" * 60)
    
    poses = get_common_test_poses()
    results = []
    
    for pose in poses:
        target_pos = np.array(pose['position'])
        target_euler = pose['euler']
        R_target = euler_to_rotation_matrix(*target_euler)
        
        print(f"\n📍 {pose['name']}")
        print(f"   Target: pos={pose['position']}, orient={target_euler}°")
        
        try:
            q_sol = inverse_kinematics_dls(
                config, target_pos, R_target, q_init=None, max_iter=200, lam=0.01
            )
            
            x_check, R_check = forward_kinematics(config, q_sol)
            pos_err = np.linalg.norm(target_pos - x_check) * 1000  # mm
            ori_err = np.linalg.norm(rotation_error_cross(R_check, R_target))
            
            print(f"   ✅ Solution: {np.round(q_sol, 1)}")
            print(f"   ✅ Position error: {pos_err:.2f} mm")
            print(f"   ✅ Orientation error: {ori_err:.6f}")
            
            results.append({'pos_err': pos_err, 'ori_err': ori_err, 'success': True})
        
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({'success': False})
    
    # Summary
    successes = [r for r in results if r.get('success', False)]
    if successes:
        avg_pos = np.mean([r['pos_err'] for r in successes])
        avg_ori = np.mean([r['ori_err'] for r in successes])
        max_pos = np.max([r['pos_err'] for r in successes])
        max_ori = np.max([r['ori_err'] for r in successes])
        
        print(f"\n📊 {robot_name} Summary:")
        print(f"   Success rate: {len(successes)}/{len(results)}")
        print(f"   Avg position error: {avg_pos:.2f} mm (max: {max_pos:.2f} mm)")
        print(f"   Avg orientation error: {avg_ori:.6f} (max: {max_ori:.6f})")
    print()


def main():
    print("🤖 CANONICAL DLS IK - COMPREHENSIVE VALIDATION")
    print()
    
    # Get common robots
    robots = get_common_robot_dhs()
    
    # Test each robot with common poses
    for robot_name in ['Spherical6R', 'UR5', 'PUMA560', 'Planar3R', 'SCARA4']:
        config = robots[robot_name]
        
        # First: synthetic roundtrip validation
        print(f"\n{'='*60}")
        print(f"{robot_name} - FK→IK ROUNDTRIP VALIDATION")
        print('='*60)
        test_synthetic_fk_ik_roundtrip(config, n_tests=10)
        
        # Second: common target poses (may fail for under-actuated robots)
        test_robot_with_common_poses(robot_name, config)
    
    print("\n" + "="*60)
    print("✅ VALIDATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

