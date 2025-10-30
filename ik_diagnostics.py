#!/usr/bin/env python3
import os
import numpy as np
import importlib.util
import sys

# Load dh_utils (2).py robustly
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DH_UTILS2_PATH = os.path.join(THIS_DIR, "dh_utils (2).py")
spec = importlib.util.spec_from_file_location("dh_utils2", DH_UTILS2_PATH)
dh2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dh2)  # type: ignore

# Alias as 'dh_utils' so kinematics.py can import it
sys.modules['dh_utils'] = dh2

from kinematics import (
    inverse_kinematics_dh,
    get_end_effector_pose,
    euler_deg_to_R,
    rotation_error_rvec,
    rotation_matrix_to_euler_angles,
)


def forward_T(config, q_deg):
    T = np.eye(4)
    for i, joint in enumerate(config):
        jt = joint["type"]
        theta = np.deg2rad(q_deg[i]) if "rot" in jt else 0.0
        d = joint["d"] if "rot" in jt else joint["d"] + q_deg[i] / 1000.0
        A = dh2.dh_matrix(theta, d, joint["a"], joint["alpha"])
        T = T @ A
    return T


def rotational_jacobian(config, q_deg, eps_deg=1.0):
    T0 = forward_T(config, q_deg)
    R0 = T0[:3, :3]
    n = len(q_deg)
    Jori = np.zeros((3, n))
    for k in range(n):
        qd = np.array(q_deg, float)
        qd[k] += eps_deg
        Rd = forward_T(config, qd)[:3, :3]
        rvec = rotation_error_rvec(Rd, R0)  # radians
        Jori[:, k] = rvec / np.deg2rad(eps_deg)
    return Jori


def rotational_condition(config, q_deg):
    Jori = rotational_jacobian(config, q_deg)
    s = np.linalg.svd(Jori, compute_uv=False)
    return (float(s.min()), s)


def best_approach_position(config, target_pos, q_pos_solver, radius=0.05, K=12, early_sigma=1.0):
    cand = []
    target_pos = np.asarray(target_pos, float)
    for i in range(K):
        phi = 2 * np.pi * i / K
        for dz in (0.0, radius * 0.5, -radius * 0.5):
            offset = np.array([radius * np.cos(phi), radius * np.sin(phi), dz])
            p = target_pos + offset
            try:
                q = q_pos_solver(config, p)
                sigma_min, _ = rotational_condition(config, q)
                cand.append((sigma_min, p, q))
                if sigma_min >= early_sigma:
                    return max(cand, key=lambda x: x[0])
            except Exception:
                pass
    if not cand:
        return None
    return max(cand, key=lambda x: x[0])


def make_spherical_wrist(config):
    """Force a spherical-wrist layout on the last 3 joints (4–6)."""
    if len(config) < 6:
        return config
    wrist = [3, 4, 5]
    for i in wrist:
        config[i]['type'] = 'rot360'
        config[i]['a'] = 0.0
    config[3]['alpha'] =  np.pi / 2
    config[4]['alpha'] = -np.pi / 2
    config[5]['alpha'] =  0.0
    return config


def known_spherical_6r():
    """Return a known, well-conditioned 6R with a spherical wrist (all revolute).
    Units: meters and radians (alpha in radians).
    """
    return [
        {"type": "rot360", "d": 0.15, "a": 0.0,  "alpha":  np.pi/2},  # base -> shoulder yaw
        {"type": "rot360", "d": 0.00, "a": 0.35, "alpha":  0.0     },  # shoulder pitch
        {"type": "rot360", "d": 0.00, "a": 0.25, "alpha":  0.0     },  # elbow
        {"type": "rot360", "d": 0.10, "a": 0.0,  "alpha":  np.pi/2},  # wrist roll
        {"type": "rot360", "d": 0.10, "a": 0.0,  "alpha": -np.pi/2},  # wrist pitch
        {"type": "rot360", "d": 0.10, "a": 0.0,  "alpha":  0.0     },  # wrist yaw
    ]


def solve_ik_6d_dls(
    config,
    target_pos,
    target_euler_deg,
    q_init,
    max_iter=1500,
    lam=1e-2,
    step_alpha=1.0,
    max_step_deg=2.0,
    orient_scale=1.0,
    eps_deg=0.5,
    pos_tol=1e-3,
    ori_tol_deg=5.0,
):
    target_pos = np.asarray(target_pos, float)
    q = np.array(q_init, float)
    R_tgt = euler_deg_to_R(target_euler_deg)
    n = len(q)

    def clamp_local(qd):
        qx = qd.copy()
        for i, j in enumerate(config):
            if j['type'] == 'rot180':
                qx[i] = np.clip(qx[i], -90.0, 90.0)
        return qx

    theta_step = np.deg2rad(15.0)
    for _ in range(max_iter):
        T0 = forward_T(config, q)
        p0 = T0[:3, 3]
        R0 = T0[:3, :3]
        e_pos = target_pos - p0
        rvec_full = rotation_error_rvec(R0, R_tgt)  # rad
        # Use incremental orientation target to avoid 180° branch issues
        th = np.linalg.norm(rvec_full)
        if th > theta_step:
            rvec_eff = rvec_full * (theta_step / th)
            R_local = exp_so3(rvec_eff) @ R0
        else:
            rvec_eff = rvec_full
            R_local = R_tgt
        if np.linalg.norm(e_pos) < pos_tol and np.linalg.norm(np.rad2deg(rvec_full)) < ori_tol_deg:
            return clamp_local(q)

        # Build 6xN Jacobian (meters/deg for position; (meters-equivalent)/deg for orientation)
        Jpos = np.zeros((3, n))
        Jori = np.zeros((3, n))
        for k in range(n):
            qd = q.copy(); qd[k] += eps_deg
            Td = forward_T(config, qd)
            pd = Td[:3, 3]
            Rd = Td[:3, :3]
            Jpos[:, k] = (pd - p0) / eps_deg
            rvec_d = rotation_error_rvec(Rd, R_local)
            # descent direction w.r.t. local target
            Jori[:, k] = orient_scale * (rvec_d - rvec_eff) / eps_deg

        J6 = np.vstack([Jpos, Jori])
        # Flip sign so positive step reduces orientation error
        e6 = np.hstack([e_pos, orient_scale * rvec_eff])

        # Damped least squares step
        JJt = J6 @ J6.T + lam * np.eye(6)
        dq = step_alpha * (J6.T @ np.linalg.inv(JJt) @ e6)

        # Cap step per-joint
        dq = np.clip(dq, -max_step_deg, max_step_deg)
        q = clamp_local(q + dq)

    return clamp_local(q)


def hat(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def exp_so3(rvec):
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = hat(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def damped_pinv(J, lam):
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    Sd = S / (S * S + lam * lam)
    return (Vt.T * Sd) @ U.T


def task_priority_ik(
    config,
    target_pos,
    target_euler_deg,
    q_init,
    max_iter=3000,
    pos_tol=5e-4,
    ori_tol_deg=5.0,
    lam_pos=1e-3,
    lam_ori=1e-4,
    lr_pos=0.5,
    lr_ori=0.8,
    eps_deg=1.0,
    max_step_deg=2.0,
    orient_scale=0.1
):
    target_pos = np.asarray(target_pos, float)
    q = np.array(q_init, float)
    R_tgt = euler_deg_to_R(target_euler_deg)
    n = len(q)
    I = np.eye(n)

    def clamp_local(qd):
        qx = qd.copy()
        for i, j in enumerate(config):
            if j['type'] == 'rot180':
                qx[i] = np.clip(qx[i], -90.0, 90.0)
        return qx

    for it in range(max_iter):
        T0 = forward_T(config, q)
        p0 = T0[:3, 3]
        R0 = T0[:3, :3]

        e_pos = target_pos - p0
        # Orientation error as log(R_tgt * R0^T)
        rvec = rotation_error_rvec(R0, R_tgt)
        e_ori_scaled = orient_scale * rvec  # meters-equivalent

        if np.linalg.norm(e_pos) < pos_tol and np.linalg.norm(np.rad2deg(rvec)) < ori_tol_deg:
            return clamp_local(q)

        # Jacobians via FD
        Jpos = np.zeros((3, n))
        Jori = np.zeros((3, n))
        for k in range(n):
            qd = q.copy(); qd[k] += eps_deg
            Td = forward_T(config, qd)
            pd = Td[:3, 3]
            Rd = Td[:3, :3]
            # Derivatives w.r.t. degrees
            Jpos[:, k] = (pd - p0) / eps_deg  # m/deg
            # Consistent SO(3) error: log(R_tgt * R^T)
            rvec_d = rotation_error_rvec(Rd, R_tgt)  # rad
            Jori[:, k] = orient_scale * (rvec_d - rvec) / eps_deg  # meters-equivalent/deg

        # Task-priority update
        Jp_pinv = damped_pinv(Jpos, lam_pos)
        dq_pos = Jp_pinv @ e_pos
        N = I - Jp_pinv @ Jpos
        Jori_tilde = Jori @ N
        Jo_pinv = damped_pinv(Jori_tilde, lam_ori)
        dq_ori = Jo_pinv @ e_ori_scaled
        # Gradual orientation ramp-up to protect position early
        w = min(1.0, (it + 1) / (0.3 * max_iter))
        dq_mix = lr_pos * dq_pos + (w * lr_ori) * dq_ori

        # Backtracking line search on combined cost
        def cost(vec_pos, vec_rvec):
            return (np.linalg.norm(vec_pos) / max(pos_tol, 1e-9)
                    + np.linalg.norm(np.rad2deg(vec_rvec)) / max(ori_tol_deg, 1e-6))

        cost0 = cost(e_pos, rvec)
        step = 1.0
        accepted = False
        for _ls in range(12):
            dq_step = dq_mix * step
            # Step cap
            norm_dq = np.linalg.norm(dq_step)
            if norm_dq > max_step_deg:
                dq_step *= max_step_deg / (norm_dq + 1e-9)
            q_try = clamp_local(q + dq_step)

            Tt = forward_T(config, q_try)
            pt = Tt[:3, 3]
            Rt = Tt[:3, :3]
            epos_t = target_pos - pt
            rvec_t = rotation_error_rvec(Rt, R_tgt)
            c1 = cost(epos_t, rvec_t)

            if c1 < cost0:
                q = q_try
                accepted = True
                break
            step *= 0.5

        if not accepted:
            # shrink gains and retry next iteration, do not update q this iter
            lr_pos *= 0.7
            lr_ori *= 0.7
            continue

    return clamp_local(q)


def main():
    # 1) Use a known spherical-wrist 6R (deterministic, robust for 6D IK)
    config = known_spherical_6r()

    # 2) Define target pose (meters + degrees)
    target_pos = np.array([0.40, 0.10, 0.20], float)
    target_euler_deg = np.array([0, 0, 45], float)

    # 3) Position-only IK seed
    q_pos = inverse_kinematics_dh(config, target_pos)
    print("Pos-only IK q (deg):", np.round(q_pos, 2))
    # 4) Direct 6D DLS solve (tighter damping and step)
    q_refined = solve_ik_6d_dls(
        config,
        target_pos,
        target_euler_deg,
        q_init=q_pos,
        max_iter=3000,
        lam=1e-2,
        step_alpha=0.7,
        max_step_deg=1.5,
        orient_scale=1.0,
        eps_deg=0.5,
        pos_tol=5e-4,
        ori_tol_deg=5.0,
    )
    pose = get_end_effector_pose(config, q_refined)

    pos_err = np.linalg.norm(target_pos - pose["position"]) * 1000.0  # mm
    R_cur = euler_deg_to_R(pose["orientation"])  # degrees in, convert to R
    R_tgt = euler_deg_to_R(target_euler_deg)
    ang_err_deg = np.rad2deg(np.linalg.norm(rotation_error_rvec(R_cur, R_tgt)))

    print("\n== 6D IK (DLS) ==")
    print("Target position:", np.round(target_pos, 4))
    print("q_6d (deg):", np.round(q_refined, 2))
    print(f"Position error: {pos_err:.2f} mm")
    print(f"Orientation error (geodesic): {ang_err_deg:.2f} deg")


if __name__ == "__main__":
    main()


