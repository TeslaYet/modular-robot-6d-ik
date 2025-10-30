#!/usr/bin/env python3
import os
import sys
import numpy as np

# Robustly import the helper with parentheses in file name
import importlib.util

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DH_UTILS_PATH = os.path.join(THIS_DIR, "dh_utils (2).py")

spec = importlib.util.spec_from_file_location("dh_utils_mod", DH_UTILS_PATH)
dh_utils_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dh_utils_mod)  # type: ignore

try:
    from ikpy.chain import Chain
    from ikpy.link import OriginLink
except Exception as e:
    print("ikpy is required. Install with:  pip install ikpy")
    raise


def euler_deg_to_R(euler_deg):
    roll, pitch, yaw = np.deg2rad(euler_deg)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0,            0,           1]])
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0,              1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    return Rz @ Ry @ Rx


def rotation_error_rvec(R_current, R_target):
    R_err = R_target @ R_current.T
    tr = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(tr)
    if angle < 1e-9:
        return np.zeros(3)
    denom = 2.0 * np.sin(angle)
    rx = (R_err[2, 1] - R_err[1, 2]) / denom
    ry = (R_err[0, 2] - R_err[2, 0]) / denom
    rz = (R_err[1, 0] - R_err[0, 1]) / denom
    return angle * np.array([rx, ry, rz])


def save_urdf(config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urdf_str = dh_utils_mod.dh_to_urdf(config, name="modular_robot_ikpy")
    with open(path, "w", encoding="utf-8") as f:
        f.write(urdf_str)


def build_chain(urdf_path):
    # base_link and tool_link are defined in the URDF from dh_utils (1).py
    chain = Chain.from_urdf_file(urdf_path, base_elements=["base_link"])  # tool_link is last
    # Activate only the revolute joints (named 'joint_*'); keep all fixed/base/tip inactive
    mask = [False] * len(chain.links)
    for i, link in enumerate(chain.links):
        name = getattr(link, "name", "") or ""
        if name.startswith("joint_"):
            mask[i] = True
    # Optional: print active links for verification
    try:
        active_names = [link.name for i, link in enumerate(chain.links) if mask[i]]
        print("Active IK links:", active_names)
    except Exception:
        pass
    chain.active_links_mask = mask
    return chain


def inverse_kinematics_ikpy(chain, target_pos, target_euler_deg, initial_q=None):
    R = euler_deg_to_R(target_euler_deg)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(target_pos, dtype=float)

    # Initial guess: position-only IK warm start, then full pose
    if initial_q is None:
        try:
            initial_q = chain.inverse_kinematics(target_position=T[:3, 3], max_iter=1000)
        except Exception:
            initial_q = np.zeros(len(chain.links))

    # Prefer explicit orientation API; fallback to frame/position-only if unavailable
    angles = None
    try:
        angles = chain.inverse_kinematics(
            target_position=T[:3, 3],
            target_orientation=target_euler_deg,  # degrees
            orientation_mode="all",
            initial_position=initial_q,
            max_iter=3000,
        )
    except Exception:
        if hasattr(chain, "inverse_kinematics_frame"):
            angles = chain.inverse_kinematics_frame(T, initial_position=initial_q, max_iter=3000)
        else:
            angles = chain.inverse_kinematics(
                target_position=T[:3, 3],
                initial_position=initial_q,
                max_iter=2000,
            )

    # Compute forward to evaluate errors
    T_fk = chain.forward_kinematics(angles)
    pos_err = np.linalg.norm(T[:3, 3] - T_fk[:3, 3])
    ang_err_deg = np.rad2deg(np.linalg.norm(rotation_error_rvec(T_fk[:3, :3], T[:3, :3])))
    return angles, T_fk, pos_err, ang_err_deg


def main():
    # 1) Build a robot from modules (random 6-DOF here, or replace with your combo)
    config = dh_utils_mod.random_robot_dh(6)

    # 2) Generate URDF from modules and load into IK stack
    urdf_path = os.path.join(THIS_DIR, "urdf", "robot_modular_ikpy.urdf")
    save_urdf(config, urdf_path)
    chain = build_chain(urdf_path)

    # 3) Define a 6D target pose (meters + degrees)
    target_pos = [0.3, 0.2, 0.1]
    target_euler_deg = [0, 0, 45]

    # 4) Solve IK and report errors
    angles, T_fk, pos_err, ang_err_deg = inverse_kinematics_ikpy(
        chain, target_pos, target_euler_deg
    )

    print("== IKPY 6D IK Result ==")
    print("Target position:", np.round(target_pos, 4))
    print("Reached position:", np.round(T_fk[:3, 3], 4))
    print(f"Position error: {pos_err*1000:.2f} mm")
    print("Target orientation (deg):", target_euler_deg)
    print("Reached R (first row):", np.round(T_fk[0, :3], 4))
    print(f"Orientation error (geodesic): {ang_err_deg:.2f} deg")


if __name__ == "__main__":
    main()

