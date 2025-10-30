import numpy as np
try:
    from IKsolverNode.dh_utils import dh_matrix
except Exception:
    from dh_utils import dh_matrix

def forward_kinematics_dh(config, q, return_points=False, only_positive_z=False):
    T = np.eye(4)
    points = [np.zeros(3)]

    for i, joint in enumerate(config):
        jt = joint['type']
        theta = np.deg2rad(q[i]) if 'rot' in jt else 0.0
        d = joint['d'] if 'rot' in jt else joint['d'] + q[i] / 1000.0
        A = dh_matrix(theta, d, joint['a'], joint['alpha'])
        T = T @ A
        points.append(T[:3, 3])
        if np.linalg.norm(T[:3, 3]) > 2.0:
            return np.array([np.nan, np.nan, np.nan])

    points = np.array(points)
    if only_positive_z:
        mask = points[:, 2] > 0
        if not np.any(mask):
            return np.array([np.nan, np.nan, np.nan])
        points = points[mask]
    return points if return_points else points[-1]


def get_end_effector_pose(config, q):
    """
    Get the full 6D pose (position + orientation) of the end effector.
    Returns: {'position': [x, y, z], 'orientation': [roll, pitch, yaw]} in degrees
    """
    T = np.eye(4)
    
    for i, joint in enumerate(config):
        jt = joint['type']
        theta = np.deg2rad(q[i]) if 'rot' in jt else 0.0
        d = joint['d'] if 'rot' in jt else joint['d'] + q[i] / 1000.0
        A = dh_matrix(theta, d, joint['a'], joint['alpha'])
        T = T @ A
    
    # Extract position
    position = T[:3, 3]
    
    # Extract rotation matrix and convert to Euler angles
    R = T[:3, :3]
    roll, pitch, yaw = rotation_matrix_to_euler_angles(R)
    
    return {
        'position': position,
        'orientation': [roll, pitch, yaw]  # in degrees
    }


def rotation_matrix_to_euler_angles(R):
    """
    Convert rotation matrix to Euler angles (ZYX convention).
    Returns roll, pitch, yaw in degrees.
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    return np.rad2deg([roll, pitch, yaw])


def euler_deg_to_R(euler_deg):
    """
    Convert Euler angles in degrees (roll, pitch, yaw, ZYX convention) to rotation matrix.
    """
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
    """
    Compute orientation error as a rotation vector (axis*angle, radians) that
    rotates R_current to R_target. This is the logarithm map on SO(3).
    """
    R_err = R_target @ R_current.T
    tr = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(tr)
    if angle < 1e-8:
        return np.zeros(3)
    denom = 2.0 * np.sin(angle)
    rx = (R_err[2, 1] - R_err[1, 2]) / denom
    ry = (R_err[0, 2] - R_err[2, 0]) / denom
    rz = (R_err[1, 0] - R_err[0, 1]) / denom
    return angle * np.array([rx, ry, rz])


def is_reachable(config, pos):
    reach = sum(abs(j['a']) + abs(j['d']) for j in config)
    return np.linalg.norm(pos) <= reach


def inverse_kinematics_dh(config, target_pos, q_init=None, max_iter=2000, lr=0.5, tol=1e-4, n_restarts=2):
    n = len(config)
    best_q, best_err = None, np.inf

    def clamp_angles(q):
        q_clamped = q.copy()
        for i, joint in enumerate(config):
            if joint['type'] == 'rot180':
                q_clamped[i] = np.clip(q[i], -70, 70)
        return q_clamped

    for _ in range(n_restarts):
        q = np.random.uniform(-45, 45, size=n) if q_init is None else np.array(q_init) + np.random.normal(0, 5, n)
        prev_err = np.inf
        for _ in range(max_iter):
            pos = forward_kinematics_dh(config, q)
            error = target_pos - pos
            err_norm = np.linalg.norm(error)
            if err_norm < tol:
                return clamp_angles(q)

            J = np.zeros((3, n))
            eps = 1e-4
            for i in range(n):
                dq = np.zeros(n); dq[i] = eps
                pos_d = forward_kinematics_dh(config, q + dq)
                J[:, i] = (pos_d - pos) / eps

            lam = 1e-3
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(3))
            dq = lr * (J_pinv @ error)
            q += dq
            if err_norm > prev_err * 1.2: lr *= 0.5
            prev_err = err_norm
        if err_norm < best_err:
            best_q, best_err = q, err_norm

    print(f"⚠️ IK non convergée après {max_iter} itérations (meilleur err: {best_err:.4f})")
    return clamp_angles(best_q if best_q is not None else np.zeros(n))


def inverse_kinematics_dh_6d(config, target_pos, target_orientation, q_init=None, max_iter=5000, lr=0.1, tol=1e-4, n_restarts=1, weights=[1.0, 1.0, 1.0, 1.5, 1.5, 1.5]):
    """
    6D inverse kinematics solver with position and orientation.
    
    Args:
        config: Robot configuration
        target_pos: Target position [x, y, z]
        target_orientation: Target orientation [roll, pitch, yaw] in degrees
        weights: Error weights [pos_x, pos_y, pos_z, roll, pitch, yaw]
    
    Returns:
        Joint angles that achieve the target pose
    """
    n = len(config)
    best_q, best_err = None, np.inf
    
    def clamp_angles(q):
        q_clamped = q.copy()
        for i, joint in enumerate(config):
            if joint['type'] == 'rot180':
                q_clamped[i] = np.clip(q[i], -70, 70)
        return q_clamped
    
    # Normalize inputs
    target_pos = np.asarray(target_pos, dtype=float)
    target_orientation = np.asarray(target_orientation, dtype=float)
    weights = np.asarray(weights, dtype=float)
    target_R = euler_deg_to_R(target_orientation)

    # Orientation scaling to comparable units (meters per radian)
    orient_scale = 0.25
    
    for _ in range(n_restarts):
        if q_init is None:
            # Warm start with 3D IK solution for better convergence
            q = inverse_kinematics_dh(config, target_pos)
        else:
            q = np.array(q_init) + np.random.normal(0, 2, n)
        prev_err = np.inf
        
        for _ in range(max_iter):
            # Forward kinematics to current pose (position + rotation)
            T = np.eye(4)
            for i_joint, joint in enumerate(config):
                jt = joint['type']
                theta = np.deg2rad(q[i_joint]) if 'rot' in jt else 0.0
                d = joint['d'] if 'rot' in jt else joint['d'] + q[i_joint] / 1000.0
                A = dh_matrix(theta, d, joint['a'], joint['alpha'])
                T = T @ A
            current_pos = T[:3, 3]
            current_R = T[:3, :3]

            # 6D error: position (m) and rotation-vector (rad) scaled to meters
            pos_error = target_pos - current_pos
            rvec_error = rotation_error_rvec(current_R, target_R)
            error6 = np.concatenate([pos_error, orient_scale * rvec_error])

            # Weighted error
            W = np.diag(weights)
            Wsqrt = np.sqrt(W)
            err_w = Wsqrt @ error6

            err_norm = np.linalg.norm(err_w)
            if err_norm < tol:
                return clamp_angles(q)
            
            # Compute 6D Jacobian by finite differences on the same error6
            J = np.zeros((6, n))
            eps_deg = 1.0
            for i in range(n):
                dq_vec = np.zeros(n)
                dq_vec[i] = eps_deg
                q_d = q + dq_vec

                Td = np.eye(4)
                for j, joint in enumerate(config):
                    jt = joint['type']
                    theta = np.deg2rad(q_d[j]) if 'rot' in jt else 0.0
                    d = joint['d'] if 'rot' in jt else joint['d'] + q_d[j] / 1000.0
                    A = dh_matrix(theta, d, joint['a'], joint['alpha'])
                    Td = Td @ A
                pos_d = Td[:3, 3]
                R_d = Td[:3, :3]

                pos_err_d = target_pos - pos_d
                rvec_err_d = rotation_error_rvec(R_d, target_R)
                e6_d = np.concatenate([pos_err_d, orient_scale * rvec_err_d])
                J[:, i] = (e6_d - error6) / eps_deg
            
            # Weighted least-squares step (Levenberg-Marquardt)
            J_w = Wsqrt @ J
            lam = 1e-2
            J_pinv = J_w.T @ np.linalg.inv(J_w @ J_w.T + lam * np.eye(6))
            dq = lr * (J_pinv @ (Wsqrt @ error6))
            q += dq
            
            if err_norm > prev_err * 1.2:
                lr *= 0.5
            prev_err = err_norm
        
        if err_norm < best_err:
            best_q, best_err = q, err_norm
    
    print(f"⚠️ 6D IK non convergée après {max_iter} itérations (meilleur err: {best_err:.4f})")
    
    # Fallback to 3D position-only if 6D fails
    print("🔄 Fallback vers IK 3D (position seulement)")
    return inverse_kinematics_dh(config, target_pos, q_init, max_iter, 0.5, tol, 2)
