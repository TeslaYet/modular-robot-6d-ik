import numpy as np
import matplotlib.pyplot as plt
try:
    from IKsolverNode.dh_utils import dh_matrix
except Exception:
    from dh_utils import dh_matrix

def plot_robot_3d(config, q, pos_target=None, orient_target=None, show_all_frames=True, title_suffix=""):
    T = np.eye(4)
    origins = [T[:3, 3]]
    frames = [T[:3, :3].copy()]

    for i, joint in enumerate(config):
        jt = joint['type']
        theta = np.deg2rad(q[i]) if 'rot' in jt else 0.0
        d = joint['d'] if 'rot' in jt else joint['d'] + q[i] / 1000.0
        A = dh_matrix(theta, d, joint['a'], joint['alpha'])
        T = T @ A
        origins.append(T[:3, 3])
        frames.append(T[:3, :3].copy())

    origins = np.array(origins)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot robot links
    for i, joint in enumerate(config):
        p1, p2 = origins[i], origins[i + 1]
        color = 'orange' if joint['type'] == 'rot180' else 'blue'
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=3)

    # Plot joints
    ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], c='k', s=25, label="Articulations")
    ax.scatter(0, 0, 0, c='blue', s=50, label="Origine")

    # Plot end effector
    pos_final = origins[-1]
    ax.scatter(*pos_final, c='green', s=60, label="Position finale")

    # Plot target position
    if pos_target is not None:
        ax.scatter(*pos_target, c='red', s=60, label="Position cible")
        
        # Calculate position error
        pos_error = np.linalg.norm(pos_target - pos_final)
        error_text = f"Erreur position: {pos_error*1000:.1f} mm"

    # Plot target orientation if provided
    if orient_target is not None:
        # Convert target orientation to rotation matrix
        target_rotation = euler_angles_to_rotation_matrix(orient_target)
        axis_len = 0.05
        
        # Plot target orientation axes
        for i, (dir_vec, color, label) in enumerate(zip(target_rotation.T, ['red', 'green', 'blue'], ['X_t', 'Y_t', 'Z_t'])):
            ax.quiver(*pos_target, *dir_vec, length=axis_len, color=color, 
                     arrow_length_ratio=0.3, linewidth=2, alpha=0.7)
            ax.text(*(pos_target + dir_vec * axis_len * 1.2), label, color=color, 
                   fontsize=8, fontweight='bold')
        
        # Calculate geodesic orientation error
        from kinematics import euler_deg_to_R, rotation_error_rvec, get_end_effector_pose
        current_pose = get_end_effector_pose(config, q)
        current_orient = current_pose['orientation']
        pose_R = euler_deg_to_R(current_orient)
        target_R = euler_deg_to_R(orient_target)
        orient_error_norm = np.rad2deg(np.linalg.norm(rotation_error_rvec(pose_R, target_R)))
        
        if 'error_text' in locals():
            error_text += f"\nErreur orientation: {orient_error_norm:.1f}°"
        else:
            error_text = f"Erreur orientation: {orient_error_norm:.1f}°"

    # Plot current end effector orientation
    axis_len = 0.025
    for i, R in enumerate(frames):
        if not show_all_frames and i != len(frames) - 1:
            continue
        origin = origins[i]
        for dir_vec, color, label in zip(R.T, ['r', 'g', 'b'], ['X', 'Y', 'Z']):
            ax.quiver(*origin, *dir_vec, length=axis_len, color=color, 
                     arrow_length_ratio=0.4, linewidth=1, alpha=0.6)
            if i == len(frames) - 1:  # Only label end effector
                ax.text(*(origin + dir_vec * axis_len * 1.1), label, color=color, 
                       fontsize=8, fontweight='bold')

    # Labels and title
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    title = f'Robot DH - Configuration 6D{title_suffix}'
    ax.set_title(title)
    
    # Configuration info
    config_str = ", ".join([joint['type'] for joint in config])
    ax.text2D(0.05, 0.95, f"Configuration : ({config_str})", 
              transform=ax.transAxes, fontsize=10, color='black')
    
    # Error info
    if 'error_text' in locals():
        ax.text2D(0.05, 0.05, error_text, transform=ax.transAxes, 
                  fontsize=10, color='red', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.legend(loc='upper right')
    ax.view_init(elev=30, azim=45)
    ax.set_box_aspect([1, 1, 1])
    plt.show()


def euler_angles_to_rotation_matrix(euler_angles):
    """Convert Euler angles (roll, pitch, yaw) in degrees to rotation matrix"""
    roll, pitch, yaw = np.deg2rad(euler_angles)
    
    # ZYX rotation order
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    return Rz @ Ry @ Rx


def angle_error(current, target):
    """Calculate angle error with proper wrapping"""
    error = current - target
    # Wrap to [-180, 180]
    error = ((error + 180) % 360) - 180
    return error


def get_end_effector_pose(config, q):
    """Get the full 6D pose (position + orientation) of the end effector"""
    from kinematics import get_end_effector_pose
    return get_end_effector_pose(config, q)
