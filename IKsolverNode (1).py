import os
import numpy as np
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header
    from ament_index_python.packages import get_package_share_directory
    ROS_AVAILABLE = True
except Exception:
    ROS_AVAILABLE = False
    rclpy = None
    Node = object
    JointState = None
    Header = None
try:
    from IKsolverNode.dh_utils import random_robot_dh, dh_to_urdf
    from IKsolverNode.kinematics import forward_kinematics_dh, inverse_kinematics_dh, is_reachable
    from IKsolverNode.plot_robot import plot_robot_3d
except Exception:
    from dh_utils import random_robot_dh, dh_to_urdf
    from kinematics import forward_kinematics_dh, inverse_kinematics_dh, is_reachable
    from plot_robot import plot_robot_3d
import threading


if ROS_AVAILABLE:
    URDF_PATH = os.path.expanduser(os.path.join(get_package_share_directory("IKsolverNode"), "urdf/robot_dh.urdf"))
else:
    URDF_PATH = os.path.join(os.path.dirname(__file__), "urdf", "robot_dh.urdf")

class JointPublisher(Node):
    """Publie les positions articulaires calculées."""
    def __init__(self, joint_names, q):
        super().__init__('joint_publisher')
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.joint_names = joint_names
        self.q = q
        self.get_logger().info("JointPublisher prêt. Publie sur /joint_states.")

    def timer_callback(self):
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = np.deg2rad(self.q).tolist()  # radians pour URDF
        self.publisher_.publish(msg)


def main():
    # === Étape 1 : Génération du robot DH ===
    config = random_robot_dh(6)
    pos_target = [0.3, 0.2, 0.0]

    if not is_reachable(config, pos_target):
        print("❌ Cible hors de portée.")
        return

    q_guess = inverse_kinematics_dh(config, pos_target)
    pos_check = forward_kinematics_dh(config, q_guess)
    
    print("Position cible :", np.round(pos_target, 4))
    print("Position atteinte :", np.round(pos_check, 4))
    print("Erreur (mm):", np.linalg.norm(np.array(pos_target) - pos_check) * 1000)
    print("Angles (°):", np.round(q_guess, 2))

    # === Étape 2 : Génération de l’URDF ===
    os.makedirs(os.path.dirname(URDF_PATH), exist_ok=True)
    urdf_str = dh_to_urdf(config, name="modular_robot")
    with open(URDF_PATH, "w") as f:
        f.write(urdf_str)
    print(f"✅ URDF sauvegardé : {URDF_PATH}")
    threading.Thread(target=plot_robot_3d, args=(config, q_guess, pos_target)).start()

    print("\n🚀 Pour visualiser dans RViz :")
    print("1️⃣  Lance RViz dans un autre terminal :")
    print(f"   rviz2 -d $(ros2 pkg prefix my_robot)/share/my_robot/config/display.rviz")
    print("2️⃣  Puis exécute dans ce terminal le nœud joint_publisher (voir ci-dessous)\n")

    # === Étape 4 : Publication des joints ===
    if ROS_AVAILABLE:
        rclpy.init()
        joint_names = [f"joint_{i+1}" for i in range(len(config))]
        node = JointPublisher(joint_names, q_guess)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
