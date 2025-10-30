import numpy as np
import os
try:
    from ament_index_python.packages import get_package_share_directory
    ROS_AVAILABLE = True
except Exception:
    ROS_AVAILABLE = False

def random_robot_dh(n_axes):
    config = []
    first_type = np.random.choice(['coude', 'poignet'])
    prev_type = first_type

    for i in range(n_axes):
        if i == 0:
            joint_type = 'rot360'
            d = 0.133
            a = 0.0
            alpha = 0.0
        else:
            if prev_type == 'coude':
                if np.random.rand() < 0.5:
                    joint_type = 'rot180'
                    d = 0.0
                    a = 0.122
                    alpha = 0.0
                else:
                    joint_type = 'rot360'
                    d = 0.0
                    a = 0.1925
                    alpha = np.pi / 2
            else:
                joint_type = 'rot180'
                d = 0.0625
                a = 0.0
                alpha = -np.pi / 2

        config.append({'type': joint_type, 'd': d, 'a': a, 'alpha': alpha})
        prev_type = 'coude' if joint_type == 'rot180' else 'poignet'

    return config


def dh_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,       sa,     ca,    d],
        [0,        0,      0,    1]
    ], dtype=float)

def dh_to_urdf(config, name="modular_robot"):
    """
    Génère un URDF complet à partir d'une configuration DH.
    Structure compatible avec test.urdf :
      - rot180 (coude) = ElbowBottom + ElbowTop
      - rot360 (poignet) = WristBottom + WristTop
      - Les liens sont correctement chaînés (top -> next bottom)
    """
    urdf = ['<?xml version="1.0" ?>', f'<robot name="{name}">']

    # === BASE FIXE ===
    urdf.append('  <link name="base_link">')
    urdf.append('    <visual><geometry><mesh filename="package://IKsolverNode/urdf/meshes/Base.stl" scale="0.001 0.001 0.001"/></geometry></visual>')
    urdf.append('    <collision><geometry><mesh filename="package://IKsolverNode/urdf/meshes/Base.stl" scale="0.001 0.001 0.001"/></geometry></collision>')
    urdf.append('  </link>')

    # === Connexion base → premier module ===
    first_type = config[0]["type"]
    first_child = "link_0_bottom" if first_type in ["rot180", "rot360"] else "link_0"
    urdf.append('  <joint name="base_to_first" type="fixed">')
    urdf.append('    <parent link="base_link"/>')
    urdf.append(f'    <child link="{first_child}"/>')
    urdf.append('    <origin xyz="-0.005 -0.02 0.136" rpy="0 0 0"/>')
    urdf.append('  </joint>')

    # === GÉNÉRATION DES MODULES ===
    for i, joint in enumerate(config):
        joint_type = joint["type"]
        a, d, alpha = joint["a"], joint["d"], joint["alpha"]

        # Définir les noms
        bottom_link = f"link_{i}_bottom"
        top_link = f"link_{i}_top"
        joint_name = f"joint_{i}"

        # === COUDE (rot180) ===
        if joint_type == "rot180":
            bottom_mesh = "ElbowBottom.stl"
            top_mesh = "ElbowTop.stl"
            axis = "1 0 0"
            lower, upper = "-1.5708", "1.5708"
            origin = ""

        # === POIGNET (rot360) ===
        elif joint_type == "rot360":
            bottom_mesh = "WristBottom.stl"
            top_mesh = "WristTop.stl"
            axis = "0 0 1"
            lower, upper = "-3.1416", "3.1416"

        else:
            bottom_mesh, top_mesh = None, None
            axis, lower, upper = "0 0 1", "-1.5708", "1.5708"

        # === Liens ===
        for link_name, mesh in [(bottom_link, bottom_mesh), (top_link, top_mesh)]:
            urdf.append(f'  <link name="{link_name}">')
            if mesh:
                urdf.append(f'    <visual><geometry><mesh filename="package://IKsolverNode/urdf/meshes/{mesh}" scale="0.001 0.001 0.001"/></geometry></visual>')
                urdf.append(f'    <collision><geometry><mesh filename="package://IKsolverNode/urdf/meshes/{mesh}" scale="0.001 0.001 0.001"/></geometry></collision>')
            urdf.append('  </link>')

        if joint_type == "rot180":
            urdf.append(f'  <joint name="{joint_name}" type="revolute">')
            urdf.append(f'    <parent link="{bottom_link}"/>')
            urdf.append(f'    <child link="{top_link}"/>')
            urdf.append(f'    <origin xyz="0 0 0" rpy="0 0 0"/>')
            urdf.append(f'    <axis xyz="{axis}"/>')
            urdf.append(f'    <limit lower="{lower}" upper="{upper}" effort="5.0" velocity="1.0"/>')
            urdf.append('  </joint>')
        if joint_type == "rot360":
            urdf.append(f'  <joint name="{joint_name}" type="revolute">')
            urdf.append(f'    <parent link="{bottom_link}"/>')
            urdf.append(f'    <child link="{top_link}"/>')
            urdf.append(f'    <origin xyz="0 0 0" rpy="0 0 {alpha:.4f}"/>')
            urdf.append(f'    <axis xyz="{axis}"/>')
            urdf.append(f'    <limit lower="{lower}" upper="{upper}" effort="5.0" velocity="1.0"/>')
            urdf.append('  </joint>')

        if i < len(config) - 1:
            if config[i + 1]["type"] == "rot180" and config[i]["type"] == "rot360":
                next_bottom = f"link_{i+1}_bottom"
                urdf.append(f'  <joint name="link_{i}_to_{i+1}" type="fixed">')
                urdf.append(f'    <parent link="{top_link}"/>')
                urdf.append(f'    <child link="{next_bottom}"/>')
                urdf.append(f'    <origin xyz="-0.027 0 0.1469" rpy="0 0 0"/>')
                urdf.append('  </joint>')
            if config[i + 1]["type"] == "rot180" and config[i]["type"] == "rot180":
                next_bottom = f"link_{i+1}_bottom"
                urdf.append(f'  <joint name="link_{i}_to_{i+1}" type="fixed">')
                urdf.append(f'    <parent link="{top_link}"/>')
                urdf.append(f'    <child link="{next_bottom}"/>')
                urdf.append(f'    <origin xyz="-0.0005 0 0.1125" rpy="0 0 0"/>')
                urdf.append('  </joint>')
            elif config[i + 1]["type"] == "rot360" and config[i]["type"] == "rot180":
                next_bottom = f"link_{i+1}_bottom"
                urdf.append(f'  <joint name="link_{i}_to_{i+1}" type="fixed">')
                urdf.append(f'    <parent link="{top_link}"/>')
                urdf.append(f'    <child link="{next_bottom}"/>')
                urdf.append(f'    <origin xyz="0.0258 0 0.0895" rpy="0 0 0"/>')
                urdf.append('  </joint>')
            

    # === Dernier link (outil) ===
    last_top = f"link_{len(config)-1}_top"
    urdf.append(f'  <link name="tool_link">')
    urdf.append(f'    <visual><geometry><sphere radius="0.01"/></geometry></visual>')
    urdf.append(f'  </link>')
    urdf.append(f'  <joint name="end_effector" type="fixed">')
    urdf.append(f'    <parent link="{last_top}"/>')
    urdf.append(f'    <child link="tool_link"/>')
    urdf.append(f'    <origin xyz="0 0 0" rpy="0 0 0"/>')
    urdf.append(f'  </joint>')

    urdf.append('</robot>')
    return "\n".join(urdf)









