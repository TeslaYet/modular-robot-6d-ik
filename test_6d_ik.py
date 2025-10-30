#!/usr/bin/env python3
"""
Test script to compare 3D vs 6D inverse kinematics results.
Demonstrates the difference between position-only and pose-based IK.
"""

import numpy as np
import matplotlib.pyplot as plt
from dh_utils import random_robot_dh
from kinematics import forward_kinematics_dh, inverse_kinematics_dh, inverse_kinematics_dh_6d, get_end_effector_pose, is_reachable, euler_deg_to_R, rotation_error_rvec
from plot_robot import plot_robot_3d

def test_3d_vs_6d_ik():
    """Compare 3D position-only vs 6D pose-based IK"""
    
    print("🤖 Test de comparaison IK 3D vs 6D")
    print("=" * 50)
    
    # Generate random robot
    config = random_robot_dh(6)
    print(f"Configuration robot: {[joint['type'] for joint in config]}")
    
    # Test cases with different target poses
    test_cases = [
        {
            'name': 'Approche verticale (cup)',
            'position': [0.3, 0.2, 0.1],
            'orientation': [0, 0, 0],  # Straight down
            'description': 'Robot doit s\'approcher verticalement'
        },
        {
            'name': 'Approche horizontale (bottle)',
            'position': [0.25, 0.15, 0.05],
            'orientation': [0, 0, 90],  # Horizontal approach
            'description': 'Robot doit s\'approcher horizontalement'
        },
        {
            'name': 'Approche inclinée',
            'position': [0.2, 0.3, 0.08],
            'orientation': [30, 45, 60],  # Complex orientation
            'description': 'Robot doit atteindre une orientation complexe'
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📋 Test {i+1}: {test_case['name']}")
        print(f"   Description: {test_case['description']}")
        print(f"   Position cible: {test_case['position']}")
        print(f"   Orientation cible: {test_case['orientation']}°")
        
        target_pos = np.array(test_case['position'])
        target_orient = np.array(test_case['orientation'])
        
        # Check reachability
        if not is_reachable(config, target_pos):
            print("   ❌ Position hors de portée")
            continue
        
        print("\n   🔄 Résolution IK 3D (position seulement)...")
        try:
            q_3d = inverse_kinematics_dh(config, target_pos)
            pos_3d = forward_kinematics_dh(config, q_3d)
            pos_error_3d = np.linalg.norm(target_pos - pos_3d)
            print(f"   ✅ IK 3D: Erreur position = {pos_error_3d*1000:.1f} mm")
        except Exception as e:
            print(f"   ❌ IK 3D échoué: {e}")
            continue
        
        print("\n   🔄 Résolution IK 6D (position + orientation)...")
        try:
            q_6d = inverse_kinematics_dh_6d(config, target_pos, target_orient)
            pose_6d = get_end_effector_pose(config, q_6d)
            pos_error_6d = np.linalg.norm(target_pos - pose_6d['position'])
            # Geodesic orientation error (degrees)
            pose_R = euler_deg_to_R(pose_6d['orientation'])
            target_R = euler_deg_to_R(target_orient)
            orient_error_6d = np.rad2deg(np.linalg.norm(rotation_error_rvec(pose_R, target_R)))
            print(f"   ✅ IK 6D: Erreur position = {pos_error_6d*1000:.1f} mm")
            print(f"   ✅ IK 6D: Erreur orientation = {orient_error_6d:.1f}°")
        except Exception as e:
            print(f"   ❌ IK 6D échoué: {e}")
            continue
        
        # Compare results
        print(f"\n   📊 Comparaison:")
        print(f"      Position 3D: {np.round(pos_3d, 4)}")
        print(f"      Position 6D: {np.round(pose_6d['position'], 4)}")
        print(f"      Orientation 6D: {np.round(pose_6d['orientation'], 1)}° (géodésique err: {orient_error_6d:.1f}°)")
        
        # Visualize both results
        print(f"\n   🎨 Génération des visualisations...")
        
        # 3D result
        plot_robot_3d(config, q_3d, pos_target=target_pos, 
                     title_suffix=f" - Test {i+1} - IK 3D")
        plt.savefig(f"test_{i+1}_ik3d.png", dpi=150, bbox_inches='tight'); plt.close()
        
        # 6D result
        plot_robot_3d(config, q_6d, pos_target=target_pos, orient_target=target_orient,
                     title_suffix=f" - Test {i+1} - IK 6D")
        plt.savefig(f"test_{i+1}_ik6d.png", dpi=150, bbox_inches='tight'); plt.close()
        
        print(f"   ✅ Visualisations générées pour le test {i+1}")


def test_orientation_weights():
    """Test different orientation weights"""
    
    print("\n\n⚖️ Test des poids d'orientation")
    print("=" * 50)
    
    config = random_robot_dh(6)
    target_pos = np.array([0.3, 0.2, 0.1], dtype=float)
    target_orient = np.array([0, 0, 45], dtype=float)  # 45 degree yaw
    
    weight_configs = [
        {'weights': [1.0, 1.0, 1.0, 0.1, 0.1, 0.1], 'name': 'Position prioritaire'},
        {'weights': [1.0, 1.0, 1.0, 0.5, 0.5, 0.5], 'name': 'Équilibré'},
        {'weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'name': 'Orientation prioritaire'}
    ]
    
    for weight_config in weight_configs:
        print(f"\n🔧 Configuration: {weight_config['name']}")
        print(f"   Poids: {weight_config['weights']}")
        
        try:
            q = inverse_kinematics_dh_6d(config, target_pos, target_orient, 
                                       weights=weight_config['weights'])
            pose = get_end_effector_pose(config, q)
            
            pos_error = np.linalg.norm(target_pos - pose['position'])
            orient_error = np.linalg.norm(target_orient - pose['orientation'])
            
            print(f"   ✅ Erreur position: {pos_error*1000:.1f} mm")
            print(f"   ✅ Erreur orientation: {orient_error:.1f}°")
            
        except Exception as e:
            print(f"   ❌ Échec: {e}")


if __name__ == '__main__':
    # Set matplotlib backend for headless operation
    import os
    import matplotlib
    os.environ['MPLBACKEND'] = 'Agg'
    matplotlib.use('Agg')
    
    print("🚀 Démarrage des tests IK 6D")
    
    # Run tests
    test_3d_vs_6d_ik()
    test_orientation_weights()
    
    print("\n✅ Tests terminés!")
    print("\n📝 Résumé:")
    print("   - IK 3D: Optimise seulement la position")
    print("   - IK 6D: Optimise position + orientation avec poids")
    print("   - Fallback: Si 6D échoue, utilise 3D automatiquement")
    print("   - Visualisation: Montre erreurs position et orientation")
