#!/usr/bin/env python3
"""
Generate improved IK training dataset: 10k samples from SET_D
Improved version with more samples for better MLP performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import importlib.util
import time
from multiprocessing import Pool, cpu_count

# Load dh_utils (2).py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DH_UTILS2_PATH = os.path.join(THIS_DIR, "dh_utils (2).py")
spec = importlib.util.spec_from_file_location("dh2", DH_UTILS2_PATH)
dh2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dh2)
sys.modules['dh_utils'] = dh2

from module_catalog import get_module_catalog
from dls_ik_baseline import (
    forward_kinematics,
    inverse_kinematics_dls,
    euler_to_rotation_matrix,
)
from generate_seed_dataset_poc import (
    quaternion_from_rotation_matrix,
    encode_dh_params,
    generate_sample,
)


def worker_generate_sample(args):
    """Worker function for parallel sample generation."""
    config, seed = args
    np.random.seed(seed)  # Ensure different random samples per worker
    return generate_sample(config)


def main():
    print("🧠 GÉNÉRATION DATASET AMÉLIORÉ - 10K SAMPLES (PARALLÈLE)")
    print("="*70)
    
    # Get SET_D configuration
    catalog = get_module_catalog()
    config = catalog['SET_D_EXTENDED_REACH'].config
    
    n_cores = cpu_count()
    print(f"\n💻 CPU Cores: {n_cores}")
    print(f"Configuration: SET_D Extended Reach")
    print(f"DDL: {len(config)}")
    print(f"Modules: {[j['type'] for j in config]}")
    
    # Generate samples
    n_samples = 10000
    save_path = 'ik_seed_dataset_10k.npz'
    
    # Estimate speedup
    estimated_time_serial = n_samples * 0.9 / 60
    estimated_time_parallel = estimated_time_serial / (n_cores * 0.7)  # 70% efficiency
    
    print(f"\n🎲 Génération {n_samples} échantillons...")
    print(f"Temps estimé (série): ~{estimated_time_serial:.0f} minutes")
    print(f"Temps estimé (parallèle {n_cores} cores): ~{estimated_time_parallel:.0f} minutes")
    print(f"⚡ Accélération estimée: {n_cores * 0.7:.1f}×")
    print()
    
    start_time = time.time()
    
    # Prepare arguments for workers (config + unique seed per sample)
    worker_args = [(config, i) for i in range(n_samples)]
    
    # Parallel processing
    print(f"🚀 Lancement traitement parallèle...")
    with Pool(processes=n_cores) as pool:
        try:
            from tqdm import tqdm
            results = list(tqdm(pool.imap(worker_generate_sample, worker_args), 
                              total=n_samples, desc="Generating samples"))
        except ImportError:
            print("Génération en cours (installez tqdm pour barre de progression)...")
            results = pool.map(worker_generate_sample, worker_args)
            
            # Manual progress updates
            for i in range(0, n_samples, 1000):
                if i > 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed
                    eta = (n_samples - i) / rate / 60
                    print(f"  {i}/{n_samples} ({i/n_samples*100:.0f}%) - ETA: {eta:.0f} min")
    
    # Filter successful samples
    features_list = []
    labels_list = []
    metadata_list = []
    failed = 0
    
    for sample in results:
        if sample['success']:
            features_list.append(sample['features'])
            labels_list.append(sample['label'])
            metadata_list.append({
                'pos_error_mm': sample['pos_error_mm'],
                'solve_time_ms': sample['solve_time_ms'],
            })
        else:
            failed += 1
    
    # Convert to arrays
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    # Compute normalization statistics
    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0) + 1e-8
    label_mean = np.mean(labels, axis=0)
    label_std = np.std(labels, axis=0) + 1e-8
    
    # Save dataset
    np.savez(
        save_path,
        features=features,
        labels=labels,
        feature_mean=feature_mean,
        feature_std=feature_std,
        label_mean=label_mean,
        label_std=label_std,
        metadata=metadata_list,
    )
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"✅ Dataset Generated Successfully")
    print(f"{'='*70}")
    print(f"Samples: {len(features)}")
    print(f"Failed: {failed}")
    print(f"Success rate: {len(features)/(len(features)+failed)*100:.1f}%")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Label dimension: {labels.shape[1]}")
    print(f"\nSaved to: {save_path}")
    print(f"File size: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
    
    # Dataset statistics
    print(f"\n📊 Dataset Statistics:")
    pos_errors = [m['pos_error_mm'] for m in metadata_list]
    solve_times = [m['solve_time_ms'] for m in metadata_list]
    
    print(f"DLS Solution Quality:")
    print(f"  Mean position error: {np.mean(pos_errors):.3f} mm")
    print(f"  Median: {np.median(pos_errors):.3f} mm")
    print(f"  Max: {np.max(pos_errors):.3f} mm")
    
    print(f"DLS Solve Time:")
    print(f"  Mean: {np.mean(solve_times):.1f} ms")
    print(f"  Median: {np.median(solve_times):.1f} ms")
    
    print(f"\nLabel Statistics (Δq in degrees):")
    print(f"  Mean |Δq|: {np.mean(np.abs(labels)):.2f}°")
    print(f"  Max |Δq|:  {np.max(np.abs(labels)):.2f}°")
    
    print(f"\n⏱️  Temps total: {total_time/60:.1f} minutes")
    print(f"   Temps par échantillon: {total_time/n_samples*1000:.1f} ms")
    
    print("\n✅ DATASET 10K PRÊT")
    print("Prochaine étape: python train_improved_network.py")


if __name__ == "__main__":
    main()

