# Modular Robot 6D Inverse Kinematics System

Production-ready 6D IK solver for modular robotic arms with vision integration capabilities.

##  Project Overview

This system provides:
-  **Validated 6D IK solver** (position + orientation control)
-  **Module catalog** with 5 pre-validated robot configurations
-  **Sub-5mm position accuracy, <1° orientation accuracy**
-  **Ready for vision integration** (YOLO + depth estimation)
-  **Optional ML acceleration** (experimental hybrid solver)

##  Performance Highlights

| Robot Set | Position Error | Orientation Error | Use Case |
|-----------|---------------|-------------------|----------|
| **Set D** (Extended) | **0.4mm**  | <0.001° | Best precision, large workspace |
| **Set A** (Full 6D) | **2.4mm**  | <0.001° | General purpose, vision-guided |
| **Set E** (Compact) | **4.3mm**  | <0.001° | Tight spaces, desktop |

Validated against industrial standards (UR5, PUMA560).

##  Quick Start

### Prerequisites

```bash
Python 3.10+
pip install numpy matplotlib
```

### Optional (for ML features):
```bash
pip install torch tqdm  # PyTorch + progress bars
```

### Basic Usage

```python
from module_catalog import get_module_catalog
from dls_ik_baseline import inverse_kinematics_dls, euler_to_rotation_matrix

# 1. Select robot configuration
catalog = get_module_catalog()
config = catalog['SET_D_EXTENDED_REACH'].config  # Best precision

# 2. Define target pose
target_pos = [0.40, 0.10, 0.20]  # meters [x, y, z]
target_euler = [0, 0, 45]  # degrees [roll, pitch, yaw]
R_target = euler_to_rotation_matrix(*target_euler)

# 3. Solve IK
q_solution = inverse_kinematics_dls(config, target_pos, R_target)

# 4. Result: joint angles in degrees
print(f"Joint angles: {q_solution}")
```

##  Project Structure

```
ProjetFilRouge/
├── Core IK Solver
│   ├── dls_ik_baseline.py          #  Main DLS IK solver (production)
│   ├── module_catalog.py           #  Pre-validated robot configurations
│   └── dh_utils (2).py             # DH parameter generator
│
├── Validation & Testing
│   ├── validate_set_d_roundtrip.py # FK→IK→FK validation (100 tests)
│   ├── visualize_catalog_results.py # 3D visualization generator
│   └── adaptive_modular_ik.py      # Auto-capability detection
│
├── ML Acceleration (Experimental)
│   ├── generate_improved_dataset.py # Dataset generator (parallel)
│   ├── train_improved_network.py    # Train 128-128 MLP
│   └── test_hybrid_solver_improved.py # Hybrid MLP+DLS testing
│
├── Legacy/Development
│   ├── kinematics.py               # Original 3D IK + 6D extensions
│   ├── plot_robot.py               # 3D plotting utilities
│   ├── ik_diagnostics.py           # Development/debugging tools
│   └── IKsolverNode (1).py         # ROS2 integration (WIP)
│
├── Documentation
│   ├── TECHNICAL_DOCUMENTATION.md  # Full technical docs (English)
│   ├── DOCUMENTATION_TECHNIQUE.md  # Full technical docs (French)
│   └── SOLUTION_RESUME.md          # Summary for colleagues
│
└── Generated Results
    ├── catalog_*.png               # Validation visualizations
    ├── SET_D_*.png                 # Roundtrip test results
    └── hybrid_*.png                # ML experiment results
```

##  Validation & Testing

### Validate Catalog Sets
```bash
python module_catalog.py
```
Tests all 5 catalog sets against workspace-appropriate target poses.

### Validate SET_D (100 roundtrip tests)
```bash
python validate_set_d_roundtrip.py
```
Generates comprehensive validation with 100 FK→IK→FK tests.

### Generate 3D Visualizations
```bash
python visualize_catalog_results.py
```
Creates publication-quality 3D plots of IK results.

##  ML Acceleration (Optional - Requires GPU)

### PoC Pipeline (5k samples, 64-64 model)
```bash
# 1. Generate dataset (~75 min serial, ~10 min on 8+ cores)
python generate_seed_dataset_poc.py

# 2. Train model (~10 min CPU, ~2 min GPU)
python train_seed_network_poc.py

# 3. Test hybrid solver
python test_hybrid_solver_poc.py
```

### Improved Pipeline (10k samples, 128-128 model)
```bash
# 1. Generate dataset (parallel - uses all CPU cores)
python generate_improved_dataset.py  # ~20-30 min on multi-core

# 2. Train improved model (~60 min CPU, ~10 min GPU)
python train_improved_network.py

# 3. Test improved hybrid
python test_hybrid_solver_improved.py
```

**Current Results**:
- MLP prediction error: 13.77° (good)
- Speedup: 1.5× (marginal)
- Accuracy: 5mm (acceptable)
- **Status**: Experimental - pure DLS recommended for production

##  Documentation

- **TECHNICAL_DOCUMENTATION.md**: Complete technical reference (46KB, English)
- **DOCUMENTATION_TECHNIQUE.md**: Complete technical reference (34KB, French)
- **SOLUTION_RESUME.md**: Executive summary for stakeholders

##  Module Catalog

5 pre-validated robot configurations:

### Set A: Full 6D Precision
- **Performance**: 2.4mm position, <0.001° orientation
- **Reach**: 0.64m
- **Use**: Vision-guided pick-and-place

### Set D: Extended Reach  (Best)
- **Performance**: 0.4mm position, <0.001° orientation
- **Reach**: 0.77m
- **Use**: Large workspace, maximum precision

### Set E: Compact Precision
- **Performance**: 4.3mm position, <0.001° orientation
- **Reach**: 0.47m
- **Use**: Desktop, confined spaces

### Set B: Partial 6D (5-DOF)
- **Performance**: 7.3mm position, <0.3° orientation
- **Reach**: 0.57m
- **Use**: Cost-effective pick-and-place

### Set C: SCARA (4-DOF)
- **Performance**: 59mm position, yaw-only
- **Reach**: 0.51m (planar)
- **Use**: High-speed horizontal assembly

##  Visualizations Included

The repository includes pre-generated validation visualizations:
- `catalog_SET_*_visualization.png`: Individual set validations
- `catalog_comparison_all_sets.png`: Side-by-side comparison
- `SET_D_roundtrip_*.png`: 100-test validation results
- `hybrid_solver_*.png`: ML experiment results

##  Technical Details

### IK Method: Damped Least Squares (DLS)

```
Δq = (JᵀJ + λ²I)⁻¹ Jᵀ e

Where:
- J: 6×n Jacobian (position + orientation)
- e: 6D error using cross-product orientation error
- λ: damping factor (0.01)
```

**Key Innovation**: Cross-product orientation error
- Smooth, convex (no gimbal lock)
- Unit-balanced (no artificial scaling)
- Industry-standard formulation

### Spherical Wrist Requirement

For full 6D control, last 3 joints must form spherical wrist:
- All `a = 0` (intersecting axes)
- Alphas: `[+π/2, -π/2, 0]` pattern
- Result: σmin(Jori) > 0.7

Random module combinations typically have σmin < 0.02 → catalog approach essential.

##  Troubleshooting

### High Position Errors
- Check if target is reachable: `||target|| < 0.95 × reach`
- Increase iterations: `max_iter=2000`
- Use multi-restart strategy

### High Orientation Errors
- Verify spherical wrist (check last 3 joints have a=0)
- Check σmin(Jori) > 0.7
- Use catalog sets (validated geometries)

##  Citation

If you use this work, please reference:
- Nakamura & Hanafusa (1986): DLS formulation
- Siciliano et al. (2009): Cross-product orientation error

##  Contributing

Current maintainer: Rayan
Original DH generator: Colleague

##  License

[Add your license here]

##  Future Work

- [ ] Real-time vision integration (YOLO + monocular depth)
- [ ] Obstacle avoidance with path planning
- [ ] ROS2 full integration
- [ ] Improved ML model (>3× speedup target)
- [ ] Multi-arm coordination

---

**Status**: Production-Ready   
**Version**: 1.0  
**Last Updated**: October 30, 2025

