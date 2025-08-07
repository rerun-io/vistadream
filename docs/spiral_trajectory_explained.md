# Spiral Trajectory Parameters Explained

## Overview
The Spiral trajectory generates camera movement paths for 3D scene visualization. It creates a smooth spiral motion around the scene center, with various parameters controlling the movement characteristics.

## Core Parameters

### 1. Radius Calculation Parameters

#### `traj_min_percentage` (default: 5)
- **What it does**: Controls the lower bound for scene extent calculation
- **How it works**: Uses the 5th percentile of 3D point positions as the minimum bounds
- **Why it matters**: Excludes outlier points that might be noise or far background
- **Example**: If your scene has some far-away background points, this prevents them from making the trajectory too large

#### `traj_max_percentage` (default: 50)  
- **What it does**: Controls the upper bound for scene extent calculation
- **How it works**: Uses the 50th percentile (median) of 3D point positions as maximum bounds
- **Why it matters**: Prevents the trajectory from being too large due to outlier points
- **Example**: If your scene has some very close foreground points, this prevents the trajectory from being too tight

#### `radius` (calculated automatically)
- **Formula**: `mean(max_percentile - min_percentile)` across x, y, z axes
- **What it does**: Sets the base scale for all camera movement
- **How it works**: 
  1. Convert depth map to 3D coordinates using camera intrinsics
  2. Calculate percentile bounds for x, y, z coordinates  
  3. Take the average range across all three axes
- **Visual effect**: Larger radius = camera moves farther from scene center

### 2. Movement Pattern Parameters

#### `rot_ratio` (default: 0.3)
- **What it does**: Controls the tightness/amplitude of the spiral pattern
- **Formula**: `r = sin(2πt) * radius * rot_ratio`
- **Range**: 0.0 to 1.0+ (typically 0.1 - 0.5)
- **Visual effect**:
  - `0.1`: Very tight spiral, camera stays close to center
  - `0.3`: Moderate spiral (default)
  - `0.5`: Wide spiral, more dramatic movement
  - `1.0`: Very wide spiral, camera moves to full radius extent

#### `x_damping` & `y_damping` (orientation-aware)
- **What they do**: Scale the horizontal and vertical movement amplitude
- **Landscape images**: `x_damping=1.0, y_damping=0.3`
  - Full horizontal movement to showcase wide scenes
  - Reduced vertical movement for stability
- **Portrait images**: `x_damping=0.6, y_damping=0.3`
  - Reduced horizontal movement (narrower field of view)
  - Reduced vertical movement to maintain subject focus

### 3. Depth Movement Parameters

#### `forward_ratio` (default: 0.3)
- **What it does**: Controls depth scaling when camera moves toward the scene
- **When applied**: When `z > 0` (camera in front of scene center)
- **Formula**: `z = z * forward_ratio` when `z > 0`
- **Visual effect**: 
  - `0.1`: Camera stays very close to scene center depth
  - `0.3`: Moderate depth variation (default)
  - `0.5`: More pronounced forward movement

#### `backward_ratio` (default: 0.4)
- **What it does**: Controls depth scaling when camera moves away from scene
- **When applied**: When `z < 0` (camera behind scene center)  
- **Formula**: `z = z * backward_ratio` when `z < 0`
- **Visual effect**:
  - `0.1`: Camera stays close to scene center depth
  - `0.4`: Moderate backward movement (default)
  - `0.8`: Camera moves significantly behind scene

### 4. Look-At Parameters

#### `look_at_ratio` (default: 0.5)
- **What it does**: Controls where the camera looks relative to scene center
- **Formula**: `target = [0, 0, radius * look_at_ratio]`
- **Range**: 0.0 to 1.0+ 
- **Visual effect**:
  - `0.0`: Camera always looks at scene center (0,0,0)
  - `0.5`: Camera looks halfway between center and scene edge (default)
  - `1.0`: Camera looks at the scene edge
  - `>1.0`: Camera looks beyond the scene

## Mathematical Breakdown

### Spiral Generation Formula
```python
# Time parameter from 0 to 1
t = np.linspace(0, 1, nframe)

# Radial distance oscillates sinusoidally  
r = sin(2π * t) * radius * rot_ratio

# Angular rotation - full rotations over trajectory
θ = 2π * t * nframe

# 3D position calculation
x = r * cos(θ) * x_damping    # Horizontal position
y = r * sin(θ) * y_damping    # Vertical position
z = -r * (forward_ratio or backward_ratio)  # Depth position
```

### Camera Orientation
- **Up vector**: Fixed to `[0, -1, 0]` (pointing downward in image space)
- **Look-at target**: `[0, 0, radius * look_at_ratio]` (fixed point in front)
- **Camera position**: Moves along the calculated spiral path

## Tuning Guide

### For Tight Indoor Scenes
- `rot_ratio = 0.2`: Smaller spiral for confined spaces
- `forward_ratio = 0.2, backward_ratio = 0.3`: Less depth variation
- `look_at_ratio = 0.3`: Focus closer to center

### For Wide Outdoor Scenes  
- `rot_ratio = 0.4`: Wider spiral to showcase scope
- `forward_ratio = 0.4, backward_ratio = 0.5`: More depth variation
- `look_at_ratio = 0.7`: Look toward scene edges

### For Portrait Subjects
- `x_damping = 0.6, y_damping = 0.3`: Constrained movement
- `rot_ratio = 0.25`: Tighter spiral to stay focused on subject
- `look_at_ratio = 0.4`: Keep focus on subject area

### For Landscape Scenes
- `x_damping = 1.0, y_damping = 0.3`: Full horizontal movement
- `rot_ratio = 0.35`: Wider spiral to showcase breadth  
- `look_at_ratio = 0.6`: Explore scene extent

## Common Issues & Solutions

**Problem**: Camera moves too far from scene
- **Solution**: Reduce `rot_ratio` or adjust `traj_max_percentage`

**Problem**: Camera movement too jerky
- **Solution**: Increase `nframe` for smoother interpolation

**Problem**: Camera looks at wrong area
- **Solution**: Adjust `look_at_ratio` to focus on desired region

**Problem**: Too much/little depth variation
- **Solution**: Tune `forward_ratio` and `backward_ratio`

**Problem**: Scene bounds calculated incorrectly
- **Solution**: Adjust `traj_min_percentage` and `traj_max_percentage` to exclude outliers
