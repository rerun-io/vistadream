# VistaDream Code Style and Development Guidelines

This document outlines the coding standards, conventions, and development practices for the VistaDream project. These rules are designed to ensure code quality, maintainability, and consistency across the codebase.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Core Architecture](#core-architecture)
3. [Type Annotations](#type-annotations)
4. [Development Workflow](#development-workflow)
5. [Project-Specific Conventions](#project-specific-conventions)
6. [File Organization](#file-organization)
7. [Memory Management](#memory-management)
8. [Integration Points](#integration-points)

## Project Overview

VistaDream is a Flux-based diffusion model for image outpainting and 3D scene generation, integrated with Rerun for visualization. The project uses Pixi for package management and CUDA 12.9 for GPU acceleration.

## Core Architecture

### Model System (`src/vistadream/flux/`)
- **Entry Point**: Multiple CLI interfaces (`cli.py`, `cli_fill.py`, `cli_control.py`, `cli_redux.py`) 
- **Model Loading**: `util.py` contains model configurations and loading functions
- **Inference Pipeline**: `sampling.py` handles diffusion sampling, `model.py` defines Flux transformer
- **Model Variants**: `flux-dev`, `flux-schnell`, `flux-dev-fill`, `flux-dev-depth`, `flux-dev-canny`
- **Memory Management**: All CLIs support `--offload` to move models between CPU/GPU for memory efficiency

### 3D Scene Generation (`src/vistadream/ops/gs/`)
- **Frame System**: `Frame` class handles camera parameters, RGB, depth, and inpainting masks
- **Gaussian Splatting**: `Gaussian_Scene` manages 3D Gaussians for scene representation
- **Training Loop**: `GS_Train_Tool` optimizes Gaussians using RGB/SSIM losses
- **Critical**: Always set `frame.inpaint_wo_edge` before adding to scene to avoid NoneType errors

### VistaDream Pipeline (`src/vistadream/api/vistadream_pipeline.py`)
- **Main Pipeline**: Combines outpainting → depth prediction → 3D scene generation
- **Trajectory Generation**: Uses `_generate_trajectory()` for camera movement
- **Rerun Integration**: Dynamic blueprint updates with `logged_cam_idx_list`
- **Frame Processing**: Inpaints every 5th frame, adds to Gaussian scene

### API Layer (`src/vistadream/api/`)
- **Outpainting API**: `flux_outpainting.py` - standalone outpainting with Rerun
- **VistaDream Pipeline**: `vistadream_pipeline.py` - full 3D pipeline 
- **Configuration Pattern**: Uses `@dataclass` configs with tyro for CLI generation
- **Image Processing**: Ensures dimensions are multiples of 32, handles megapixel limits

### Operations (`src/vistadream/ops/`)
- **Flux Integration**: `flux.py` wraps Flux models for inpainting
- **Trajectory Generation**: `trajs/` module for camera movement (spiral, wobble, interp)
- **Scene Utilities**: `utils.py` for point cloud operations and depth processing
- **Visual Check**: `visual_check.py` renders videos from Gaussian scenes

## Type Annotations

### Runtime Type Checking with Beartype
The project uses beartype for runtime type validation, automatically activated in `src/vistadream/__init__.py`:
```python
from beartype.claw import beartype_this_package
beartype_this_package()
```
- Full runtime type checking enabled across all modules
- No need to manually add `@beartype` decorators
- Automatic validation of function parameters and return types

### Array Type Annotations with JAXTyping
**Critical**: Every array must specify both dtype and shape using jaxtyping, and variables should be annotated at assignment time:
```python
from jaxtyping import Float, UInt8, Bool, BFloat16
import torch
import numpy as np

# Variable assignment annotations (PEP 526 style)
rgb_hw3: UInt8[np.ndarray, "h w 3"] = ...
depth: Float[np.ndarray, "h w"] = ...
mask: Bool[np.ndarray, "h w"] = ...
latents: BFloat16[torch.Tensor, "batch channels latent_height latent_width"] = ...
intrinsics: Float[np.ndarray, "3 3"] = ...
```

### Function Signatures with JAXTyping
Follow the established pattern from `ops/flux.py` and `ops/gs/basic.py`:
```python
def __call__(
    self,
    rgb_hw3: UInt8[np.ndarray, "h w 3"],
    mask: UInt8[np.ndarray, "h w"],
) -> Image.Image:
    height: int = rgb_hw3.shape[0]
    width: int = rgb_hw3.shape[1]
    x: BFloat16[torch.Tensor, "batch channels latent_height latent_width"] = get_noise(...)
```

### Dataclass Type Annotations
Frame classes use extensive jaxtyping annotations:
```python
@dataclass
class Frame:
    H: int | None = None
    W: int | None = None
    rgb: np.ndarray | None = None
    dpt: np.ndarray | None = None
    inpaint: Bool[ndarray, "H W"] | None = None
    inpaint_wo_edge: Bool[ndarray, "H W"] | None = None
    intrinsic: Float[ndarray, "3 3"] | None = None
    cam_T_world: Float[ndarray, "4 4"] | None = None
```

References: [beartype docs](https://beartype.readthedocs.io/en/latest/), [jaxtyping docs](https://docs.kidger.site/jaxtyping/)

## Development Workflow

### Environment Setup
```bash
# Primary commands - use pixi for all package management
pixi install              # Install all dependencies
pixi run python <script>  # Run scripts in pixi environment
```

### Model Checkpoints
Models expect checkpoints in `./ckpt/` directory:
- `./ckpt/flux_fill/flux1-fill-dev.safetensors`
- `./ckpt/flux_fill/ae.safetensors` 
- Environment variables: `FLUX_DEV`, `FLUX_SCHNELL`, `AE` for custom paths

### Running Components
```bash
# VistaDream pipeline with 3D scene generation
pixi run python tools/run_vistadream.py --image-path <path> --expansion-percent 0.2 --n-frames 10

# Outpainting only with Rerun visualization
pixi run python tools/run_flux_outpainting.py --image-path <path> --expansion-percent 0.2

# Gradio web interface
pixi run python tools/gradio_app.py
```

## Project-Specific Conventions

### Frame Management Pattern
```python
# Critical: Always set inpaint_wo_edge before adding to scene
frame.inpaint_wo_edge = mask_wo_edges
scene._add_trainable_frame(frame)
```

### File Organization Pattern
- `cli_*.py` - Command-line interfaces with interactive loops
- `api/*.py` - High-level APIs with configuration dataclasses  
- `ops/*.py` - Core operations (flux, gaussian splatting, trajectories)
- `tools/*.py` - Standalone applications

### Memory Management
- **Critical**: Always use `torch.cuda.empty_cache()` after moving models
- **Pattern**: CPU → GPU → process → CPU cycle for large models
- **Dimensions**: Ensure all image dimensions are multiples of 32

### Rerun Blueprint Pattern
- **Dynamic Updates**: Call `rr.send_blueprint()` when `logged_cam_idx_list` changes
- **3D Content Exclusions**: Use `"-"` prefix to hide depth/mask/rgb from 3D view
- **View Distribution**: Sample max 5 cameras evenly for grid view

## File Organization

The project follows a specific organizational structure:

```
├── src/vistadream/
│   ├── api/                 # High-level pipeline APIs
│   │   ├── flux_outpainting.py    # Outpainting-only pipeline
│   │   └── vistadream_pipeline.py # Full 3D reconstruction pipeline
│   ├── flux/                # Flux diffusion model integration
│   │   ├── cli_*.py         # Command-line interfaces
│   │   ├── model.py         # Flux transformer architecture
│   │   ├── sampling.py      # Diffusion sampling logic
│   │   └── util.py          # Model loading and configuration
│   └── ops/                 # Core operations
│       ├── flux.py          # Flux model wrappers
│       ├── gs/              # Gaussian splatting implementation
│       ├── trajs/           # Camera trajectory generation
│       └── visual_check.py  # 3D scene validation tools
└── tools/                   # Standalone applications
    ├── gradio_app.py        # Web interface
    ├── run_flux_outpainting.py
    ├── run_vistadream.py    # Main 3D pipeline
    └── run_single_img.py    # Single image processing
```

## Memory Management

Proper memory management is crucial for the VistaDream project due to its large model sizes:

### Model Offloading Pattern
```python
# Standard offloading pattern used throughout
if offload:
    ae = ae.to(torch_device)  # Move to GPU for encoding
    # ... use model ...
    ae = ae.cpu()             # Move back to CPU
    torch.cuda.empty_cache()  # Clear CUDA memory
    model = model.to(torch_device)  # Move next model to GPU
```

### Best Practices
1. Always use `torch.cuda.empty_cache()` after moving models between CPU and GPU
2. Follow the CPU → GPU → process → CPU cycle for large models
3. Ensure all image dimensions are multiples of 32 for optimal performance

## Integration Points

### Rerun
- Blueprint-based 3D visualization with dynamic camera management
- Dynamic blueprint updates with `logged_cam_idx_list`
- Use `"-"` prefix to hide depth/mask/rgb from 3D view

### Pixi
- All dependency management, no pip/conda commands
- Environment setup with `pixi install`

### Tyro
- Automatic CLI generation from dataclasses
- Configuration pattern using `@dataclass` with tyro

### GSplat
- Gaussian splatting backend for 3D scene representation

## Key Files for Understanding
- `src/vistadream/flux/util.py` - Model configurations and loading logic
- `src/vistadream/api/vistadream_pipeline.py` - Main 3D pipeline
- `src/vistadream/ops/gs/basic.py` - Frame and Gaussian scene classes
- `src/vistadream/ops/gs/train.py` - Gaussian optimization training
- `pyproject.toml` - Pixi configuration with CUDA dependencies
- `tools/run_vistadream.py` - Entry point for full 3D pipeline

## Common Tasks
1. **Adding new model variant**: Update `configs` dict in `util.py`
2. **New CLI interface**: Follow pattern in existing `cli_*.py` files
3. **Memory optimization**: Implement offloading pattern consistently
4. **Image processing**: Always ensure 32-pixel alignment and MP limits
5. **3D scene debugging**: Use `visual_check.py` to render videos and point clouds
