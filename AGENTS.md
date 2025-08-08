# Repository Guidelines

## Project Structure & Module Organization
- `src/vistadream/`: Core package.
  - `api/`: Pipelines & CLIs (e.g., `single_img_pipeline.py`, `flux_outpainting.py`).
  - `flux/`: Flux integration (`model.py`, `sampling.py`, `util.py`, CLIs).
  - `ops/`: Core ops (`ops/gs` Gaussians, `ops/trajs` camera paths).
  - `resize_utils.py`: Resizing and border/mask helpers.
- `tools/`: Entry scripts (`run_single_img.py`, `run_vistadream.py`, `gradio_app.py`).
- `ckpt/`: Model weights (auto-downloaded). `data/`: inputs/examples.
- `tests/`: Unit tests as `test_*.py`.

## Build, Test, and Development Commands
- Dev env: `pixi shell -e dev` for an interactive shell; or one-shot `pixi run -e dev <command>`.
- Quick example run: `pixi run example`.
- Full pipeline: `pixi run python tools/run_vistadream.py --image-path <img> --n-frames 10 --expansion-percent 0.2`.
- Outpainting only: `pixi run python tools/run_flux_outpainting.py --image-path <img> --expansion-percent 0.2`.
- Gradio UI: `pixi run python tools/gradio_app.py`.
- Lint/format: `pixi run -e dev ruff check . --fix`.

## Coding Style & Naming Conventions
- Python 3.12; 4‑space indent; max line length 150.
- Strong typing required: use type hints and jaxtyping (e.g., `rgb_hw3: UInt8[np.ndarray, "h w 3"]`, `depth: Float[np.ndarray, "h w"]`).
- Naming: `snake_case` funcs/modules, `PascalCase` classes, `UPPER_SNAKE` constants.
- Dev runtime checks enable automatically in the Pixi dev env.
- Image dims must be multiples of 32; respect MP limits when resizing.

## Testing Guidelines
- Framework: `pytest`; place tests in `tests/` as `test_*.py`.
- Run tests: `pixi run -e dev pytest -q`.
- Prefer deterministic CPU-fast units; mock CUDA/IO.
- For scene ops, validate masks (e.g., `inpaint_wo_edge`) and confirm shapes/dtypes.

## Commit & Pull Request Guidelines
- Commits: imperative subject + details (e.g., "Refactor trajectory sampling; add edge masking").
- PRs: include purpose, CLI output/screenshots, repro steps, and linked issues.
- Requirements: ruff clean, scripts under `tools/*` runnable, docs updated when behavior changes.

## Security & Configuration Tips
- CUDA 12.x GPU recommended; offload CPU → GPU → process → CPU; call `torch.cuda.empty_cache()` after transfers.
- Checkpoints live under `ckpt/` (via `huggingface-cli`); do not commit weights or large data.

