# Repository Guidelines

## Project Structure & Module Organization
- `src/vistadream/`: Core package
  - `api/`: Pipelines and CLIs (`single_img_pipeline.py`, `flux_outpainting.py`)
  - `flux/`: Flux integration (CLIs, `model.py`, `sampling.py`, `util.py`)
  - `ops/`: Core ops (`ops/gs` Gaussians, `ops/trajs` camera paths)
  - `resize_utils.py`: Resizing and border/mask helpers
- `tools/`: Entry scripts (`run_single_img.py`, `run_vistadream.py`, `gradio_app.py`)
- `ckpt/`: Model weights (auto-downloaded); `data/`: inputs/examples

## Build, Test, and Development Commands
- Always use Pixi; avoid pip/conda/venv outside Pixi
- Enter the dev env with `pixi shell -e dev` or run one-shot commands with `pixi run -e dev <command>`.
- Environment and example run: `pixi run example`
- Full pipeline: `pixi run python tools/run_vistadream.py --image-path <img> --n-frames 10 --expansion-percent 0.2`
- Outpainting only: `pixi run python tools/run_flux_outpainting.py --image-path <img> --expansion-percent 0.2`
- Gradio UI: `pixi run python tools/gradio_app.py`
- Lint/format: `pixi run -e dev ruff check . --fix`

## Coding Style & Naming Conventions
- Python 3.12 via Pixi; 4-space indent; max line length 150 (see `pyproject.toml`)
- Strong typing required: use Python type hints and jaxtyping for arrays
  - Example: `rgb_hw3: UInt8[np.ndarray, "h w 3"]`, `depth: Float[np.ndarray, "h w"]`
- Names: `snake_case` for funcs/modules, `PascalCase` for classes, `UPPER_SNAKE` constants
- Dev runtime checks: enabled automatically whenever the Pixi dev env is active (`pixi shell -e dev` or `pixi run -e dev`)
- Image dims must be multiples of 32; respect MP limits in resizing

## Testing Guidelines
- Put tests in `tests/` as `test_*.py`; use `pytest`
- Prefer deterministic units (e.g., `resize_utils`, math in `ops/trajs`)
- Mock CUDA/IO; keep CPU-fast. Run with `pixi run -e dev pytest -q`
- For scene ops: validate masks (e.g., `inpaint_wo_edge`) and shapes/dtypes

## Commit & Pull Request Guidelines
- Commits: imperative subject + details
  - Example: `Refactor trajectory sampling; add edge masking`
- PRs include: purpose, CLI output/screenshots, repro steps, linked issues
- Required: Ruff clean, scripts runnable (`tools/*`), docs updated when behavior changes

## Security & Configuration Tips
- CUDA 12.x GPU recommended; use offloading flags where provided
- Offload pattern: CPU → GPU → process → CPU; call `torch.cuda.empty_cache()` after transfers
- Checkpoints under `ckpt/` (via `huggingface-cli`); don’t commit weights or large data
