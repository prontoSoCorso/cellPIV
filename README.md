# cellPIV

Particle Image Velocimetry (PIV) for estimating velocity fields in time‑lapse microscopy of cells.

- Dense, windowed cross‑correlation with multi‑pass refinement
- Optional masking, validation, smoothing, and visualization
- CLI and Python API
- Saves vectors, magnitude, and figures

## Installation

- From source (recommended during development):
```bash
git clone <your-repo-url>.git
cd cellPIV
python -m venv .venv && source .venv/bin/activate  # or use conda
pip install -U pip
pip install -e .  # editable install
# optional: pip install -r requirements.txt
```

- From PyPI (if published):
```bash
pip install cellpiv
```

## Quick start

- CLI
```bash
cellpiv \
    --input data/stack.tif \
    --window 32 --overlap 16 \
    --dt 1.0 \
    --out results/ \
    --plot
```

- Python
```python
# minimal example; adjust to your actual API
from cellpiv import run_piv, io

imgs = io.load_stack("data/stack.tif")  # 2D+T stack
field = run_piv(
        imgs,
        window=32,
        overlap=16,
        dt=1.0,
        preproc=dict(denoise=True, equalize=True),
        postproc=dict(outlier_filter=True, smooth=True),
)
field.save("results/")  # vectors, magnitude, figures
```

## Inputs and outputs

- Inputs: 2D+T image stacks (TIFF/PNG sequence), optional ROI mask
- Outputs: velocity components (u, v), speed, divergence/curl (optional), CSV/NumPy, quiver plots

## Configuration

Use flags or a YAML file:
```yaml
# piv.yaml
window: 32
overlap: 16
dt: 1.0
preproc:
    denoise: true
    equalize: true
postproc:
    outlier_filter: true
    smooth: true
io:
    input: data/stack.tif
    outdir: results/
    plot: true
```
```bash
cellpiv --config piv.yaml
```

## Project structure

- cellpiv/ … library and CLI
- examples/ … small datasets and scripts
- tests/ … unit tests
- docs/ … additional documentation

## Development

- Run tests:
```bash
pytest -q
```
- Lint/format:
```bash
ruff check . && ruff format .
```

## Contributing

- Open an issue before large changes
- Fork, create a feature branch, add tests, open a PR

## Troubleshooting

- Noisy vectors: increase window size, enable smoothing/outlier filtering
- Low signal: denoise and contrast‑enhance; check dt
- Bad scaling: verify pixel size and dt

## Citation

If this helps your research, please cite this repository (add DOI when available).

## License

Specify your license here (e.g., MIT). See LICENSE.