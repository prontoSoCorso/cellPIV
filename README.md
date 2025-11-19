# cellPIV

Particle Image Velocimetry (PIV) for estimating velocity fields in time‑lapse microscopy of cells.

- Dense, windowed cross‑correlation with multi‑pass refinement
- Optional masking, validation, smoothing, and visualization
- Saves vectors, magnitude, and figures
- Blasto prediction within day 1, 3, 5
- Grad-Cam explainability, local and global

## Installation

## Quick start

## Inputs and outputs

- Inputs: 2D+T image stacks (TIFF/PNG sequence)
- Outputs: velocity components (u, v), speed, divergence/curl, CSV/NumPy, quiver plots

## Configuration

## Project structure

## Development

- Run tests:

## Contributing


## Troubleshooting

- Noisy vectors: increase window size, enable smoothing/outlier filtering
- Low signal: denoise and contrast‑enhance; check dt
- Bad scaling: verify pixel size and dt

## Citation

If this helps your research, please cite this repository

## License
