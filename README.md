# Conway's Game of Life — Desktop Edition

A high-performance implementation of Conway's Game of Life built with PyQt5 and PyOpenGL. It features both a CPU backend (using NumPy vectorisation) and a GPU backend (using Numba CUDA), with an OpenGL texture rendering pipeline capable of displaying massive grids (e.g., 50,000 × 50,000 cells) smoothly.

## Features

- **Massive Scale**: Limited only by GPU VRAM (up to ~50k × 50k on a 12GB GPU).
- **GPU Acceleration**: Utilizes Numba CUDA for simulation logic, keeping the CPU free.
- **OpenGL Rendering**: Grid is rendered as a texture on a single GPU quad for zero-overhead rendering.
- **Pattern Library**: 22 classic patterns built-in (Gosper Glider Gun, Pulsar, etc.).
- **Tiling Support**: Automatically tile patterns across the entire grid or place a single centered copy.
- **Custom RLE Import**: Directly paste pattern strings from [conwaylife.com](https://conwaylife.com).
- **Benchmarking**: Built-in benchmarking utility to test CPU vs GPU performance.

## Installation

Ensure you have Python 3.10+ installed. A CUDA-capable NVIDIA GPU is highly recommended but not strictly required (it will fall back to CPU mode).

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Launch the GUI application:

```bash
cd backend
source .venv/bin/activate
python run_gui.py
```

### Controls

- **Left-Click**: Draw/toggle cells
- **Middle/Right-Click & Drag**: Pan around the grid
- **Scroll Wheel**: Zoom in/out (pivots around the cursor)

### Keyboard Shortcuts

- `Space` — Toggle Run / Pause
- `S` — Single step forward
- `C` — Clear grid
- `R` — Random fill (uses density slider from the Pattern Library)
- `Ctrl+Q` — Quit
