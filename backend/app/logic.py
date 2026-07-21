from __future__ import annotations

import numpy as np

try:
    from numba import cuda

    CUDA_AVAILABLE = bool(cuda.is_available())
except Exception:  # pragma: no cover
    cuda = None
    CUDA_AVAILABLE = False


def _normalize_grid(grid: np.ndarray) -> np.ndarray:
    data = np.asarray(grid, dtype=np.uint8)
    return (data > 0).astype(np.uint8)


def next_generation_cpu(grid: np.ndarray) -> np.ndarray:
    grid = _normalize_grid(grid)
    neighbors = sum(
        np.roll(np.roll(grid, i, axis=0), j, axis=1)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        if not (i == 0 and j == 0)
    )
    survives = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
    born = (grid == 0) & (neighbors == 3)
    return (survives | born).astype(np.uint8)


if CUDA_AVAILABLE:

    @cuda.jit
    def _next_generation_kernel(grid, output, rows, cols):
        row, col = cuda.grid(2)
        if row >= rows or col >= cols:
            return

        alive_neighbors = 0
        for d_row in range(-1, 2):
            for d_col in range(-1, 2):
                if d_row == 0 and d_col == 0:
                    continue
                n_row = (row + d_row + rows) % rows
                n_col = (col + d_col + cols) % cols
                alive_neighbors += grid[n_row, n_col]

        cell_alive = grid[row, col]
        if cell_alive == 1 and (alive_neighbors == 2 or alive_neighbors == 3):
            output[row, col] = 1
        elif cell_alive == 0 and alive_neighbors == 3:
            output[row, col] = 1
        else:
            output[row, col] = 0


def next_generation_gpu(grid: np.ndarray) -> np.ndarray:
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA is not available")

    grid = _normalize_grid(grid)
    rows, cols = grid.shape

    device_grid = cuda.to_device(grid)
    device_output = cuda.device_array((rows, cols), dtype=np.uint8)

    threads_per_block = (16, 16)
    blocks_per_grid = (
        (rows + threads_per_block[0] - 1) // threads_per_block[0],
        (cols + threads_per_block[1] - 1) // threads_per_block[1],
    )

    _next_generation_kernel[blocks_per_grid, threads_per_block](
        device_grid, device_output, rows, cols
    )
    cuda.synchronize()

    return device_output.copy_to_host()


def compute_next_generation(grid: np.ndarray, mode: str = "cpu") -> tuple[np.ndarray, str]:
    normalized_mode = str(mode or "cpu").lower()

    if normalized_mode == "gpu":
        try:
            return next_generation_gpu(grid), "gpu"
        except Exception:
            return next_generation_cpu(grid), "cpu"

    return next_generation_cpu(grid), "cpu"
