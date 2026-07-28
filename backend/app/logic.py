from __future__ import annotations

import os
import warnings

# Use the NVIDIA cuda-python binding instead of Numba's ctypes wrapper.
# This avoids the cuCtxSynchronize_v2 context-destroyed error (709) on
# CUDA 13.0+ drivers.
# IMPORTANT: Must be set BEFORE any numba import.
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import numpy as np
from scipy.signal import convolve2d
from numba import njit, prange

try:
    from numba import cuda
    from numba.core.errors import NumbaPerformanceWarning
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

    CUDA_AVAILABLE = bool(cuda.is_available())
except Exception:  # pragma: no cover
    cuda = None
    CUDA_AVAILABLE = False


def _normalize_grid(grid: np.ndarray) -> np.ndarray:
    data = np.asarray(grid, dtype=np.uint8)
    return (data > 0).astype(np.uint8)


def next_generation_numpy(grid: np.ndarray) -> np.ndarray:
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

def next_generation_scipy(grid: np.ndarray) -> np.ndarray:
    grid = _normalize_grid(grid)
    
    # 1. Define the neighbor-counting kernel
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    # 2. Count neighbors using convolution
    neighbors = convolve2d(grid, kernel, mode='same', boundary='wrap')
    
    # 3. Apply Conway's rules (unchanged)
    survives = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
    born = (grid == 0) & (neighbors == 3)
    
    return (survives | born).astype(np.uint8)

def next_generation_loop(grid: np.ndarray) -> np.ndarray:
    grid = _normalize_grid(grid)
    rows, cols = grid.shape
    new_grid = np.zeros_like(grid)

    for i in range(rows):
        for j in range(cols):
            live_neighbors = 0
            # Check 8 neighbors
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    if x == 0 and y == 0:
                        continue
                    neighbor_x = (i + x) % rows
                    neighbor_y = (j + y) % cols
                    if grid[neighbor_x, neighbor_y] == 1:
                        live_neighbors += 1
            
            # Apply Conway's rules
            if grid[i, j] == 1 and (live_neighbors == 2 or live_neighbors == 3):
                new_grid[i, j] = 1
            elif grid[i, j] == 0 and live_neighbors == 3:
                new_grid[i, j] = 1
                
    return new_grid


# @njit tells Numba to compile this to C. 
# parallel=True tells it to split the work across all CPU cores.
@njit(parallel=True)
def next_generation_numba_multicore(grid: np.ndarray) -> np.ndarray:
    rows, cols = grid.shape
    # Create an empty grid for the next generation
    new_grid = np.zeros_like(grid)
    
    # prange (parallel range) splits the rows across your CPU cores
    for i in prange(rows):
        for j in range(cols):
            
            # 1. Count neighbors manually (with toroidal wrapping)
            neighbors = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    
                    # The modulo (%) operator handles the wrap-around edges
                    r = (i + di) % rows
                    c = (j + dj) % cols
                    neighbors += grid[r, c]
            
            # 2. Apply Conway's rules
            is_alive = grid[i, j] == 1
            if is_alive and (neighbors == 2 or neighbors == 3):
                new_grid[i, j] = 1
            elif not is_alive and neighbors == 3:
                new_grid[i, j] = 1
                
    return new_grid

@njit(parallel=True)
def next_generation_numba_multicore_without_modulo(grid: np.ndarray) -> np.ndarray:
    rows, cols = grid.shape
    new_grid = np.zeros_like(grid)
    
    for i in prange(rows):
        for j in range(cols):
            
            neighbors = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    
                    # Calculate row index with fast branching
                    r = i + di
                    if r == -1:
                        r = rows - 1
                    elif r == rows:
                        r = 0
                        
                    # Calculate col index with fast branching
                    c = j + dj
                    if c == -1:
                        c = cols - 1
                    elif c == cols:
                        c = 0
                        
                    neighbors += grid[r, c]
            
            # Apply Conway's rules
            is_alive = grid[i, j] == 1
            if is_alive and (neighbors == 2 or neighbors == 3):
                new_grid[i, j] = 1
            elif not is_alive and neighbors == 3:
                new_grid[i, j] = 1
                
    return new_grid


# We tell Numba to forcefully inject this helper code wherever it's called,
# eliminating the tiny fraction of time it takes to call a function.
@njit(inline='always')
def _process_edge_cell(grid, new_grid, rows, cols, i, j):
    neighbors = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            
            r = i + di
            if r == -1: r = rows - 1
            elif r == rows: r = 0
            
            c = j + dj
            if c == -1: c = cols - 1
            elif c == cols: c = 0
            
            neighbors += grid[r, c]
            
    is_alive = grid[i, j]
    if is_alive == 1 and (neighbors == 2 or neighbors == 3):
        new_grid[i, j] = 1
    elif is_alive == 0 and neighbors == 3:
        new_grid[i, j] = 1

@njit(parallel=True)
def next_generation_numba_optimized(grid: np.ndarray) -> np.ndarray:
    rows, cols = grid.shape
    new_grid = np.zeros_like(grid)
    
    # ==========================================
    # 1. THE CORE (99.9% of the grid)
    # ==========================================
    # We strictly iterate from 1 to rows-2, avoiding the edges completely.
    for i in prange(1, rows - 1):
        for j in range(1, cols - 1):
            
            # UNROLLING THE LOOP: Instead of using `for di/dj`, we explicitly 
            # hardcode the 8 neighbor positions. The CPU loves this because 
            # there is zero loop overhead and zero conditional branching.
            neighbors = (
                grid[i-1, j-1] + grid[i-1, j] + grid[i-1, j+1] +
                grid[i,   j-1]                + grid[i,   j+1] +
                grid[i+1, j-1] + grid[i+1, j] + grid[i+1, j+1]
            )
            
            if grid[i, j] == 1:
                if neighbors == 2 or neighbors == 3:
                    new_grid[i, j] = 1
            else:
                if neighbors == 3:
                    new_grid[i, j] = 1

    # ==========================================
    # 2. THE EDGES (0.1% of the grid)
    # ==========================================
    # We fall back to the safe, bounds-checked logic only for the perimeter.
    # Since it's a tiny amount of cells, we don't need `prange` here.

    # Top and Bottom rows (this naturally covers all 4 corners too)
    for i in (0, rows - 1):
        for j in range(cols):
            _process_edge_cell(grid, new_grid, rows, cols, i, j)
            
    # Left and Right columns (skipping the top/bottom corners we just did)
    for i in range(1, rows - 1):
        for j in (0, cols - 1):
            _process_edge_cell(grid, new_grid, rows, cols, i, j)
            
    return new_grid


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

    @cuda.jit
    def _next_generation_kernel_fast(grid, output, rows, cols):
        row, col = cuda.grid(2)
        if row >= rows or col >= cols:
            return

        alive_neighbors = 0
        for d_row in range(-1, 2):
            for d_col in range(-1, 2):
                if d_row == 0 and d_col == 0:
                    continue
                    
                # Fast branching instead of modulo
                n_row = row + d_row
                if n_row == -1: n_row = rows - 1
                elif n_row == rows: n_row = 0
                    
                n_col = col + d_col
                if n_col == -1: n_col = cols - 1
                elif n_col == cols: n_col = 0
                    
                alive_neighbors += grid[n_row, n_col]

        cell_alive = grid[row, col]
        if cell_alive == 1 and (alive_neighbors == 2 or alive_neighbors == 3):
            output[row, col] = 1
        elif cell_alive == 0 and alive_neighbors == 3:
            output[row, col] = 1
        else:
            output[row, col] = 0


import logging
logger = logging.getLogger(__name__)


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


def next_generation_gpu_optimized(grid: np.ndarray, steps: int = 1) -> np.ndarray:
    """Batched GPU computation: uploads once, runs N steps on-device, downloads once."""

    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA is not available")

    # 1. Setup Phase — upload to GPU once
    initial_grid = _normalize_grid(grid)
    rows, cols = initial_grid.shape

    threads_per_block = (16, 16)
    blocks_per_grid = (
        (rows + threads_per_block[0] - 1) // threads_per_block[0],
        (cols + threads_per_block[1] - 1) // threads_per_block[1],
    )

    # Send data to GPU ONCE
    d_grid = cuda.to_device(initial_grid)
    d_output = cuda.device_array((rows, cols), dtype=np.uint8)

    # 2. Simulation loop — blistering fast, no host<->device transfers
    for _ in range(steps):
        _next_generation_kernel_fast[blocks_per_grid, threads_per_block](
            d_grid, d_output, rows, cols
        )
        
        # Swap the pointers! Next frame, the old output becomes the new input.
        # This takes 0.000001 seconds because we just swap variables.
        d_grid, d_output = d_output, d_grid

    # 3. Download only when needed (e.g., rendering the frame)
    cuda.synchronize()
    return d_grid.copy_to_host()


def compute_next_generation(grid: np.ndarray, mode: str = "numpy") -> tuple[np.ndarray, str]:
    normalized_mode = str(mode or "numpy").lower()

    if normalized_mode == "gpu":
        try:
            return next_generation_gpu(grid), "gpu"
        except Exception as e:
            logger.error("GPU mode failed, falling back to numpy: %s", e)
            return next_generation_numpy(grid), "numpy"
    elif normalized_mode == "gpu_opt":
        try:
            return next_generation_gpu_optimized(grid, steps=1), "gpu_opt"
        except Exception as e:
            logger.error("GPU Optimized mode failed, falling back to numpy: %s", e)
            return next_generation_numpy(grid), "numpy"
    elif normalized_mode == "scipy":
        return next_generation_scipy(grid), "scipy"
    elif normalized_mode == "loop":
        return next_generation_loop(grid), "loop"
    elif normalized_mode == "numba":
        return next_generation_numba_multicore(grid), "numba"
    elif normalized_mode == "numba_fast":
        return next_generation_numba_multicore_without_modulo(grid), "numba_fast"
    elif normalized_mode == "numba_opt":
        return next_generation_numba_optimized(grid), "numba_opt"

    return next_generation_numpy(grid), "numpy"


def compute_next_generation_batch(grid: np.ndarray, mode: str, steps: int) -> tuple[np.ndarray, str]:
    """Batch-aware entry point. For gpu_opt mode, runs all steps in a single
    GPU upload/download cycle. For all other modes, falls back to a simple loop."""
    normalized_mode = str(mode or "numpy").lower()

    if normalized_mode == "gpu_opt":
        try:
            return next_generation_gpu_optimized(grid, steps=steps), "gpu_opt"
        except Exception as e:
            logger.error("GPU Optimized batch failed, falling back to numpy: %s", e)
            normalized_mode = "numpy"  # fall through to loop below

    # All other modes: loop one step at a time
    mode_used = normalized_mode
    for _ in range(steps):
        grid, mode_used = compute_next_generation(grid, normalized_mode)
    return grid, mode_used
