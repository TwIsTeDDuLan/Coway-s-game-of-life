import numpy as np
import numba
from numba import cuda
import traceback

try:
    CUDA_AVAILABLE = bool(cuda.is_available())
    
    @cuda.jit(debug=True, opt=False)
    def _next_generation_kernel_debug(grid, output, rows, cols):
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
            
    grid = np.zeros((3, 3), dtype=np.uint8)
    grid[1, 0:3] = 1
    rows, cols = grid.shape
    device_grid = cuda.to_device(grid)
    device_output = cuda.device_array((rows, cols), dtype=np.uint8)

    threads_per_block = (16, 16)
    blocks_per_grid = (
        (rows + threads_per_block[0] - 1) // threads_per_block[0],
        (cols + threads_per_block[1] - 1) // threads_per_block[1],
    )

    _next_generation_kernel_debug[blocks_per_grid, threads_per_block](
        device_grid, device_output, rows, cols
    )
    cuda.synchronize()
    print("GPU Success")
except Exception as e:
    traceback.print_exc()
