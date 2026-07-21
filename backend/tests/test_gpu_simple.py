import numpy as np
from numba import cuda

@cuda.jit
def simple_kernel(output, rows, cols):
    row, col = cuda.grid(2)
    if row < rows and col < cols:
        output[row, col] = 1

rows, cols = 3, 3
device_output = cuda.device_array((rows, cols), dtype=np.uint8)
simple_kernel[(1, 1), (16, 16)](device_output, rows, cols)
cuda.synchronize()
print("Success")
