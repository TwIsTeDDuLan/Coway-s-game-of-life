"""Root conftest – sets up environment before any test imports."""

import os
import sys

# Ensure Numba uses the NVIDIA cuda-python binding (avoids the
# cuCtxSynchronize_v2 context-destroyed error on CUDA 13.0 drivers).
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

# Allow ``from app.logic import …`` without installing as a package.
sys.path.insert(0, os.path.dirname(__file__))
