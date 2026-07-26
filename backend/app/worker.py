import time
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from app.logic import compute_next_generation

class SimulationWorker(QObject):
    step_done = pyqtSignal(np.ndarray, float, str)
    benchmark_done = pyqtSignal(np.ndarray, float, float, int, str)

    @pyqtSlot(np.ndarray, str)
    def do_step(self, grid, mode):
        t0 = time.perf_counter()
        next_grid, mode_used = compute_next_generation(grid, mode)
        t1 = time.perf_counter()
        self.step_done.emit(next_grid, (t1 - t0) * 1000, mode_used)

    @pyqtSlot(np.ndarray, str, int)
    def do_benchmark(self, grid, mode, steps):
        t0 = time.perf_counter()
        mode_used = mode
        for _ in range(steps):
            grid, mode_used = compute_next_generation(grid, mode)
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000
        avg_ms = total_ms / steps
        self.benchmark_done.emit(grid, total_ms, avg_ms, steps, mode_used)
