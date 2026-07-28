import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QMainWindow, QLabel, QMessageBox

from app.worker import SimulationWorker
from app.renderer import GridRenderer
from app.controls import ControlsDock
from app.patterns import parse_rle, tile_pattern, place_pattern_centered, random_fill, PATTERN_LIBRARY

class MainWindow(QMainWindow):
    do_step_signal = pyqtSignal(np.ndarray, str)
    do_benchmark_signal = pyqtSignal(np.ndarray, str, int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Conway's Game of Life")
        self.resize(1200, 800)
        self.apply_dark_theme()

        self.running = False
        self.generation = 0
        self.step_in_progress = False

        self.renderer = GridRenderer()
        self.setCentralWidget(self.renderer)

        self.setup_worker()
        self.setup_dock()
        self.setup_status_bar()

        self.sim_timer = QTimer(self)
        self.sim_timer.timeout.connect(self.request_step)
        
        # Initial empty grid
        self.apply_grid(50, 80)

    def apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(17, 24, 39))
        palette.setColor(QPalette.WindowText, QColor(229, 231, 235))
        palette.setColor(QPalette.Base, QColor(15, 23, 42))
        palette.setColor(QPalette.AlternateBase, QColor(17, 24, 39))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.Text, QColor(229, 231, 235))
        palette.setColor(QPalette.Button, QColor(31, 41, 55))
        palette.setColor(QPalette.ButtonText, QColor(229, 231, 235))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        self.setPalette(palette)
        
        self.setStyleSheet("""
            QWidget {
                color: #ffffff;
            }
            QDockWidget {
                titlebar-close-icon: url();
                titlebar-normal-icon: url();
                color: #ffffff;
            }
            QPushButton {
                background-color: #1f2937;
                border: 1px solid #4b5563;
                border-radius: 4px;
                padding: 5px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #374151;
            }
            QComboBox, QSpinBox, QPlainTextEdit {
                background-color: #0f172a;
                border: 1px solid #4b5563;
                border-radius: 4px;
                padding: 4px;
                color: #ffffff;
            }
            QComboBox QAbstractItemView {
                background-color: #0f172a;
                color: #ffffff;
                selection-background-color: #374151;
            }
            QLabel {
                color: #ffffff;
            }
        """)

    def setup_worker(self):
        self.worker_thread = QThread()
        self.worker = SimulationWorker()
        self.worker.moveToThread(self.worker_thread)
        
        self.do_step_signal.connect(self.worker.do_step)
        self.do_benchmark_signal.connect(self.worker.do_benchmark)
        
        self.worker.step_done.connect(self.on_step_done)
        self.worker.benchmark_done.connect(self.on_benchmark_done)
        
        self.worker_thread.start()

    def closeEvent(self, event):
        self.worker_thread.quit()
        self.worker_thread.wait()
        super().closeEvent(event)

    def setup_dock(self):
        self.controls = ControlsDock(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.controls)
        
        # Connect signals
        self.controls.apply_size_requested.connect(self.apply_grid)
        self.controls.clear_grid_requested.connect(self.clear_grid)
        self.controls.speed_changed.connect(self.update_timer)
        self.controls.step_requested.connect(self.single_step)
        self.controls.toggle_run_requested.connect(self.toggle_run)
        self.controls.benchmark_requested.connect(self.run_benchmark)
        self.controls.pattern_load_requested.connect(self.load_pattern)

    def setup_status_bar(self):
        self.status = self.statusBar()
        self.label_status_mode = QLabel("Idle")
        self.label_status_gen = QLabel("Gen: 0")
        self.label_status_dim = QLabel("")
        self.label_status_vram = QLabel("")
        
        self.status.addWidget(self.label_status_mode)
        self.status.addWidget(QLabel(" | "))
        self.status.addWidget(self.label_status_gen)
        self.status.addWidget(QLabel(" | "))
        self.status.addWidget(self.label_status_dim)
        self.status.addWidget(QLabel(" | "))
        self.status.addWidget(self.label_status_vram)

    def check_vram(self, rows, cols):
        bytes_needed = rows * cols * 3
        gb = bytes_needed / (1024**3)
        self.label_status_dim.setText(f"{rows} × {cols}")
        if gb > 10.0:
            self.label_status_vram.setText(f"VRAM Est: {gb:.1f} GB ⚠️")
            self.label_status_vram.setStyleSheet("color: #ef4444; font-weight: bold;")
        else:
            self.label_status_vram.setText(f"VRAM Est: {gb:.2f} GB")
            self.label_status_vram.setStyleSheet("")

    @pyqtSlot(int, int)
    def apply_grid(self, rows, cols):
        self.generation = 0
        self.label_status_gen.setText("Gen: 0")
        self.check_vram(rows, cols)
        
        grid = np.zeros((rows, cols), dtype=np.uint8)
        self.renderer.set_grid(grid)
        self.stop_run()

    @pyqtSlot()
    def clear_grid(self):
        self.renderer.grid.fill(0)
        self.renderer.needs_texture_upload = True
        self.renderer.update()
        self.generation = 0
        self.label_status_gen.setText("Gen: 0")
        self.stop_run()

    @pyqtSlot(int)
    def update_timer(self, speed_ms):
        if self.running:
            self.sim_timer.setInterval(speed_ms)

    @pyqtSlot()
    def request_step(self):
        if self.step_in_progress:
            return # Frame drop! Prevent event loop flooding
        self.step_in_progress = True
        self.do_step_signal.emit(self.renderer.grid, self.controls.get_current_mode())

    @pyqtSlot()
    def single_step(self):
        self.stop_run()
        self.request_step()

    @pyqtSlot()
    def toggle_run(self):
        if self.running:
            self.stop_run()
        else:
            self.running = True
            self.controls.set_running_state(True)
            self.sim_timer.start(self.controls.get_speed())

    def stop_run(self):
        self.running = False
        self.controls.set_running_state(False)
        self.sim_timer.stop()

    @pyqtSlot(int)
    def run_benchmark(self, steps):
        self.stop_run()
        self.controls.set_benchmark_result("Running...")
        self.step_in_progress = True
        self.do_benchmark_signal.emit(self.renderer.grid, self.controls.get_current_mode(), steps)

    @pyqtSlot(np.ndarray, float, str)
    def on_step_done(self, next_grid, duration_ms, mode_used):
        self.renderer.set_grid(next_grid)
        self.generation += 1
        self.label_status_gen.setText(f"Gen: {self.generation}")
        self.label_status_mode.setText(f"{mode_used.upper()} step: {duration_ms:.2f} ms")
        self.step_in_progress = False

    @pyqtSlot(np.ndarray, float, float, int, str)
    def on_benchmark_done(self, next_grid, total_ms, avg_ms, steps, mode_used):
        if self.controls.check_render_bench.isChecked():
            self.renderer.set_grid(next_grid)
        self.generation += steps
        self.label_status_gen.setText(f"Gen: {self.generation}")
        res = f"Benchmark ({mode_used.upper()}, {steps} steps): total {total_ms:.1f} ms, avg {avg_ms:.3f} ms/step"
        self.controls.set_benchmark_result(res)
        self.label_status_mode.setText(f"Benchmark finished ({mode_used.upper()})")
        self.step_in_progress = False

    @pyqtSlot(str, str, float, str)
    def load_pattern(self, val, mode, density, custom_rle):
        if not val:
            return
            
        self.stop_run()
        self.generation = 0
        self.label_status_gen.setText("Gen: 0")
        
        rows = self.controls.spin_rows.value()
        cols = self.controls.spin_cols.value()
        grid = np.zeros((rows, cols), dtype=np.uint8)
        
        if val == "__random__":
            grid = random_fill(grid, density)
        else:
            rle_str = ""
            if val == "__custom__":
                rle_str = custom_rle
            else:
                pat = next((p for p in PATTERN_LIBRARY if p["name"] == val), None)
                if pat:
                    rle_str = pat["rle"]
            
            if not rle_str.strip():
                return
                
            try:
                w, h, cells = parse_rle(rle_str)
                if mode == "tile":
                    grid = tile_pattern(grid, cells, 2, 2)
                else:
                    grid = place_pattern_centered(grid, cells)
            except Exception as e:
                QMessageBox.critical(self, "Parse Error", f"Failed to parse RLE:\n{str(e)}")
                return
                
        if self.controls.check_render_bench.isChecked():
            self.renderer.set_grid(grid)
        else:
            self.renderer.grid = grid

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_run()
        elif event.key() == Qt.Key_S:
            self.single_step()
        elif event.key() == Qt.Key_C:
            self.clear_grid()
        elif event.key() == Qt.Key_R:
            # Random fill hotkey
            idx = self.controls.combo_pat.findData("__random__")
            self.controls.combo_pat.setCurrentIndex(idx)
            self.load_pattern("__random__", "center", self.controls.slider_density.value() / 100.0, "")
        else:
            super().keyPressEvent(event)
