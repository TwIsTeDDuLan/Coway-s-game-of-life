import time
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot, QObject
from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSpinBox, QSlider, QComboBox, QPlainTextEdit, QDockWidget,
    QMessageBox, QOpenGLWidget
)
from OpenGL import GL

from app.logic import compute_next_generation
from app.patterns import (
    parse_rle, tile_pattern, place_pattern_centered, random_fill,
    PATTERN_LIBRARY, PATTERN_CATEGORIES
)

VERTEX_SHADER = """
#version 120
varying vec2 vTexCoord;
void main() {
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    vTexCoord = gl_MultiTexCoord0.st;
}
"""

FRAGMENT_SHADER = """
#version 120
uniform sampler2D tex;
varying vec2 vTexCoord;
void main() {
    float val = texture2D(tex, vTexCoord).r;
    if (val > 0.0) {
        gl_FragColor = vec4(0.086, 0.639, 0.18, 1.0); // #16a34a (green)
    } else {
        gl_FragColor = vec4(0.067, 0.094, 0.153, 1.0); // #111827 (dark background)
    }
}
"""


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


class GridRenderer(QOpenGLWidget):
    cell_toggled = pyqtSignal(int, int, int)  # row, col, new_val

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = np.zeros((50, 80), dtype=np.uint8)
        self.zoom = 1.0
        self.base_cell_size = 10.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.needs_texture_upload = True

        self.panning = False
        self.last_mouse = None
        self.drawing = False
        self.draw_val = 1
        self.shader = None
        self.texture_id = None
        self.setMouseTracking(True)

    def set_grid(self, grid):
        self.grid = grid
        self.needs_texture_upload = True
        self.update()

    def initializeGL(self):
        from OpenGL.GL import shaders
        GL.glClearColor(0.067, 0.094, 0.153, 1.0)
        self.texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        vs = shaders.compileShader(VERTEX_SHADER, GL.GL_VERTEX_SHADER)
        fs = shaders.compileShader(FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER)
        self.shader = shaders.compileProgram(vs, fs)

    def clamp_offset(self):
        w, h = self.width(), self.height()
        rows, cols = self.grid.shape
        cell_px = self.base_cell_size * self.zoom
        world_w = cols * cell_px
        world_h = rows * cell_px
        self.offset_x = max(0.0, min(self.offset_x, max(0.0, world_w - w)))
        self.offset_y = max(0.0, min(self.offset_y, max(0.0, world_h - h)))

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        rows, cols = self.grid.shape

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        if self.needs_texture_upload:
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_LUMINANCE, cols, rows, 0,
                            GL.GL_LUMINANCE, GL.GL_UNSIGNED_BYTE, self.grid)
            self.needs_texture_upload = False

        from OpenGL.GL import shaders
        shaders.glUseProgram(self.shader)

        w, h = self.width(), self.height()
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0, w, h, 0, -1, 1)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        cell_px = self.base_cell_size * self.zoom
        world_w = cols * cell_px
        world_h = rows * cell_px

        self.clamp_offset()
        GL.glTranslatef(-self.offset_x, -self.offset_y, 0)

        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0, 0); GL.glVertex2f(0, 0)
        GL.glTexCoord2f(1, 0); GL.glVertex2f(world_w, 0)
        GL.glTexCoord2f(1, 1); GL.glVertex2f(world_w, world_h)
        GL.glTexCoord2f(0, 1); GL.glVertex2f(0, world_h)
        GL.glEnd()

        shaders.glUseProgram(0)

        if cell_px >= 3:
            GL.glColor4f(0.21, 0.25, 0.32, 1.0)
            GL.glBegin(GL.GL_LINES)
            start_c = max(0, int(self.offset_x / cell_px))
            end_c = min(cols, int((self.offset_x + w) / cell_px) + 1)
            start_r = max(0, int(self.offset_y / cell_px))
            end_r = min(rows, int((self.offset_y + h) / cell_px) + 1)

            for c in range(start_c, end_c + 1):
                x = c * cell_px
                GL.glVertex2f(x, start_r * cell_px)
                GL.glVertex2f(x, end_r * cell_px)

            for r in range(start_r, end_r + 1):
                y = r * cell_px
                GL.glVertex2f(start_c * cell_px, y)
                GL.glVertex2f(end_c * cell_px, y)
            GL.glEnd()

    def screen_to_grid(self, x, y):
        cell_px = self.base_cell_size * self.zoom
        col = int((self.offset_x + x) / cell_px)
        row = int((self.offset_y + y) / cell_px)
        rows, cols = self.grid.shape
        if 0 <= row < rows and 0 <= col < cols:
            return row, col
        return None

    def mousePressEvent(self, event):
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self.panning = True
            self.last_mouse = event.pos()
        elif event.button() == Qt.LeftButton:
            self.drawing = True
            cell = self.screen_to_grid(event.x(), event.y())
            if cell:
                r, c = cell
                self.draw_val = 0 if self.grid[r, c] else 1
                self.grid[r, c] = self.draw_val
                self.needs_texture_upload = True
                self.update()
                self.cell_toggled.emit(r, c, self.draw_val)

    def mouseMoveEvent(self, event):
        if self.panning and self.last_mouse:
            dx = event.x() - self.last_mouse.x()
            dy = event.y() - self.last_mouse.y()
            self.offset_x -= dx
            self.offset_y -= dy
            self.last_mouse = event.pos()
            self.clamp_offset()
            self.update()
        elif self.drawing:
            cell = self.screen_to_grid(event.x(), event.y())
            if cell:
                r, c = cell
                if self.grid[r, c] != self.draw_val:
                    self.grid[r, c] = self.draw_val
                    self.needs_texture_upload = True
                    self.update()
                    self.cell_toggled.emit(r, c, self.draw_val)

    def mouseReleaseEvent(self, event):
        self.panning = False
        self.drawing = False
        self.last_mouse = None

    def wheelEvent(self, event):
        old_cell_px = self.base_cell_size * self.zoom
        delta = 0.1 if event.angleDelta().y() > 0 else -0.1
        self.zoom = max(0.1, min(50.0, self.zoom + delta))
        new_cell_px = self.base_cell_size * self.zoom

        mx, my = event.x(), event.y()
        cell_c = (self.offset_x + mx) / old_cell_px
        cell_r = (self.offset_y + my) / old_cell_px
        
        self.offset_x = cell_c * new_cell_px - mx
        self.offset_y = cell_r * new_cell_px - my
        
        self.clamp_offset()
        self.update()


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

        self.renderer = GridRenderer()
        self.setCentralWidget(self.renderer)

        self.setup_worker()
        self.setup_dock()
        self.setup_status_bar()

        self.sim_timer = QTimer(self)
        self.sim_timer.timeout.connect(self.request_step)
        
        # Initial empty grid
        self.apply_grid()

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
        self.dock = QDockWidget("Controls", self)
        self.dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # 1. Grid Config
        grid_layout = QVBoxLayout()
        grid_layout.addWidget(QLabel("<b>Grid Config</b>"))
        
        row_col_lay = QHBoxLayout()
        self.spin_rows = QSpinBox(); self.spin_rows.setRange(1, 100000); self.spin_rows.setValue(50)
        self.spin_cols = QSpinBox(); self.spin_cols.setRange(1, 100000); self.spin_cols.setValue(80)
        row_col_lay.addWidget(QLabel("Rows:")); row_col_lay.addWidget(self.spin_rows)
        row_col_lay.addWidget(QLabel("Cols:")); row_col_lay.addWidget(self.spin_cols)
        grid_layout.addLayout(row_col_lay)
        
        btn_lay = QHBoxLayout()
        btn_apply = QPushButton("Apply Size")
        btn_apply.clicked.connect(self.apply_grid)
        btn_clear = QPushButton("Clear Grid")
        btn_clear.clicked.connect(self.clear_grid)
        btn_lay.addWidget(btn_apply); btn_lay.addWidget(btn_clear)
        grid_layout.addLayout(btn_lay)
        layout.addLayout(grid_layout)

        # 2. Simulation
        sim_layout = QVBoxLayout()
        sim_layout.addWidget(QLabel("<b>Simulation</b>"))
        
        mode_lay = QHBoxLayout()
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["CPU (NumPy)", "GPU (CUDA)"])
        mode_lay.addWidget(QLabel("Mode:")); mode_lay.addWidget(self.combo_mode)
        sim_layout.addLayout(mode_lay)
        
        speed_lay = QHBoxLayout()
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(10, 1000)
        self.slider_speed.setValue(100)
        self.slider_speed.setInvertedAppearance(True) # Left is slower (high ms), right is faster (low ms)
        self.label_speed = QLabel("100 ms")
        self.slider_speed.valueChanged.connect(lambda v: self.label_speed.setText(f"{v} ms"))
        self.slider_speed.valueChanged.connect(self.update_timer)
        speed_lay.addWidget(QLabel("Speed:")); speed_lay.addWidget(self.slider_speed); speed_lay.addWidget(self.label_speed)
        sim_layout.addLayout(speed_lay)

        ctrl_lay = QHBoxLayout()
        self.btn_step = QPushButton("Step")
        self.btn_step.clicked.connect(self.single_step)
        self.btn_run = QPushButton("Run")
        self.btn_run.clicked.connect(self.toggle_run)
        ctrl_lay.addWidget(self.btn_step); ctrl_lay.addWidget(self.btn_run)
        sim_layout.addLayout(ctrl_lay)
        layout.addLayout(sim_layout)

        # 3. Benchmark
        bench_layout = QVBoxLayout()
        bench_layout.addWidget(QLabel("<b>Benchmark</b>"))
        
        b_lay = QHBoxLayout()
        self.spin_bench = QSpinBox(); self.spin_bench.setRange(1, 10000); self.spin_bench.setValue(200)
        btn_bench = QPushButton("Run Benchmark")
        btn_bench.clicked.connect(self.run_benchmark)
        b_lay.addWidget(QLabel("Steps:")); b_lay.addWidget(self.spin_bench); b_lay.addWidget(btn_bench)
        bench_layout.addLayout(b_lay)
        self.label_bench_res = QLabel("")
        self.label_bench_res.setWordWrap(True)
        bench_layout.addWidget(self.label_bench_res)
        layout.addLayout(bench_layout)

        # 4. Pattern Library
        pat_layout = QVBoxLayout()
        pat_layout.addWidget(QLabel("<b>Pattern Library</b>"))
        
        self.combo_pat = QComboBox()
        self.combo_pat.addItem("— Select a pattern —", "")
        self.combo_pat.addItem("🎲 Random Fill", "__random__")
        self.combo_pat.addItem("📝 Custom RLE…", "__custom__")
        for cat in PATTERN_CATEGORIES:
            self.combo_pat.insertSeparator(self.combo_pat.count())
            for pat in PATTERN_LIBRARY:
                if pat["category"] == cat:
                    self.combo_pat.addItem(f"{cat}: {pat['name']}", pat["name"])
        pat_layout.addWidget(self.combo_pat)
        
        self.combo_pat.currentIndexChanged.connect(self.on_pattern_select)
        
        self.combo_pat_mode = QComboBox()
        self.combo_pat_mode.addItems(["Tile (fill grid)", "Center (single copy)"])
        pat_layout.addWidget(self.combo_pat_mode)
        
        self.slider_density = QSlider(Qt.Horizontal)
        self.slider_density.setRange(5, 80)
        self.slider_density.setValue(30)
        self.label_density = QLabel("30%")
        self.slider_density.valueChanged.connect(lambda v: self.label_density.setText(f"{v}%"))
        dens_lay = QHBoxLayout()
        dens_lay.addWidget(QLabel("Density:")); dens_lay.addWidget(self.slider_density); dens_lay.addWidget(self.label_density)
        self.widget_density = QWidget(); self.widget_density.setLayout(dens_lay)
        pat_layout.addWidget(self.widget_density)
        self.widget_density.hide()
        
        self.text_rle = QPlainTextEdit()
        self.text_rle.setPlaceholderText("#C My pattern\nx = 3, y = 3, rule = B3/S23\nbob$2bo$3o!")
        self.text_rle.setMaximumHeight(100)
        pat_layout.addWidget(self.text_rle)
        self.text_rle.hide()
        
        self.label_pat_desc = QLabel("")
        self.label_pat_desc.setWordWrap(True)
        self.label_pat_desc.setStyleSheet("color: #94a3b8; font-style: italic;")
        pat_layout.addWidget(self.label_pat_desc)
        
        btn_load = QPushButton("Load Pattern")
        btn_load.clicked.connect(self.load_pattern)
        btn_load.setStyleSheet("background-color: #16a34a; font-weight: bold;")
        pat_layout.addWidget(btn_load)
        layout.addLayout(pat_layout)

        layout.addStretch()
        widget.setLayout(layout)
        self.dock.setWidget(widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock)

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

    def apply_grid(self):
        rows = self.spin_rows.value()
        cols = self.spin_cols.value()
        self.generation = 0
        self.label_status_gen.setText("Gen: 0")
        self.check_vram(rows, cols)
        
        grid = np.zeros((rows, cols), dtype=np.uint8)
        self.renderer.set_grid(grid)
        self.stop_run()

    def clear_grid(self):
        self.renderer.grid.fill(0)
        self.renderer.needs_texture_upload = True
        self.renderer.update()
        self.generation = 0
        self.label_status_gen.setText("Gen: 0")
        self.stop_run()

    def update_timer(self):
        if self.running:
            self.sim_timer.setInterval(self.slider_speed.value())

    def get_current_mode(self):
        return "gpu" if self.combo_mode.currentIndex() == 1 else "cpu"

    def request_step(self):
        self.do_step_signal.emit(self.renderer.grid, self.get_current_mode())

    def single_step(self):
        self.stop_run()
        self.request_step()

    def toggle_run(self):
        if self.running:
            self.stop_run()
        else:
            self.running = True
            self.btn_run.setText("Pause")
            self.btn_step.setEnabled(False)
            self.sim_timer.start(self.slider_speed.value())

    def stop_run(self):
        self.running = False
        self.btn_run.setText("Run")
        self.btn_step.setEnabled(True)
        self.sim_timer.stop()

    def run_benchmark(self):
        self.stop_run()
        self.label_bench_res.setText("Running...")
        self.do_benchmark_signal.emit(self.renderer.grid, self.get_current_mode(), self.spin_bench.value())

    @pyqtSlot(np.ndarray, float, str)
    def on_step_done(self, next_grid, duration_ms, mode_used):
        self.renderer.set_grid(next_grid)
        self.generation += 1
        self.label_status_gen.setText(f"Gen: {self.generation}")
        self.label_status_mode.setText(f"{mode_used.upper()} step: {duration_ms:.2f} ms")

    @pyqtSlot(np.ndarray, float, float, int, str)
    def on_benchmark_done(self, next_grid, total_ms, avg_ms, steps, mode_used):
        self.renderer.set_grid(next_grid)
        self.generation += steps
        self.label_status_gen.setText(f"Gen: {self.generation}")
        res = f"Benchmark ({mode_used.upper()}, {steps} steps): total {total_ms:.1f} ms, avg {avg_ms:.3f} ms/step"
        self.label_bench_res.setText(res)
        self.label_status_mode.setText(f"Benchmark finished ({mode_used.upper()})")

    def on_pattern_select(self):
        val = self.combo_pat.currentData()
        self.widget_density.setVisible(val == "__random__")
        self.text_rle.setVisible(val == "__custom__")
        
        self.label_pat_desc.setText("")
        if val and val not in ("__random__", "__custom__"):
            pat = next((p for p in PATTERN_LIBRARY if p["name"] == val), None)
            if pat:
                self.label_pat_desc.setText(pat["description"])

    def load_pattern(self):
        val = self.combo_pat.currentData()
        if not val:
            return
            
        self.stop_run()
        self.generation = 0
        self.label_status_gen.setText("Gen: 0")
        
        rows = self.spin_rows.value()
        cols = self.spin_cols.value()
        grid = np.zeros((rows, cols), dtype=np.uint8)
        
        if val == "__random__":
            density = self.slider_density.value() / 100.0
            grid = random_fill(grid, density)
        else:
            mode = "tile" if self.combo_pat_mode.currentIndex() == 0 else "center"
            rle_str = ""
            if val == "__custom__":
                rle_str = self.text_rle.toPlainText()
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
                
        self.renderer.set_grid(grid)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_run()
        elif event.key() == Qt.Key_S:
            self.single_step()
        elif event.key() == Qt.Key_C:
            self.clear_grid()
        elif event.key() == Qt.Key_R:
            idx = self.combo_pat.findData("__random__")
            self.combo_pat.setCurrentIndex(idx)
            self.load_pattern()
        else:
            super().keyPressEvent(event)
