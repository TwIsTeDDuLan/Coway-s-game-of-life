from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QSlider, QComboBox, QPlainTextEdit, QDockWidget, QCheckBox
)

from app.patterns import PATTERN_LIBRARY, PATTERN_CATEGORIES

class ControlsDock(QDockWidget):
    # Signals to communicate with MainWindow
    apply_size_requested = pyqtSignal(int, int) # rows, cols
    clear_grid_requested = pyqtSignal()
    speed_changed = pyqtSignal(int)
    step_requested = pyqtSignal()
    toggle_run_requested = pyqtSignal()
    benchmark_requested = pyqtSignal(int) # steps
    pattern_load_requested = pyqtSignal(str, str, float, str) # val, mode (tile/center), density, custom_rle

    def __init__(self, parent=None):
        super().__init__("Controls", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.setup_ui()

    def setup_ui(self):
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
        btn_apply.clicked.connect(lambda: self.apply_size_requested.emit(self.spin_rows.value(), self.spin_cols.value()))
        btn_clear = QPushButton("Clear Grid")
        btn_clear.clicked.connect(self.clear_grid_requested.emit)
        btn_lay.addWidget(btn_apply); btn_lay.addWidget(btn_clear)
        grid_layout.addLayout(btn_lay)
        layout.addLayout(grid_layout)

        # 2. Simulation
        sim_layout = QVBoxLayout()
        sim_layout.addWidget(QLabel("<b>Simulation</b>"))
        
        mode_lay = QHBoxLayout()
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("CPU (NumPy)", "numpy")
        self.combo_mode.addItem("CPU (SciPy Convolution)", "scipy")
        self.combo_mode.addItem("CPU (Native Loop)", "loop")
        self.combo_mode.addItem("CPU (Numba Multicore)", "numba")
        self.combo_mode.addItem("CPU (Numba Fast Branching)", "numba_fast")
        self.combo_mode.addItem("CPU (Numba Optimized)", "numba_opt")
        self.combo_mode.addItem("GPU (CUDA)", "gpu")
        self.combo_mode.addItem("GPU (CUDA Optimized)", "gpu_opt")
        mode_lay.addWidget(QLabel("Mode:")); mode_lay.addWidget(self.combo_mode)
        sim_layout.addLayout(mode_lay)
        
        speed_lay = QHBoxLayout()
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(10, 1000)
        self.slider_speed.setValue(100)
        self.slider_speed.setInvertedAppearance(True)
        self.label_speed = QLabel("100 ms")
        self.slider_speed.valueChanged.connect(self._on_speed_changed)
        speed_lay.addWidget(QLabel("Speed:")); speed_lay.addWidget(self.slider_speed); speed_lay.addWidget(self.label_speed)
        sim_layout.addLayout(speed_lay)

        ctrl_lay = QHBoxLayout()
        self.btn_step = QPushButton("Step")
        self.btn_step.clicked.connect(self.step_requested.emit)
        self.btn_run = QPushButton("Run")
        self.btn_run.clicked.connect(self.toggle_run_requested.emit)
        ctrl_lay.addWidget(self.btn_step); ctrl_lay.addWidget(self.btn_run)
        sim_layout.addLayout(ctrl_lay)
        layout.addLayout(sim_layout)

        # 3. Benchmark
        bench_layout = QVBoxLayout()
        bench_layout.addWidget(QLabel("<b>Benchmark</b>"))
        
        b_lay = QHBoxLayout()
        self.spin_bench = QSpinBox(); self.spin_bench.setRange(1, 10000); self.spin_bench.setValue(200)
        btn_bench = QPushButton("Run Benchmark")
        btn_bench.clicked.connect(lambda: self.benchmark_requested.emit(self.spin_bench.value()))
        b_lay.addWidget(QLabel("Steps:")); b_lay.addWidget(self.spin_bench); b_lay.addWidget(btn_bench)
        bench_layout.addLayout(b_lay)
        
        self.check_render_bench = QCheckBox("Render result (Benchmark / Load)")
        self.check_render_bench.setChecked(True)
        bench_layout.addWidget(self.check_render_bench)
        
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
        btn_load.clicked.connect(self._on_load_pattern)
        btn_load.setStyleSheet("background-color: #16a34a; font-weight: bold;")
        pat_layout.addWidget(btn_load)
        layout.addLayout(pat_layout)

        layout.addStretch()
        widget.setLayout(layout)
        self.setWidget(widget)

    def _on_speed_changed(self, v):
        self.label_speed.setText(f"{v} ms")
        self.speed_changed.emit(v)

    def on_pattern_select(self):
        val = self.combo_pat.currentData()
        self.widget_density.setVisible(val == "__random__")
        self.text_rle.setVisible(val == "__custom__")
        
        self.label_pat_desc.setText("")
        if val and val not in ("__random__", "__custom__"):
            pat = next((p for p in PATTERN_LIBRARY if p["name"] == val), None)
            if pat:
                self.label_pat_desc.setText(pat["description"])

    def _on_load_pattern(self):
        val = self.combo_pat.currentData()
        mode = "tile" if self.combo_pat_mode.currentIndex() == 0 else "center"
        density = self.slider_density.value() / 100.0
        custom_rle = self.text_rle.toPlainText()
        self.pattern_load_requested.emit(val or "", mode, density, custom_rle)

    def get_current_mode(self):
        return self.combo_mode.currentData()
        
    def get_speed(self):
        return self.slider_speed.value()

    def set_running_state(self, is_running):
        if is_running:
            self.btn_run.setText("Pause")
            self.btn_step.setEnabled(False)
        else:
            self.btn_run.setText("Run")
            self.btn_step.setEnabled(True)

    def set_benchmark_result(self, res):
        self.label_bench_res.setText(res)
