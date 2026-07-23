# Migration Plan: Web → PyQt5 Desktop Application

## Goal

Replace the React + FastAPI/WebSocket architecture with a self-contained PyQt5 desktop application.
The simulation logic (`logic.py`) is called directly — no serialization, no network, no frame-size limits.

---

## Decisions (Confirmed)

| Decision | Choice |
|---|---|
| Renderer | **QOpenGLWidget** — texture-based grid rendering |
| Simulation thread | **QThread** — keeps UI responsive during heavy GPU steps |
| Theme | **Dark** — hardcoded dark palette, matching the original web UI |
| Controls layout | **QDockWidget** sidebar — resizable, detachable |
| Web frontend | **Remove completely** — delete `frontend/` dir and FastAPI server |
| Grid size limit | **No hard cap** — limited only by hardware (see GPU analysis below) |

---

## RTX 3060 12 GB — Grid Size Analysis

Memory consumers per cell at peak (during a simulation step):

| Resource | Location | Bytes/cell | Lifetime |
|---|---|---|---|
| `numpy` grid | CPU RAM | 1 | Persistent |
| OpenGL texture (`GL_R8`) | GPU VRAM | 1 | Persistent |
| CUDA input array | GPU VRAM | 1 | Per-step (transient) |
| CUDA output array | GPU VRAM | 1 | Per-step (transient) |
| **Peak GPU total** | | **3** | |

With ~10 GB usable VRAM (after driver, CUDA context, desktop compositor):

```
10 GB ÷ 3 bytes/cell ≈ 3.3 billion cells
√3.3B ≈ 57,000
```

**Safe maximum: ~50,000 × 50,000** (7.5 GB VRAM peak).
Pushing to **55,000 × 55,000** is possible but tight.

> [!NOTE]
> CPU-mode (NumPy) is only limited by system RAM — 1 byte/cell means 32 GB RAM can hold
> a ~180,000 × 180,000 grid, though NumPy `np.roll` becomes very slow at that scale.

The UI will display a warning when estimated VRAM exceeds 10 GB but will **not** block creation.

---

## Architecture

```
backend/
├── app/
│   ├── __init__.py
│   ├── logic.py            ← KEEP (CPU/GPU simulation, add warning suppression)
│   ├── patterns.py          ← NEW  (RLE parser + pattern library, ported from JS)
│   └── main_window.py       ← NEW  (PyQt5 application — all UI lives here)
├── tests/
│   ├── test_logic.py        ← KEEP
│   ├── test_patterns.py     ← NEW  (RLE parser tests)
│   └── conftest.py          ← KEEP
├── requirements.txt         ← MODIFY (add PyQt5, PyOpenGL; remove fastapi, uvicorn, httpx)
└── run_gui.py               ← NEW  (entry point: `python run_gui.py`)
```

### Deleted

```
frontend/                    ← DELETE (entire directory)
backend/app/main.py          ← DELETE (FastAPI WebSocket server)
backend/tests/test_websocket.py ← DELETE
```

---

## Phase 1 — Foundations

### 1.1 — [DELETE] Web Frontend & FastAPI Server

Remove completely:
- `frontend/` directory (React, Vite, node_modules, etc.)
- `backend/app/main.py` (FastAPI WebSocket server)
- `backend/tests/test_websocket.py`

Update `.gitignore` — remove frontend-specific entries if any.

### 1.2 — [MODIFY] `backend/requirements.txt`

```diff
-fastapi==0.116.1
 numpy==2.2.6
 numba==0.61.2
 cuda-python>=12.6,<13
-uvicorn[standard]==0.35.0
 pytest==8.4.1
-httpx==0.28.1
+PyQt5>=5.15
+PyOpenGL>=3.1
```

### 1.3 — [MODIFY] `backend/app/logic.py`

One-line change — suppress the harmless Numba GPU under-utilization warning:

```python
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
```

---

## Phase 2 — Pattern Library (Python Port)

### 2.1 — [NEW] `backend/app/patterns.py`

Port from `frontend/src/patterns.js`. Key differences from JS version:

| Function | Signature | Notes |
|---|---|---|
| `parse_rle` | `(rle: str) → tuple[int, int, np.ndarray]` | Returns `(width, height, cells)` where `cells` is `np.ndarray` (uint8) |
| `tile_pattern` | `(grid, pattern, spacing_x=2, spacing_y=2) → grid` | Mutates & returns grid |
| `place_pattern_centered` | `(grid, pattern) → grid` | Single centered copy |
| `random_fill` | `(grid, density=0.3) → grid` | Uses `np.random` (faster than element-wise) |

`PATTERN_LIBRARY` — same 22 patterns as the JS version, stored as list of dicts:
```python
{"name": "Gosper Glider Gun", "category": "Gun", "description": "...", "rle": "..."}
```

`PATTERN_CATEGORIES` — derived from library, same order as JS.

### 2.2 — [NEW] `backend/tests/test_patterns.py`

- Test `parse_rle` against Glider, Gosper Gun, Block (known dimensions + cell positions)
- Test `tile_pattern` verifies multiple copies are placed
- Test `place_pattern_centered` verifies centering math
- Test `random_fill` density is approximately correct

---

## Phase 3 — PyQt5 Application

### 3.1 — [NEW] `backend/app/main_window.py`

This is the core file. It contains four classes:

---

#### Class: `GridRenderer(QOpenGLWidget)`

Renders the grid as an OpenGL texture.

**Rendering pipeline:**
1. On grid update: upload `numpy` array as a `GL_R8` texture via `glTexImage2D`
2. Draw a fullscreen quad textured with the grid
3. Fragment shader: `cell == 1 → alive_color (#16a34a)`, `cell == 0 → dead_color (#111827)`
4. Grid lines drawn as a second pass (only when zoom ≥ 3 px/cell, same as web version)

**Interactions:**
- `wheelEvent` → zoom (pivot around cursor), clamped to 0.1× – 50× (no max_zoom cap from web)
- `mousePressEvent(MiddleButton | RightButton)` → start pan
- `mouseMoveEvent` while panning → update viewport offset
- `mousePressEvent(LeftButton)` → toggle cell, record draw value
- `mouseMoveEvent` while drawing → paint cells with same value

**Key methods:**
- `set_grid(grid: np.ndarray)` → store reference + flag texture for re-upload
- `initializeGL()` → compile shaders, create VAO/VBO for quad, create texture
- `paintGL()` → bind texture, draw quad with current zoom/offset uniforms
- `screen_to_grid(x, y) → (row, col)` → coordinate conversion

---

#### Class: `SimulationWorker(QObject)` — runs on QThread

Decouples the simulation from the UI thread.

**Signals:**
- `step_done(grid: np.ndarray, duration_ms: float, mode_used: str)`
- `benchmark_done(grid: np.ndarray, total_ms: float, avg_ms: float, steps: int, mode_used: str)`

**Slots:**
- `do_step(grid, mode)` → calls `compute_next_generation`, emits `step_done`
- `do_benchmark(grid, mode, steps)` → runs N steps, emits `benchmark_done`

**Thread management:**
- Worker lives on a `QThread` created by `MainWindow`
- `QTimer` on main thread fires at configurable interval → invokes `do_step` via signal
- When a step completes, `step_done` delivers the new grid back to the main thread
- The timer does **not** fire a new step until the previous one completes (avoids queueing)

---

#### Class: `ControlsDock(QDockWidget)`

Docked sidebar with all controls. Sections:

**Grid Settings:**
- Rows / Cols spin boxes (no upper limit, VRAM warning label)
- "Apply Grid" button — creates new empty grid
- "Clear" button — zeros the current grid

**Simulation:**
- Logic mode combo: CPU (NumPy) / GPU (CUDA)
- Speed slider: 10ms – 2000ms interval
- Step / Run / Pause buttons
- Generation counter label

**Benchmark:**
- Steps spin box
- "Run Benchmark" button
- Results label (total ms, avg ms/step, mode used)

**Pattern Library:**
- Pattern combo box (`QComboBox`) with `optgroup`-style separators per category
- Includes "🎲 Random Fill" and "📝 Custom RLE…" special entries
- Placement mode combo: Tile / Center
- Density slider (visible only for Random Fill)
- Custom RLE text edit (`QPlainTextEdit`, visible only for Custom RLE)
- "Load Pattern" button
- Description label

**Zoom:**
- Zoom slider (0.1× – 50×)
- Reset zoom button

---

#### Class: `MainWindow(QMainWindow)`

Ties everything together.

- Central widget: `GridRenderer`
- Left dock: `ControlsDock`
- Status bar: mode, last step time, generation count, grid dimensions, estimated VRAM
- Dark palette applied in `__init__` via `QPalette` (matching `#111827` / `#e5e7eb` scheme)
- Window title: "Conway's Game of Life"
- Default size: 1200 × 800
- Keyboard shortcuts:
  - `Space` → toggle Run/Pause
  - `S` → single Step
  - `C` → Clear
  - `R` → Random fill (current density)
  - `Ctrl+Q` → Quit

**Wiring flow:**
```
ControlsDock.run_clicked  → MainWindow starts QTimer
QTimer.timeout            → MainWindow calls SimulationWorker.do_step (via signal)
SimulationWorker.step_done → MainWindow.on_step_done:
                              - updates self.grid
                              - calls GridRenderer.set_grid()
                              - updates status bar
                              - increments generation counter
```

### 3.2 — [NEW] `backend/run_gui.py`

```python
#!/usr/bin/env python3
"""Launch Conway's Game of Life — PyQt5 desktop edition."""
import sys
from PyQt5.QtWidgets import QApplication
from app.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
```

---

## Phase 4 — Cleanup & Documentation

### 4.1 — [MODIFY] `README.md`

Rewrite to reflect the new desktop-only architecture:
- Installation: `pip install -r requirements.txt`
- Run: `cd backend && python run_gui.py`
- Usage guide (controls, keyboard shortcuts, pattern library)
- GPU requirements section

### 4.2 — [MODIFY] `.AI/FUTURE_ENHANCEMENTS.md`

- Remove items 1 (Binary WebSocket) and 4 (Server-Side Viewport Slicing) — no longer relevant
- Keep items 3 (WebGL → already using OpenGL), 5 (Chunk-Based Storage), 6 (Touch Gestures)
- Update item 2 (Web Worker → QThread — already implemented)

---

## File Summary

| Action | File | Description |
|---|---|---|
| DELETE | `frontend/` (entire dir) | React + Vite frontend |
| DELETE | `backend/app/main.py` | FastAPI WebSocket server |
| DELETE | `backend/tests/test_websocket.py` | WebSocket tests |
| KEEP | `backend/app/logic.py` | CPU/GPU simulation (minor warning fix) |
| KEEP | `backend/app/__init__.py` | Package init |
| KEEP | `backend/tests/test_logic.py` | Logic unit tests |
| KEEP | `backend/tests/conftest.py` | Test fixtures |
| NEW | `backend/app/patterns.py` | RLE parser + 22-pattern library |
| NEW | `backend/app/main_window.py` | PyQt5 application (4 classes) |
| NEW | `backend/run_gui.py` | Entry point |
| NEW | `backend/tests/test_patterns.py` | Pattern parser tests |
| MODIFY | `backend/requirements.txt` | Add PyQt5/PyOpenGL, remove web deps |
| MODIFY | `README.md` | Desktop-focused docs |
| MODIFY | `.AI/FUTURE_ENHANCEMENTS.md` | Remove web-specific items |

---

## Execution Order

1. Phase 1 — Delete web stuff, update deps, fix warning
2. Phase 2 — patterns.py + tests (can verify independently)
3. Phase 3 — main_window.py + run_gui.py (the big one)
4. Phase 4 — Docs cleanup

---

## Verification Plan

### Automated
```bash
cd backend
pytest tests/test_logic.py tests/test_patterns.py -v
```

### Manual
- `python run_gui.py` → window opens with dark theme, docked sidebar
- Set grid to 200×200 → load Gosper Glider Gun (tile mode) → Run → gliders fill the screen
- Set grid to 10000×10000 → load Random Fill (30%) → Run on GPU → verify smooth stepping
- Test zoom (scroll wheel), pan (middle-click), draw (left-click)
- Paste custom RLE from conwaylife.com → verify it loads
- Run benchmark (200 steps, GPU) → verify results display
- Test keyboard shortcuts (Space, S, C, R)
- Set grid to 50000×50000 → verify VRAM warning appears but grid creates successfully
