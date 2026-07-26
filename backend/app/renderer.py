import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL import GL

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
        self.last_bounds = None

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
        w, h = self.width(), self.height()
        cell_px = self.base_cell_size * self.zoom

        self.clamp_offset()
        
        start_c = max(0, int(self.offset_x / cell_px))
        end_c = min(cols, int((self.offset_x + w) / cell_px) + 1)
        start_r = max(0, int(self.offset_y / cell_px))
        end_r = min(rows, int((self.offset_y + h) / cell_px) + 1)
        
        if start_c >= end_c or start_r >= end_r:
            return

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        if self.needs_texture_upload or self.last_bounds != (start_r, end_r, start_c, end_c):
            visible_grid = self.grid[start_r:end_r, start_c:end_c]
            if not visible_grid.flags.c_contiguous:
                visible_grid = np.ascontiguousarray(visible_grid)
            v_rows, v_cols = visible_grid.shape
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_LUMINANCE, v_cols, v_rows, 0,
                            GL.GL_LUMINANCE, GL.GL_UNSIGNED_BYTE, visible_grid)
            self.needs_texture_upload = False
            self.last_bounds = (start_r, end_r, start_c, end_c)

        from OpenGL.GL import shaders
        shaders.glUseProgram(self.shader)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0, w, h, 0, -1, 1)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glTranslatef(-self.offset_x, -self.offset_y, 0)

        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0, 0); GL.glVertex2f(start_c * cell_px, start_r * cell_px)
        GL.glTexCoord2f(1, 0); GL.glVertex2f(end_c * cell_px, start_r * cell_px)
        GL.glTexCoord2f(1, 1); GL.glVertex2f(end_c * cell_px, end_r * cell_px)
        GL.glTexCoord2f(0, 1); GL.glVertex2f(start_c * cell_px, end_r * cell_px)
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
            self.needs_texture_upload = True
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
        self.needs_texture_upload = True
        self.update()
