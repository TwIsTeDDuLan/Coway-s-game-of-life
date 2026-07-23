# Future Enhancements

## 1. WebGL / OffscreenCanvas Rendering (IMPLEMENTED)

**Problem:** Canvas 2D `fillRect` calls become slow when rendering millions of visible cells at low zoom levels.

**Solution:** *Implemented via PyQt5 + PyOpenGL.* The grid is now rendered as a single `GL_LUMINANCE` texture mapped to a fullscreen quad. A fragment shader colors the alive/dead cells. This handles grids up to 50,000 × 50,000 trivially.

---

## 2. Chunk-Based Grid Storage

**Problem:** A 100,000 × 100,000 grid would require 10 GB of contiguous RAM/VRAM just for the 1-byte cell array.

**Solution:** Use a sparse / chunked storage model:
- Divide the grid into fixed-size chunks (e.g., 256×256 or 1024×1024).
- Only allocate chunks that contain alive cells (Hash Map / Quadtree).
- The simulation kernel operates chunk-by-chunk.
- The OpenGL renderer requests only chunks overlapping the viewport and uploads them as multiple small textures, or uses bindless textures.

---

## 3. Advanced Touch / Pen Gesture Support

If running on a touch-enabled desktop or tablet (e.g., Surface Pro):
- Pinch-to-zoom mapping to OpenGL scale.
- Two-finger pan mapping to offset translation.

---

## 4. Multi-State Rulesets (Generations / Wireworld)

Expand beyond binary (alive/dead) B3/S23 to support rules with multiple states.
- E.g., Brian's Brain, Wireworld, or generalized Generations rules.
- Update OpenGL fragment shader to map integer state values to a color palette (e.g., 1D texture lookup).
