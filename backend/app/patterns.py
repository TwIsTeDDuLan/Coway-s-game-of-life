import re
import numpy as np

# ── RLE Parser ──────────────────────────────────────────────────────────

def parse_rle(rle_string: str) -> tuple[int, int, np.ndarray]:
    """
    Parse an RLE string into a 2D numpy array.
    Returns (width, height, cells) where cells[row, col] = 0 | 1
    """
    lines = [line.strip() for line in rle_string.split('\n') if line.strip()]

    header_line = ""
    body_lines = []
    past_header = False

    for line in lines:
        if line.startswith('#'):
            continue
        if not past_header:
            if re.search(r'x\s*=', line, re.IGNORECASE):
                header_line = line
                past_header = True
                continue
        body_lines.append(line)

    x_match = re.search(r'x\s*=\s*(\d+)', header_line, re.IGNORECASE)
    y_match = re.search(r'y\s*=\s*(\d+)', header_line, re.IGNORECASE)
    
    width = int(x_match.group(1)) if x_match else 0
    height = int(y_match.group(1)) if y_match else 0

    body = "".join(body_lines).replace(" ", "").replace("\r", "").replace("\n", "")

    cells = np.zeros((height, width), dtype=np.uint8)
    row = 0
    col = 0
    run_count = ""

    for ch in body:
        if ch == '!':
            break

        if '0' <= ch <= '9':
            run_count += ch
            continue

        count = int(run_count) if run_count else 1
        run_count = ""

        if ch == 'b':
            col += count
        elif ch == 'o':
            for _ in range(count):
                if row < height and col < width:
                    cells[row, col] = 1
                col += 1
        elif ch == '$':
            row += count
            col = 0

    return width, height, cells


# ── Pattern Tiling ──────────────────────────────────────────────────────

def tile_pattern(grid: np.ndarray, pattern: np.ndarray, spacing_x: int = 2, spacing_y: int = 2) -> np.ndarray:
    """
    Place a parsed pattern onto a grid, tiling it to fill the entire grid.
    Mutates and returns the grid.
    """
    grid_rows, grid_cols = grid.shape
    pat_h, pat_w = pattern.shape

    step_x = pat_w + spacing_x
    step_y = pat_h + spacing_y

    for origin_y in range(0, grid_rows, step_y):
        for origin_x in range(0, grid_cols, step_x):
            # Calculate slices to avoid out-of-bounds indexing
            h = min(pat_h, grid_rows - origin_y)
            w = min(pat_w, grid_cols - origin_x)
            if h > 0 and w > 0:
                grid[origin_y:origin_y+h, origin_x:origin_x+w] |= pattern[:h, :w]

    return grid


def place_pattern_centered(grid: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """
    Place a single copy of a pattern centered on the grid.
    Mutates and returns the grid.
    """
    grid_rows, grid_cols = grid.shape
    pat_h, pat_w = pattern.shape

    start_r = max(0, (grid_rows - pat_h) // 2)
    start_c = max(0, (grid_cols - pat_w) // 2)

    h = min(pat_h, grid_rows - start_r)
    w = min(pat_w, grid_cols - start_c)
    
    if h > 0 and w > 0:
        grid[start_r:start_r+h, start_c:start_c+w] |= pattern[:h, :w]

    return grid


# ── Random Fill ─────────────────────────────────────────────────────────

def random_fill(grid: np.ndarray, density: float = 0.3) -> np.ndarray:
    """
    Fill a grid with random alive/dead cells.
    Mutates and returns the grid.
    
    Uses a chunked approach to avoid allocating a single massive temporary
    array (e.g. np.random.rand on a 40K×40K grid would create a 12 GB float64 array).
    """
    rows, cols = grid.shape
    threshold = int(density * 256)  # Map density to 0-255 range
    
    # Process in chunks of rows to keep peak memory low
    chunk_size = max(1, min(1000, rows))
    for start in range(0, rows, chunk_size):
        end = min(start + chunk_size, rows)
        # randint with dtype=np.uint8 uses 1 byte/cell instead of 8
        grid[start:end, :] = (
            np.random.randint(0, 256, size=(end - start, cols), dtype=np.uint8) < threshold
        ).astype(np.uint8)
    
    return grid


# ── Pattern Library ─────────────────────────────────────────────────────

PATTERN_LIBRARY = [
    # ─── Still Lifes ────────────────────────────────
    {
        "name": 'Block',
        "category": 'Still Life',
        "description": 'The simplest still life – a 2×2 square.',
        "rle": "#C Block\nx = 2, y = 2, rule = B3/S23\n2o$2o!"
    },
    {
        "name": 'Beehive',
        "category": 'Still Life',
        "description": 'A 6-cell still life shaped like a beehive.',
        "rle": "#C Beehive\nx = 4, y = 3, rule = B3/S23\nb2ob$o2bo$b2o!"
    },
    {
        "name": 'Loaf',
        "category": 'Still Life',
        "description": 'A 7-cell still life.',
        "rle": "#C Loaf\nx = 4, y = 4, rule = B3/S23\nb2ob$o2bo$bobo$2bo!"
    },
    {
        "name": 'Boat',
        "category": 'Still Life',
        "description": 'A 5-cell still life.',
        "rle": "#C Boat\nx = 3, y = 3, rule = B3/S23\n2ob$obo$bo!"
    },
    {
        "name": 'Tub',
        "category": 'Still Life',
        "description": 'A 4-cell still life.',
        "rle": "#C Tub\nx = 3, y = 3, rule = B3/S23\nbob$obo$bo!"
    },

    # ─── Oscillators ────────────────────────────────
    {
        "name": 'Blinker',
        "category": 'Oscillator',
        "description": 'The smallest oscillator – period 2.',
        "rle": "#C Blinker\nx = 3, y = 1, rule = B3/S23\n3o!"
    },
    {
        "name": 'Toad',
        "category": 'Oscillator',
        "description": 'A period-2 oscillator.',
        "rle": "#C Toad\nx = 4, y = 2, rule = B3/S23\nb3o$3ob!"
    },
    {
        "name": 'Beacon',
        "category": 'Oscillator',
        "description": 'A period-2 oscillator made of two diagonal blocks.',
        "rle": "#C Beacon\nx = 4, y = 4, rule = B3/S23\n2o2b$o3b$3bo$2b2o!"
    },
    {
        "name": 'Pulsar',
        "category": 'Oscillator',
        "description": 'A famous period-3 oscillator with 4-fold symmetry.',
        "rle": "#C Pulsar\nx = 13, y = 13, rule = B3/S23\n2b3o3b3o2b$13b$o4bobo4bo$o4bobo4bo$o4bobo4bo$2b3o3b3o2b$13b$2b3o3b3o\n2b$o4bobo4bo$o4bobo4bo$o4bobo4bo$13b$2b3o3b3o!"
    },
    {
        "name": 'Pentadecathlon',
        "category": 'Oscillator',
        "description": 'A period-15 oscillator – the longest-period common oscillator.',
        "rle": "#C Pentadecathlon\nx = 10, y = 3, rule = B3/S23\n2bo4bo2b$2ob4ob2o$2bo4bo!"
    },

    # ─── Spaceships ─────────────────────────────────
    {
        "name": 'Glider',
        "category": 'Spaceship',
        "description": 'The smallest spaceship – moves diagonally.',
        "rle": "#C Glider\nx = 3, y = 3, rule = B3/S23\nbob$2bo$3o!"
    },
    {
        "name": 'LWSS',
        "category": 'Spaceship',
        "description": 'Lightweight spaceship – moves horizontally.',
        "rle": "#C Lightweight spaceship\nx = 5, y = 4, rule = B3/S23\nbo2bo$o4b$o3bo$4o!"
    },
    {
        "name": 'MWSS',
        "category": 'Spaceship',
        "description": 'Middleweight spaceship.',
        "rle": "#C Middleweight spaceship\nx = 6, y = 5, rule = B3/S23\n3bo2b$bo3bo$o5b$o4bo$5o!"
    },
    {
        "name": 'HWSS',
        "category": 'Spaceship',
        "description": 'Heavyweight spaceship.',
        "rle": "#C Heavyweight spaceship\nx = 7, y = 5, rule = B3/S23\n3b2o2b$bo4bo$o6b$o5bo$6o!"
    },

    # ─── Guns ───────────────────────────────────────
    {
        "name": 'Gosper Glider Gun',
        "category": 'Gun',
        "description": 'The first known gun – emits a glider every 30 generations.',
        "rle": "#C Gosper glider gun\nx = 36, y = 9, rule = B3/S23\n24bo11b$22bobo11b$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o14b$2o8b\no3bob2o4bobo11b$10bo5bo7bo11b$11bo3bo20b$12b2o!"
    },
    {
        "name": 'Simkin Glider Gun',
        "category": 'Gun',
        "description": 'A small gun discovered in 2015 – period 120.',
        "rle": "#C Simkin glider gun\nx = 33, y = 21, rule = B3/S23\n2o5b2o$2o5b2o2$4b2o$4b2o5$22b2ob2o$21bo4bo$21bo5bo2b2o$21b3o2bo3b2o$\n26bo4$20b2o$20bo$21b3o$23bo!"
    },

    # ─── Methuselahs ────────────────────────────────
    {
        "name": 'R-pentomino',
        "category": 'Methuselah',
        "description": 'A 5-cell pattern that takes 1103 generations to stabilize.',
        "rle": "#C R-pentomino\nx = 3, y = 3, rule = B3/S23\nb2o$2ob$bo!"
    },
    {
        "name": 'Diehard',
        "category": 'Methuselah',
        "description": 'A 7-cell pattern that vanishes after 130 generations.',
        "rle": "#C Diehard\nx = 8, y = 3, rule = B3/S23\n6bob$2o6b$bo3b3o!"
    },
    {
        "name": 'Acorn',
        "category": 'Methuselah',
        "description": 'A 7-cell pattern that takes 5206 generations to stabilize.',
        "rle": "#C Acorn\nx = 7, y = 3, rule = B3/S23\nbo5b$3bo3b$2o2b3o!"
    },

    # ─── Other / Complex ────────────────────────────
    {
        "name": 'Copperhead',
        "category": 'Spaceship',
        "description": 'A c/10 spaceship discovered in 2016.',
        "rle": "#C Copperhead\nx = 8, y = 12, rule = B3/S23\nb2o2b2o$3b2o$3b2o$obo2bobo$o6bo2$o6bo$b2o2b2o$2b4o2$3b2o$3b2o!"
    },
    {
        "name": 'Puffer Train',
        "category": 'Puffer',
        "description": 'Leaves behind debris as it moves.',
        "rle": "#C Puffer train\nx = 5, y = 18, rule = B3/S23\nbo3b$2bo2b$3o2b5$bo3b$2bo2b$o4b$o3bo$4o4$3b2o$3b2o$3b2o!"
    },
    {
        "name": 'Infinite Growth 1',
        "category": 'Methuselah',
        "description": 'A line of 10 cells that grows forever.',
        "rle": "#C 10-cell infinite growth\nx = 10, y = 1, rule = B3/S23\n10o!"
    },
    {
        "name": 'Glider Eater',
        "category": 'Still Life',
        "description": 'An eater that can absorb a glider without changing.',
        "rle": "#C Eater 1\nx = 4, y = 4, rule = B3/S23\n2o2b$obo$2bo$2b2o!"
    },
]

# Unique categories preserving order
PATTERN_CATEGORIES = []
for p in PATTERN_LIBRARY:
    if p["category"] not in PATTERN_CATEGORIES:
        PATTERN_CATEGORIES.append(p["category"])
