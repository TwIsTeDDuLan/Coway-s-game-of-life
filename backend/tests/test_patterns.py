import numpy as np
from app.patterns import parse_rle, tile_pattern, place_pattern_centered, random_fill

def test_parse_rle():
    rle = """#C Block
x = 2, y = 2, rule = B3/S23
2o$2o!"""
    width, height, cells = parse_rle(rle)
    assert width == 2
    assert height == 2
    assert cells.shape == (2, 2)
    assert np.array_equal(cells, np.array([[1, 1], [1, 1]]))

def test_parse_rle_glider():
    rle = """#C Glider
x = 3, y = 3, rule = B3/S23
bob$2bo$3o!"""
    width, height, cells = parse_rle(rle)
    assert width == 3
    assert height == 3
    assert cells.shape == (3, 3)
    expected = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)
    assert np.array_equal(cells, expected)

def test_tile_pattern():
    grid = np.zeros((10, 10), dtype=np.uint8)
    pattern = np.ones((2, 2), dtype=np.uint8)
    # Pat size 2x2, spacing 2 -> steps of 4
    # origins: (0,0), (0,4), (0,8), (4,0), (4,4), (4,8), (8,0), (8,4), (8,8)
    tiled = tile_pattern(grid, pattern, spacing_x=2, spacing_y=2)
    
    assert tiled[0, 0] == 1
    assert tiled[0, 4] == 1
    assert tiled[8, 8] == 1
    assert tiled[1, 1] == 1
    # Check spacing
    assert tiled[2, 0] == 0
    assert tiled[0, 2] == 0

def test_place_pattern_centered():
    grid = np.zeros((10, 10), dtype=np.uint8)
    pattern = np.ones((2, 2), dtype=np.uint8)
    # Center of 10x10 with 2x2 pattern:
    # start_r = (10 - 2) // 2 = 4
    # start_c = (10 - 2) // 2 = 4
    centered = place_pattern_centered(grid, pattern)
    
    assert centered[4, 4] == 1
    assert centered[5, 5] == 1
    assert centered[3, 3] == 0
    assert centered[6, 6] == 0
    # ensure it's the only ones
    assert np.sum(centered) == 4

def test_random_fill():
    grid = np.zeros((100, 100), dtype=np.uint8)
    random_fill(grid, density=0.3)
    
    alive_count = np.sum(grid)
    # Should be around 3000 out of 10000
    assert 2500 < alive_count < 3500
