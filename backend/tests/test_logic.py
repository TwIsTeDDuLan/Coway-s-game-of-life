import numpy as np

from app.logic import CUDA_AVAILABLE, compute_next_generation, next_generation_cpu


def test_block_still_life_stays_stable() -> None:
    grid = np.zeros((4, 4), dtype=np.uint8)
    grid[1:3, 1:3] = 1

    next_grid = next_generation_cpu(grid)

    np.testing.assert_array_equal(next_grid, grid)


def test_blinker_oscillator_flips_axis() -> None:
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[2, 1:4] = 1

    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[1:4, 2] = 1

    next_grid = next_generation_cpu(grid)

    np.testing.assert_array_equal(next_grid, expected)


def test_compute_next_generation_returns_valid_mode() -> None:
    grid = np.zeros((3, 3), dtype=np.uint8)
    grid[1, 0:3] = 1

    next_grid, mode_used = compute_next_generation(grid, mode="gpu")
    print(f"mode_used: {mode_used}")
    assert mode_used in {"gpu", "cpu"}
    assert next_grid.shape == grid.shape

    if not CUDA_AVAILABLE:
        assert mode_used == "cpu"
