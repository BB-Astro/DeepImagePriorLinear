"""Tiled driver tests: grid geometry, blending weights, format lock,
argument validation."""
import numpy as np
import pytest
from astropy.io import fits

from denoise_astro_tiled import tile_grid, crossfade_weights, denoise_astro_tiled
from utils.validation import validate_tiled_args


def test_grid_cores_partition_the_image():
    H, W, overlap = 500, 700, 32
    grid = tile_grid(H, W, 2, 3, overlap)
    covered = np.zeros((H, W), dtype=int)
    for t in grid:
        y0, y1, x0, x1 = t['core']
        covered[y0:y1, x0:x1] += 1
    assert (covered == 1).all(), 'cores must partition the image exactly once'
    for t in grid:
        y0, y1, x0, x1 = t['ext']
        cy0, cy1, cx0, cx1 = t['core']
        assert y0 <= cy0 and y1 >= cy1 and x0 <= cx0 and x1 >= cx1


def test_crossfade_weights_cover_everything():
    H, W, overlap = 300, 300, 40
    grid = tile_grid(H, W, 2, 2, overlap)
    wacc = np.zeros((H, W))
    for t in grid:
        y0, y1, x0, x1 = t['ext']
        wacc[y0:y1, x0:x1] += crossfade_weights(t, overlap)
    assert (wacc > 0).all(), 'every pixel must receive weight'
    # Core-only zones (outside any overlap) must have exactly weight 1
    inner = wacc[:100, :100]
    assert np.allclose(inner[:60, :60], 1.0)


def test_output_suffix_locked(tmp_path):
    with pytest.raises(ValueError, match='FITS'):
        denoise_astro_tiled('whatever.fit', str(tmp_path / 'out.tif'))


def test_grid_validation(tmp_path):
    data = np.random.default_rng(0).uniform(0, 1, (128, 128)).astype(np.float32)
    src = tmp_path / 'small.fit'
    fits.PrimaryHDU(data).writeto(src)
    with pytest.raises(ValueError, match='1x1'):
        denoise_astro_tiled(str(src), str(tmp_path / 'o.fit'), tiles='0x2')
    with pytest.raises(ValueError, match='too fine'):
        denoise_astro_tiled(str(src), str(tmp_path / 'o.fit'), tiles='4x4')
    with pytest.raises(ValueError, match='overlap'):
        denoise_astro_tiled(str(src), str(tmp_path / 'o.fit'),
                            tiles='2x2', overlap=64)


def test_validate_tiled_args_direct():
    validate_tiled_args(1000, 1000, 2, 2, 128)
    with pytest.raises(ValueError):
        validate_tiled_args(1000, 1000, 2, 2, -1)
    with pytest.raises(ValueError):
        validate_tiled_args(1000, 1000, 2, 2, 600)
