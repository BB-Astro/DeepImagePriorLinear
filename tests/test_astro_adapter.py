"""I/O regression tests for utils/astro_adapter.py.

Every test here answers a failure found in the 2026-08-02 code review:
uint16 BZERO corruption, NaN propagation, FITS axis conventions, stale
pipeline header cards, float32 scaling without a source range.
"""
import numpy as np
import pytest
from astropy.io import fits

from utils.astro_adapter import AstroImageHandler


def write_fits(path, data, header=None):
    hdu = fits.PrimaryHDU(data)
    if header:
        for k, v in header.items():
            hdu.header[k] = v
    hdu.writeto(path, overwrite=True)
    return path


def test_uint16_bzero_roundtrip(tmp_path):
    """A uint16 FITS (BZERO=32768) must round-trip without any offset."""
    values = np.array([[0, 1], [32768, 65535]], dtype=np.uint16)
    src = write_fits(tmp_path / 'u16.fit', values)
    with fits.open(src) as h:
        assert h[0].header.get('BZERO') == 32768, 'astropy should scale uint16'

    data_norm, metadata = AstroImageHandler.load_astro_image(
        src, target_channels=1, norm_range=(0.0, 65535.0))
    out = tmp_path / 'u16_out.fit'
    AstroImageHandler.save_astro_image(data_norm, metadata, out, bit_depth=32)

    with fits.open(out) as h:
        assert 'BZERO' not in h[0].header
        assert 'BSCALE' not in h[0].header
        saved = np.asarray(h[0].data, dtype=np.float64)
    assert np.allclose(saved, values.astype(np.float64), atol=0.5)


def test_nan_rejected(tmp_path):
    data = np.ones((8, 8), dtype=np.float32)
    data[3, 4] = np.nan
    src = write_fits(tmp_path / 'nan.fit', data)
    with pytest.raises(ValueError, match='non-finite'):
        AstroImageHandler.load_astro_image(src, target_channels=1)
    with pytest.raises(ValueError, match='non-finite'):
        AstroImageHandler.load_raw(src)


def test_float32_roundtrip_with_reinjection(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.uniform(1e-3, 2e-3, (64, 64)).astype(np.float32)
    data[10, 10] = 0.9  # bright core above the window
    src = write_fits(tmp_path / 'f32.fit', data)

    data_norm, metadata = AstroImageHandler.load_astro_image(
        src, target_channels=1, percentiles=(0.5, 99.5))
    assert metadata['clip_high_count'] >= 1
    out = tmp_path / 'f32_out.fit'
    AstroImageHandler.save_astro_image(data_norm, metadata, out, bit_depth=32)

    saved = np.asarray(fits.getdata(out), dtype=np.float32)
    mask = metadata['clip_high_mask']
    assert np.array_equal(saved[mask], data[mask]), 'reinjection must be bit-exact'
    with fits.open(out) as h:
        assert h[0].header['NREINJ'] == int(mask.sum())


def test_fits_channel_first_cube_collapses_to_mono(tmp_path):
    plane = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    cube = np.stack([plane, plane, plane])          # (3, H, W), FITS style
    src = write_fits(tmp_path / 'rgb.fit', cube)
    data_norm, metadata = AstroImageHandler.load_astro_image(
        src, target_channels=1)
    assert data_norm.shape == (64, 64, 1)


def test_ambiguous_cube_rejected(tmp_path):
    cube = np.zeros((5, 64, 64), dtype=np.float32)
    src = write_fits(tmp_path / 'cube5.fit', cube)
    with pytest.raises(ValueError, match='ambiguous'):
        AstroImageHandler.load_raw(src)


def test_float32_without_source_range_keeps_unit_interval(tmp_path):
    data = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)
    out = tmp_path / 'norange.fit'
    AstroImageHandler.save_astro_image(data, {}, out, bit_depth=32)
    saved = np.asarray(fits.getdata(out), dtype=np.float32)
    assert saved.max() <= 1.0
    assert np.allclose(saved, data)


def test_stale_pipeline_cards_do_not_survive(tmp_path):
    """A source that already went through the pipeline must not leak its
    old ORIGMIN/NREINJ/BZERO into the new output header."""
    data = np.linspace(1e-3, 2e-3, 64 * 64, dtype=np.float32).reshape(64, 64)
    src = write_fits(tmp_path / 'stale.fit', data,
                     header={'NREINJ': 99999, 'ORIGMIN': -1.0, 'OBJECT': 'M31'})
    data_norm, metadata = AstroImageHandler.load_astro_image(
        src, target_channels=1)
    out = tmp_path / 'stale_out.fit'
    AstroImageHandler.save_astro_image(data_norm, metadata, out, bit_depth=32)
    with fits.open(out) as h:
        hdr = h[0].header
        assert hdr['NREINJ'] == metadata['reinjected_count']
        assert hdr['ORIGMIN'] == pytest.approx(metadata['original_min'])
        assert hdr['OBJECT'] == 'M31', 'non-structural cards must survive'


def test_xisf_roundtrip_with_reinjection(tmp_path):
    """XISF output: bit-exact roundtrip, reinjection, pipeline keywords."""
    xisf = pytest.importorskip('xisf')
    rng = np.random.default_rng(0)
    data = rng.uniform(1e-3, 2e-3, (64, 64)).astype(np.float32)
    data[10, 10] = 0.9  # bright core above the window
    src = write_fits(tmp_path / 'f32.fit', data)

    data_norm, metadata = AstroImageHandler.load_astro_image(
        src, target_channels=1, percentiles=(0.5, 99.5))
    assert metadata['clip_high_count'] >= 1
    out = tmp_path / 'f32_out.xisf'
    AstroImageHandler.save_astro_image(data_norm, metadata, out, bit_depth=32)

    saved = xisf.XISF(str(out)).read_image(0).squeeze().astype(np.float32)
    mask = metadata['clip_high_mask']
    assert np.array_equal(saved[mask], data[mask]), 'reinjection must be bit-exact'
    keywords = xisf.XISF(str(out)).get_images_metadata()[0]['FITSKeywords']
    assert keywords['NREINJ'][0]['value'] == str(int(mask.sum()))


def test_xisf_input_to_xisf_output_carries_properties(tmp_path):
    """An XISF source's properties must survive into an XISF output."""
    xisf = pytest.importorskip('xisf')
    data = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64, 1)
    src = tmp_path / 'src.xisf'
    xisf.XISF.write(str(src), data)

    data_norm, metadata = AstroImageHandler.load_astro_image(
        src, target_channels=1, percentiles=(0.5, 99.5))
    out = tmp_path / 'out.xisf'
    AstroImageHandler.save_astro_image(data_norm, metadata, out, bit_depth=32)

    back = xisf.XISF(str(out)).read_image(0)
    assert back.squeeze().shape == (64, 64)
    assert back.dtype == np.float32
