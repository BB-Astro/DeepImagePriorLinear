"""CPU smoke test: the V3 pipeline runs end to end on a tiny image.
No scientific quality claim, just plumbing: load, optimize a few
iterations, select, save, sidecar."""
import json

import numpy as np
import torch
from astropy.io import fits

import denoise_astro_v3


def test_v3_end_to_end_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(denoise_astro_v3, 'get_device',
                        lambda: torch.device('cpu'))

    rng = np.random.default_rng(3)
    data = (0.001 + 0.0002 * rng.standard_normal((64, 64))).astype(np.float32)
    data[20, 20] = 0.9
    src = tmp_path / 'tiny.fit'
    fits.PrimaryHDU(data).writeto(src)

    out = tmp_path / 'tiny_dn.fit'
    results = denoise_astro_v3.denoise_astro_image_v3(
        str(src), num_iter=60, save_path=str(out),
        show_progress=False, seed=7, psnr_interval=20,
        es_window=10, es_patience=30)

    assert out.exists()
    saved = np.asarray(fits.getdata(out), dtype=np.float32)
    assert saved.shape == (64, 64)
    assert np.isfinite(saved).all()
    assert saved[20, 20] == data[20, 20], 'bright core reinjected verbatim'

    with open(str(out) + '.json') as f:
        sidecar = json.load(f)
    assert sidecar['seed'] == 7
    assert sidecar['pipeline'] == 'dip-astro-v3'
    assert 0 <= sidecar['best_iteration'] < 60
