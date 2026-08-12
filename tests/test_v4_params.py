"""V4 parameter resolution and validation tests."""
import numpy as np
import pytest

from denoise_astro_v4_mse import estimate_sensor_params
from utils.validation import validate_common_args, validate_v4_args

IMG = np.zeros((8, 8), dtype=np.float32)


def test_partial_gain_resolves_consistent_read_noise():
    gain, rn, origins = estimate_sensor_params(IMG, {}, gain=0.25)
    assert gain == 0.25
    assert rn == pytest.approx(8.0), '2 e- / gain with the PROVIDED gain'
    assert origins['gain'] == 'user'
    assert 'default' in origins['read_noise']


def test_header_readnois_picked_up():
    md = {'header': {'EGAIN': 1.5, 'READNOIS': 3.2}}
    gain, rn, origins = estimate_sensor_params(IMG, md)
    assert gain == 1.5
    assert rn == pytest.approx(3.2)
    assert origins['gain'] == 'header:EGAIN'
    assert origins['read_noise'].startswith('header:READNOIS')


def test_user_values_win_over_header():
    md = {'header': {'GAIN': 9.0, 'RDNOISE': 9.0}}
    gain, rn, origins = estimate_sensor_params(IMG, md, gain=2.0, read_noise=1.0)
    assert (gain, rn) == (2.0, 1.0)
    assert origins == {'gain': 'user', 'read_noise': 'user'}


def test_wmv_space_requires_transformed_loss():
    with pytest.raises(ValueError, match='transformed loss'):
        validate_v4_args('gat', 'raw', None, None, None, None)
    with pytest.raises(ValueError, match='transformed loss'):
        validate_v4_args('loss', 'raw', None, None, None, None)
    validate_v4_args('loss', 'asinh', None, None, None, None)
    validate_v4_args('gat', 'gat', None, None, None, None)


def test_common_args_validation():
    ok = dict(num_iter=100, learning_rate=0.01, reg_noise_std=0.04,
              percentiles=(0.5, 99.5), psnr_interval=10,
              es_window=100, es_patience=1000)
    validate_common_args(**ok)
    with pytest.raises(ValueError):
        validate_common_args(**{**ok, 'num_iter': 0})
    with pytest.raises(ValueError):
        validate_common_args(**{**ok, 'psnr_interval': 0})
    with pytest.raises(ValueError):
        validate_common_args(**{**ok, 'es_window': 0})
    with pytest.raises(ValueError):
        validate_common_args(**{**ok, 'percentiles': (99.5, 0.5)})
    with pytest.raises(ValueError):
        validate_v4_args('raw', 'gat', -1.0, None, None, None)
