"""
Strict argument validation, run BEFORE any expensive work (network
creation, output directories, hours of optimization). Late failures on a
tiled run cost a night; these checks cost microseconds.

Author: Ben & Claude
Date: 2026-08-02
"""

from typing import Optional, Tuple


def validate_common_args(
    num_iter: int,
    learning_rate: float,
    reg_noise_std: float,
    percentiles: Tuple[float, float],
    psnr_interval: int,
    es_window: int,
    es_patience: int,
    es_min_start: int = 0,
    exp_weight: float = 0.99,
    norm_range: Optional[Tuple[float, float]] = None,
) -> None:
    if num_iter < 1:
        raise ValueError(f"num_iter must be >= 1, got {num_iter}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
    if reg_noise_std < 0:
        raise ValueError(f"reg_noise_std must be >= 0, got {reg_noise_std}")
    lo, hi = percentiles
    if not (0.0 <= lo < hi <= 100.0):
        raise ValueError(f"percentiles must satisfy 0 <= low < high <= 100, "
                         f"got ({lo}, {hi})")
    if psnr_interval < 1:
        raise ValueError(f"psnr_interval must be >= 1, got {psnr_interval}")
    if es_window < 2:
        raise ValueError(f"es_window must be >= 2, got {es_window}")
    if es_patience < 1:
        raise ValueError(f"es_patience must be >= 1, got {es_patience}")
    if es_min_start < 0:
        raise ValueError(f"es_min_start must be >= 0, got {es_min_start}")
    if not (0.0 < exp_weight <= 1.0):
        raise ValueError(f"exp_weight must be in (0, 1], got {exp_weight}")
    if norm_range is not None and not (norm_range[0] < norm_range[1]):
        raise ValueError(f"norm_range low must be < high, got {norm_range}")


def validate_v4_args(
    wmv_space: str,
    loss_space: str,
    gain: Optional[float],
    read_noise: Optional[float],
    adu_scale: Optional[float],
    asinh_beta: Optional[float],
) -> None:
    if loss_space not in ('gat', 'asinh', 'raw'):
        raise ValueError(f"loss_space must be gat, asinh or raw, got {loss_space}")
    if wmv_space not in ('raw', 'gat', 'loss'):
        raise ValueError(f"wmv_space must be raw, gat or loss, got {wmv_space}")
    if wmv_space in ('gat', 'loss') and loss_space == 'raw':
        raise ValueError(
            f"wmv_space '{wmv_space}' requires a transformed loss (gat or "
            f"asinh): with loss_space 'raw' the selection would silently run "
            f"in raw space while the sidecar claims otherwise")
    if gain is not None and gain <= 0:
        raise ValueError(f"gain must be > 0 e-/ADU, got {gain}")
    if read_noise is not None and read_noise < 0:
        raise ValueError(f"read_noise must be >= 0 ADU, got {read_noise}")
    if adu_scale is not None and adu_scale <= 0:
        raise ValueError(f"adu_scale must be > 0, got {adu_scale}")
    if asinh_beta is not None and asinh_beta <= 0:
        raise ValueError(f"asinh_beta must be > 0, got {asinh_beta}")


def validate_tiled_args(
    H: int,
    W: int,
    ny: int,
    nx: int,
    overlap: int,
) -> None:
    if ny < 1 or nx < 1:
        raise ValueError(f"tile grid must be at least 1x1, got {ny}x{nx}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    core_h = H // ny
    core_w = W // nx
    if core_h < 64 or core_w < 64:
        raise ValueError(
            f"grid {ny}x{nx} on a {W}x{H} image gives {core_w}x{core_h} px "
            f"cores: too fine (minimum 64 px per side)")
    if overlap >= min(core_h, core_w):
        raise ValueError(
            f"overlap {overlap} px must be smaller than the tile core "
            f"({core_w}x{core_h} px)")
