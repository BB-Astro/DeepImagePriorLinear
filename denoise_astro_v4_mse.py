#!/usr/bin/env python3
"""
Deep Image Prior - Astronomical Image Denoising V4 MSE
AMÉLIORATION MSE : GAT pour stabilisation de variance Poisson-Gaussienne

Basé sur V3 avec ajout de :
- Generalized Anscombe Transform (GAT) pour stabiliser la variance
- Loss MSE dans l'espace transformé T(output) vs T(input)
- Paramètres capteur (gain, read noise) pour modélisation physique correcte
- Le réseau travaille toujours en RAW (Variante A du plan)

Innovation BB+Claude :
- DIP travaille sur image RAW (pas de transformation)
- Loss calculée dans l'espace GAT stabilisé
- PSNR stretché toujours utilisé pour guidage visuel
- Résultat final = RAW avec dynamique préservée ET meilleur traitement du bruit

Author: Ben & Claude
Date: 2025-11-12
"""

import copy
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models import get_net
from utils.common_utils import get_noise
from utils.astro_adapter import AstroImageHandler
from utils.early_stopping import ESWMV
from utils.sidecar import write_run_sidecar
from utils.validation import validate_common_args, validate_v4_args


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def generalized_anscombe_transform(y_adu: torch.Tensor, gain: float, read_noise: float) -> torch.Tensor:
    """
    Generalized Anscombe Transform (GAT) for variance stabilization

    Transforms Poisson-Gaussian noise to approximately Gaussian with unit variance.
    The input MUST be in ADU (detector units), not normalized [0,1] data: the
    caller is responsible for denormalizing first. Applying this formula to
    normalized data makes the 3/8 term dominate and the transform quasi affine
    (the exact bug that invalidated the November 2025 V3/V4 comparison).

    Args:
        y_adu: Input image in ADU (after bias subtraction)
        gain: Camera gain in e-/ADU
        read_noise: Read noise in ADU

    Returns:
        Transformed image with stabilized variance

    Reference:
        Mäkitalo & Foi (2013) "Optimal Inversion of the Generalized Anscombe Transformation"
    """
    alpha = 1.0 / gain  # Conversion factor (ADU/e-)

    # GAT formula: T(y) = 2/alpha * sqrt(alpha*y + 3/8*alpha^2 + sigma_r^2)
    # This stabilizes the Poisson-Gaussian noise to ~unit variance
    inside_sqrt = alpha * y_adu + (3.0/8.0) * alpha**2 + read_noise**2

    # Ensure no negative values under sqrt (can happen with noise)
    inside_sqrt = torch.clamp(inside_sqrt, min=1e-10)

    transformed = (2.0 / alpha) * torch.sqrt(inside_sqrt)

    return transformed


def estimate_sensor_params(
    img_np: np.ndarray,
    metadata: Dict,
    gain: Optional[float] = None,
    read_noise: Optional[float] = None,
) -> Tuple[float, float, Dict]:
    """
    Resolve sensor parameters, honoring any value already provided.

    Resolution order per parameter: caller-provided, then FITS header,
    then default. The default read noise is derived from the RESOLVED gain
    (2 e- / gain), so a user-provided gain is never combined with a default
    computed from another gain.

    Caution on units: header read-noise cards (RDNOISE/READNOIS/RON) carry
    no unit information and may be in electrons on some instruments; they
    are used as ADU here and their origin is reported so the sidecar keeps
    the provenance.

    Returns:
        (gain [e-/ADU], read_noise [ADU], origins dict)
    """
    # FITS headers live under metadata['header'], not at the top level.
    # Merge both so either location works. Note FITS keywords are at most
    # 8 characters: the read noise key is READNOIS, not READNOISE.
    lookup = {}
    header = metadata.get('header')
    if isinstance(header, dict):
        lookup.update({str(k).upper(): v for k, v in header.items()})
    lookup.update({str(k).upper(): v for k, v in metadata.items()
                   if isinstance(v, (int, float))})

    origins = {}

    if gain is not None:
        origins['gain'] = 'user'
    else:
        for key in ('GAIN', 'EGAIN'):
            if isinstance(lookup.get(key), (int, float)):
                gain = lookup[key]
                origins['gain'] = f'header:{key}'
                break
        else:
            gain = 1.0
            origins['gain'] = 'default'
            print("⚠️ Gain not provided nor found in header: default 1.0 e-/ADU")
            print("   For stacked images, use the effective gain: gain × number of frames")

    if read_noise is not None:
        origins['read_noise'] = 'user'
    else:
        for key in ('RDNOISE', 'READNOIS', 'READNOISE', 'RON'):
            if isinstance(lookup.get(key), (int, float)):
                read_noise = lookup[key]
                origins['read_noise'] = f'header:{key} (unit assumed ADU)'
                break
        else:
            # ~2 e- read noise converted to ADU with the RESOLVED gain
            read_noise = 2.0 / float(gain)
            origins['read_noise'] = 'default (2 e- / gain)'
            print(f"⚠️ Read noise not provided nor found: default {read_noise:.2f} ADU")

    print(f"📊 Sensor parameters: gain {gain} e-/ADU ({origins['gain']}), "
          f"read noise {read_noise:.4g} ADU ({origins['read_noise']})")

    return float(gain), float(read_noise), origins


def stretch_arcsinh(img, beta=0.05):
    """Arcsinh stretch for PSNR calculation ONLY"""
    stretched = np.arcsinh(img / beta) / np.arcsinh(1.0 / beta)
    return np.clip(stretched, 0, 1)


from denoise_astro_v3 import stf_autostretch


def calculate_psnr_stretched(img1, img2, beta=0.05):
    """
    Calculate PSNR in STRETCHED space (for visual quality guidance)
    """
    img1_stretched = stretch_arcsinh(img1, beta)
    img2_stretched = stretch_arcsinh(img2, beta)

    mse = np.mean((img1_stretched - img2_stretched) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


def denoise_astro_image_v4_mse(
    image_path: str,
    num_iter: int = 3000,
    learning_rate: float = 0.01,
    reg_noise_std: float = 0.04,
    save_path: Optional[str] = None,
    show_progress: bool = True,
    percentiles: Tuple[float, float] = (0.5, 99.5),
    preserve_dynamics: bool = True,
    psnr_stretch_beta: float = 0.05,
    save_intermediate: bool = False,
    intermediate_interval: int = 500,
    psnr_interval: int = 10,
    early_stop: bool = True,
    es_window: int = 100,
    es_patience: int = 1000,
    es_min_start: int = 0,
    # New V4 parameters for MSE stabilization
    gain: Optional[float] = None,
    read_noise: Optional[float] = None,
    use_gat: bool = True,  # Enable GAT by default
    adu_scale: Optional[float] = None,
    seed: Optional[int] = None,
    norm_range: Optional[Tuple[float, float]] = None,
    wmv_space: str = 'raw',
    loss_space: Optional[str] = None,
    asinh_beta: Optional[float] = None,
    grad_clip: Optional[float] = None
) -> Dict:
    """
    V4 MSE: DIP with variance-stabilized loss using Generalized Anscombe Transform

    KEY INNOVATIONS:
    - GAT stabilizes Poisson-Gaussian noise variance
    - MSE calculated in transformed space: Loss = ||T(output) - T(input)||²
    - Network still works on RAW (Variante A from plan)
    - PSNR stretched still used for visual quality guidance
    - Physically correct noise model respects sensor characteristics

    Args:
        image_path: Path to input astronomical image
        num_iter: Number of optimization iterations
        learning_rate: Learning rate for optimizer
        reg_noise_std: Regularization noise standard deviation
        save_path: Path to save output
        show_progress: Show progress during optimization
        percentiles: Percentiles for normalization
        preserve_dynamics: Preserve original dynamic range
        psnr_stretch_beta: Beta parameter for PSNR stretch calculation
        save_intermediate: Save intermediate results
        intermediate_interval: Interval for saving intermediate results
        gain: Camera gain in e-/ADU (if None, will try to extract/estimate)
        read_noise: Read noise in ADU (if None, will try to extract/estimate)
        use_gat: Use GAT for variance stabilization (can disable for comparison)
        adu_scale: File-value to ADU conversion factor. None = auto: 65535 for
            float files pre-normalized to [0,1] (XISF from PixInsight), 1 for
            files already in ADU
        seed: Random seed (None = drawn from the OS and logged)
    """

    # Loss space: 'gat' (variance-stabilized, rigorous), 'asinh' (aggressive
    # faint-structure weighting, like the successful 2025 runs on stretched
    # images, photometric bias to be checked), 'raw' (plain MSE, V3-like).
    # Defaults to 'gat'/'raw' according to use_gat for backward compatibility.
    if loss_space is None:
        loss_space = 'gat' if use_gat else 'raw'
    use_gat = (loss_space == 'gat')

    validate_common_args(num_iter, learning_rate, reg_noise_std, percentiles,
                         psnr_interval, es_window, es_patience, es_min_start,
                         norm_range=norm_range)
    validate_v4_args(wmv_space, loss_space, gain, read_noise, adu_scale,
                     asinh_beta)

    # Same seeding policy as V3: torch (init + input + reg noise) and numpy.
    # The ES-WMV pixel subsample keeps its own fixed seed for comparability.
    if seed is None:
        seed = int.from_bytes(os.urandom(4), 'little')
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    print(f"🚀 Using device: {device}")
    print(f"🎲 Seed: {seed}")
    print(f"\n🔬 V4 MSE WORKFLOW (Variance-stabilized optimization):")
    print(f"   1. DIP on RAW image (no stretch)")
    if use_gat:
        print(f"   2. Loss = MSE(T(output), T(input)) with GAT stabilization")
        print(f"   3. GAT handles Poisson-Gaussian noise correctly")
    else:
        print(f"   2. Loss = MSE(output, input) [GAT disabled for comparison]")
    print(f"   4. PSNR calculated in STRETCHED space for visual guidance")
    print(f"   5. Output = RAW with dynamics preserved + better noise handling")

    # Load image (RAW, no stretch)
    print(f"\n📷 Loading astronomical image: {image_path}")
    img_np, metadata = AstroImageHandler.load_astro_image(
        image_path,
        preserve_range=preserve_dynamics,
        percentiles=percentiles,
        target_channels=1,
        norm_range=norm_range
    )

    print(f"📊 Image statistics:")
    if 'original_min' in metadata:
        print(f"   Original range: [{metadata['original_min']:.6g}, {metadata['original_max']:.6g}]")
    print(f"   Shape: {img_np.shape}")
    print(f"   Current range: [{np.min(img_np):.3f}, {np.max(img_np):.3f}]")
    print(f"   ✅ Image kept in normalized RAW space [0,1]")

    # Get sensor parameters
    offset_adu = 0.0
    range_adu = 1.0
    sensor_origins = None
    if use_gat:
        gain, read_noise, sensor_origins = estimate_sensor_params(
            img_np, metadata, gain=gain, read_noise=read_noise)

        # The GAT is only valid in ADU. The network tensors are normalized to
        # [0,1] over the percentile window [p_low, p_high] in FILE units, so the
        # loss denormalizes back to ADU first:
        #     y_adu = (y_norm * (p_high - p_low) + p_low) * adu_scale
        # adu_scale converts file units to ADU. Integer FITS are already in
        # ADU (scale 1). Float files pre-normalized to [0,1] (XISF or FITS
        # exported by PixInsight) were divided by 65535: that scale is assumed
        # by default and can be overridden with --adu-scale.
        p_low = float(metadata.get('original_min', 0.0))
        p_high = float(metadata.get('original_max', 1.0))
        if adu_scale is None:
            adu_scale = 65535.0 if p_high <= 1.5 else 1.0
            print(f"   ADU scale (auto): {adu_scale:.6g}")
        else:
            print(f"   ADU scale (user): {adu_scale:.6g}")
        offset_adu = p_low * adu_scale
        range_adu = (p_high - p_low) * adu_scale
        print(f"   GAT window in ADU: [{offset_adu:.6g}, {offset_adu + range_adu:.6g}]")

    # Convert to torch
    if len(img_np.shape) == 2:
        img_np = img_np[np.newaxis, ...]
    elif len(img_np.shape) == 3:
        img_np = img_np.transpose(2, 0, 1)

    img_torch = torch.from_numpy(img_np.astype(np.float32)).unsqueeze(0).to(device)
    n_channels = img_torch.shape[1]

    # Apply GAT to input for loss calculation (in ADU, see above)
    if use_gat:
        img_torch_adu = img_torch * range_adu + offset_adu
        img_torch_gat = generalized_anscombe_transform(img_torch_adu, gain, read_noise)
        print(f"\n✨ GAT applied to input image (ADU space)")
        print(f"   Input range (norm): [{img_torch.min().item():.3f}, {img_torch.max().item():.3f}]")
        print(f"   Input range (ADU): [{img_torch_adu.min().item():.6g}, {img_torch_adu.max().item():.6g}]")
        print(f"   GAT range: [{img_torch_gat.min().item():.6g}, {img_torch_gat.max().item():.6g}]")

    # Asinh loss space: beta anchored on the background noise in NORMALIZED
    # units so the faint end is actually stretched (a fixed beta goes myopic
    # when the normalization window changes, the p99.5/p99.99 lesson).
    if loss_space == 'asinh':
        flat = img_torch.reshape(-1)
        med_n = flat.median().item()
        sigma_bg_norm = 1.4826 * (flat - med_n).abs().median().item()
        beta_origin = 'user'
        if asinh_beta is None:
            asinh_beta = max(1e-5, 4.0 * sigma_bg_norm)
            beta_origin = 'auto = 4 x sigma_bg_norm'
        img_torch_l = torch.asinh(img_torch / asinh_beta)
        print(f"\n✨ Asinh loss space: beta={asinh_beta:.4g} ({beta_origin}, "
              f"sigma_bg_norm={sigma_bg_norm:.4g})")
        print(f"   Stretched input range: [{img_torch_l.min().item():.4g}, "
              f"{img_torch_l.max().item():.4g}]")

    # Create network
    print(f"\n🏗️ Creating Skip network...")
    input_depth = 32
    pad = 'reflection'

    net = get_net(
        input_depth, 'skip', pad,
        n_channels=n_channels,
        skip_n33d=128,
        skip_n33u=128,
        skip_n11=4,
        num_scales=5,
        upsample_mode='bilinear'
    ).to(device)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"   Network parameters: {total_params:,}")

    # Initialize noise
    height = img_np.shape[1]
    width = img_np.shape[2]
    net_input = get_noise(input_depth, 'noise', (height, width)).to(device)
    net_input_saved = net_input.detach().clone()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    mse = nn.MSELoss()

    # Tracking
    exp_weight = 0.99
    out_avg = None

    # Best output selected by ES-WMV (windowed moving variance of the raw
    # outputs, arXiv:2112.06074): its minimum tracks the transition from
    # signal learning to noise fitting. PSNR vs the noisy input is only a
    # fit indicator: it grows monotonically with overfitting, so it must
    # not drive the selection.
    wmv = ESWMV(window_size=es_window, patience=es_patience,
                min_start=es_min_start)
    best_out = None
    best_iter = 0
    psnr_iters = []
    psnr_stretched_history = []
    loss_history = []
    loss_gat_history = []
    loss_raw_history = []

    # Backtracking (recovery from sudden divergence)
    last_state = None
    psnr_stretched_last = 0

    # Create intermediate dir
    if save_intermediate and save_path:
        intermediate_dir = Path(save_path).parent / "intermediate_v4_mse"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print(f"\n🎯 Starting optimization ({num_iter} iterations)...")
    if use_gat:
        print(f"   DIP on: RAW image")
        print(f"   Loss: MSE in GAT-transformed space (variance stabilized)")
        print(f"   PSNR: Calculated on STRETCHED image (visual quality)")
        print(f"   → GAT ensures physically correct noise handling ✅")
    else:
        print(f"   GAT disabled - using standard MSE for comparison")

    # Store img_np for PSNR calculation
    img_np_for_psnr = img_np[0] if img_np.shape[0] == 1 else img_np

    # Optimization loop
    for i in range(num_iter):
        optimizer.zero_grad()

        # Regularization noise
        if reg_noise_std > 0:
            net_input = net_input_saved + (torch.randn_like(net_input_saved) * reg_noise_std)

        # Forward pass (network outputs RAW image)
        out = net(net_input)

        # Exponential smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        # LOSS CALCULATION - KEY DIFFERENCE IN V4
        out_transformed = None
        if loss_space == 'gat':
            # Transform both output and input with GAT, in ADU space
            out_adu = out * range_adu + offset_adu
            out_transformed = generalized_anscombe_transform(out_adu, gain, read_noise)
            # Loss in GAT-transformed space (variance stabilized)
            loss = mse(out_transformed, img_torch_gat)

            # Also track loss in RAW space for comparison
            loss_raw = mse(out, img_torch)
            loss_gat_history.append(loss.item())
            loss_raw_history.append(loss_raw.item())
        elif loss_space == 'asinh':
            # Loss in aggressively stretched space: faint structure gets a
            # far larger share of the gradient than under raw MSE
            out_transformed = torch.asinh(out / asinh_beta)
            loss = mse(out_transformed, img_torch_l)
            loss_raw = mse(out, img_torch)
            loss_gat_history.append(loss.item())
            loss_raw_history.append(loss_raw.item())
        else:
            # Standard MSE in RAW space (V3 style)
            loss = mse(out, img_torch)

        loss.backward()
        # Chronic loss explosions (dozens of backtracking episodes on bright
        # compact content) are gradient blow-ups: clipping the global norm
        # keeps the trajectory stable where backtracking only thrashes.
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()

        # ES-WMV: a new variance minimum marks the current best iteration,
        # snapshot the smoothed output. In 'raw' space the variance is
        # dominated by the bright content (amplitude squared); in the loss
        # space (GAT or asinh) noise is (approximately) equalized across
        # brightness levels, so background, faint and bright structure
        # weigh comparably in the valley detection.
        use_transformed = wmv_space in ('gat', 'loss') and out_transformed is not None
        wmv_input = out_transformed if use_transformed else out
        if wmv.update(wmv_input, i):
            best_out = out_avg.clone()
            best_iter = i

        loss_history.append(loss.item())

        # PSNR vs noisy input (fit indicator only), every psnr_interval iters
        if i % psnr_interval == 0 or i == num_iter - 1:
            out_np = out_avg.detach().cpu().squeeze().numpy()
            out_np = np.clip(out_np, 0, 1)
            psnr_stretched = calculate_psnr_stretched(
                img_np_for_psnr,
                out_np,
                beta=psnr_stretch_beta
            )
            psnr_iters.append(i)
            psnr_stretched_history.append(psnr_stretched)

        # Backtracking on sudden divergence. Optimizer state is restored
        # together with the weights, otherwise Adam's moments immediately
        # push the network back to the diverged point.
        if i % 100 == 0 and i > 0:
            if psnr_stretched - psnr_stretched_last < -3:
                if show_progress:
                    print(f"   ⚠️ Iter {i}: Backtracking (PSNR_stretched drop: {psnr_stretched - psnr_stretched_last:.2f} dB)")

                if last_state is not None:
                    net.load_state_dict(last_state['net'])
                    optimizer.load_state_dict(last_state['opt'])
                    out_avg = last_state['out_avg'].clone()
                    # The selection must rewind with the network: the WMV
                    # ring holds outputs from the abandoned trajectory
                    wmv.set_state(last_state['wmv'])
                    best_out = (None if last_state['best_out'] is None
                                else last_state['best_out'].clone())
                    best_iter = last_state['best_iter']
            else:
                last_state = {
                    'net': copy.deepcopy(net.state_dict()),
                    'opt': copy.deepcopy(optimizer.state_dict()),
                    'out_avg': out_avg.clone(),
                    'wmv': wmv.get_state(),
                    'best_out': None if best_out is None else best_out.clone(),
                    'best_iter': best_iter,
                }
                psnr_stretched_last = psnr_stretched

        # Save intermediate
        if save_intermediate and save_path and (i + 1) % intermediate_interval == 0:
            intermediate_out = out_avg.detach().cpu().squeeze().numpy()
            intermediate_out = np.clip(intermediate_out, 0, 1)

            if len(intermediate_out.shape) == 2:
                intermediate_out = intermediate_out[..., np.newaxis]

            # Snapshots use the same format as the requested output
            intermediate_path = intermediate_dir / f"iter_{i+1:04d}{Path(save_path).suffix}"

            AstroImageHandler.save_astro_image(
                intermediate_out,
                metadata,
                intermediate_path,
                bit_depth=32
            )

        # Progress
        if show_progress and (i == 0 or (i + 1) % 100 == 0):
            if loss_space in ('gat', 'asinh'):
                print(f"   Iteration {i+1}/{num_iter} - Loss_{loss_space.upper()}: {loss.item():.6f} - Loss_RAW: {loss_raw.item():.6f} - PSNR_stretched: {psnr_stretched:.2f} dB")
            else:
                print(f"   Iteration {i+1}/{num_iter} - Loss: {loss.item():.6f} - PSNR_stretched: {psnr_stretched:.2f} dB")

        # Early stopping: variance minimum stagnant for es_patience iterations
        if early_stop and wmv.should_stop(i):
            if show_progress:
                print(f"   🛑 Early stop at iter {i}: WMV minimum (iter {wmv.min_iteration}) "
                      f"stagnant for {es_patience} iterations")
            break

    total_time = time.time() - start_time
    final_iteration = i + 1
    stopped_early = final_iteration < num_iter

    if best_out is None:
        # WMV window never filled (num_iter < es_window): use the last output
        best_out = out_avg.clone()
        best_iter = i

    # Final output (RAW, no stretch), selected at the WMV variance minimum
    final_out = best_out.detach().cpu().squeeze().numpy()
    final_out = np.clip(final_out, 0, 1)

    # PSNR vs noisy input of the selected output (fit indicator, not quality)
    best_psnr_stretched = calculate_psnr_stretched(
        img_np_for_psnr, final_out, beta=psnr_stretch_beta
    )

    if len(final_out.shape) == 2:
        final_out = final_out[..., np.newaxis]

    # Save
    if save_path:
        save_path = Path(save_path)

        AstroImageHandler.save_astro_image(
            final_out,
            metadata,
            save_path,
            bit_depth=32
        )

        print(f"\n💾 Denoised image saved to: {save_path}")
        print(f"   ✅ RAW float32 (NO stretch applied)")
        print(f"   ✅ Dynamics preserved: [{metadata.get('original_min', 0):.6g}, {metadata.get('original_max', 65535):.6g}]")
        if use_gat:
            print(f"   ✅ GAT variance stabilization applied during optimization")

        write_run_sidecar(save_path, {
            'pipeline': 'dip-astro-v4-mse',
            'input': str(image_path),
            'output': str(save_path),
            'best_iteration': best_iter,
            'final_iteration': final_iteration,
            'stopped_early': stopped_early,
            'seed': seed,
            'total_time_s': total_time,
            'time_per_iter_s': total_time / final_iteration,
            'device': str(device),
            'psnr_stretched_fit_indicator_db': best_psnr_stretched,
            'reinjected_pixels': metadata.get('reinjected_count', 0),
            'original_range': [metadata.get('original_min'), metadata.get('original_max')],
            'params': {
                'num_iter': num_iter,
                'learning_rate': learning_rate,
                'reg_noise_std': reg_noise_std,
                'percentiles': list(percentiles),
                'psnr_stretch_beta': psnr_stretch_beta,
                'psnr_interval': psnr_interval,
                'early_stop': early_stop,
                'es_window': es_window,
                'es_patience': es_patience,
                'es_min_start': es_min_start,
                'use_gat': use_gat,
                'loss_space': loss_space,
                'asinh_beta': asinh_beta if loss_space == 'asinh' else None,
                'gain': gain,
                'read_noise': read_noise,
                'adu_scale': adu_scale if use_gat else None,
                'wmv_space': wmv_space,
                'grad_clip': grad_clip,
            },
        })

        # Save comparison plots
        try:
            import matplotlib.pyplot as plt

            img_2d = img_np_for_psnr
            final_2d = final_out.squeeze()

            # Stretch for visualization: STF autostretch, parameters from the
            # noisy original so both panels are directly comparable
            img_vis = stf_autostretch(img_2d, ref=img_2d)
            final_vis = stf_autostretch(final_2d, ref=img_2d)

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Row 1: RAW
            axes[0, 0].imshow(img_2d, cmap='gray', vmin=0, vmax=1)
            axes[0, 0].set_title('Original RAW')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(final_2d, cmap='gray', vmin=0, vmax=1)
            axes[0, 1].set_title(f'Denoised RAW (iter {best_iter})')
            axes[0, 1].axis('off')

            diff_raw = np.abs(img_2d - final_2d)
            im0 = axes[0, 2].imshow(diff_raw, cmap='hot', vmin=0, vmax=0.05)
            axes[0, 2].set_title('Noise (RAW)')
            axes[0, 2].axis('off')
            plt.colorbar(im0, ax=axes[0, 2], fraction=0.046)

            # Row 2: STRETCHED
            axes[1, 0].imshow(img_vis, cmap='gray', vmin=0, vmax=1)
            axes[1, 0].set_title('Original STRETCHED')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(final_vis, cmap='gray', vmin=0, vmax=1)
            axes[1, 1].set_title(f'Denoised STRETCHED (PSNR: {best_psnr_stretched:.1f} dB)')
            axes[1, 1].axis('off')

            diff_vis = np.abs(img_vis - final_vis)
            im1 = axes[1, 2].imshow(diff_vis, cmap='hot', vmin=0, vmax=0.2)
            axes[1, 2].set_title('Noise STRETCHED')
            axes[1, 2].axis('off')
            plt.colorbar(im1, ax=axes[1, 2], fraction=0.046)

            plt.suptitle(f'V4 MSE {"with GAT" if use_gat else "without GAT"}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            comparison_path = save_path.parent / f"{save_path.stem}_comparison.png"
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"   Comparison plot: {comparison_path}")

            # Convergence plots
            if use_gat:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                # Loss in GAT space
                axes[0, 0].plot(loss_gat_history, 'b-', alpha=0.7)
                axes[0, 0].set_xlabel('Iteration')
                axes[0, 0].set_ylabel('MSE Loss (GAT space)')
                axes[0, 0].set_title('Loss in GAT-Stabilized Space')
                axes[0, 0].set_yscale('log')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].axvline(x=best_iter, color='r', linestyle='--', label=f'Best (iter {best_iter})')
                axes[0, 0].legend()

                # Loss in RAW space
                axes[0, 1].plot(loss_raw_history, 'g-', alpha=0.7)
                axes[0, 1].set_xlabel('Iteration')
                axes[0, 1].set_ylabel('MSE Loss (RAW space)')
                axes[0, 1].set_title('Loss in RAW Space (for comparison)')
                axes[0, 1].set_yscale('log')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].axvline(x=best_iter, color='r', linestyle='--')

                # PSNR stretched (fit indicator)
                axes[1, 0].plot(psnr_iters, psnr_stretched_history, 'purple', alpha=0.7)
                axes[1, 0].set_xlabel('Iteration')
                axes[1, 0].set_ylabel('PSNR (dB)')
                axes[1, 0].set_title('PSNR vs Noisy Input (Fit Indicator)')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].axvline(x=best_iter, color='r', linestyle='--')
                axes[1, 0].legend(['PSNR stretched', f'Best (iter {best_iter})'])

                # ES-WMV variance (selection metric)
                if wmv.variance_history:
                    wmv_x, wmv_y = zip(*wmv.variance_history)
                    axes[1, 1].plot(wmv_x, wmv_y, 'm-', alpha=0.7)
                    axes[1, 1].axvline(x=best_iter, color='r', linestyle='--',
                                       label=f'WMV min (iter {best_iter})')
                    axes[1, 1].legend()
                axes[1, 1].set_xlabel('Iteration')
                axes[1, 1].set_ylabel('Windowed moving variance')
                axes[1, 1].set_title('ES-WMV (Selection Metric)')
                axes[1, 1].set_yscale('log')
                axes[1, 1].grid(True, alpha=0.3)

            else:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5))

                # Loss
                ax1.plot(loss_history, 'b-', alpha=0.7)
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel(f'MSE Loss ({loss_space})')
                ax1.set_title(f'Loss ({loss_space} space)')
                ax1.set_yscale('log')
                ax1.grid(True, alpha=0.3)
                ax1.axvline(x=best_iter, color='r', linestyle='--', label=f'Best (iter {best_iter})')
                ax1.legend()

                # PSNR (fit indicator)
                ax2.plot(psnr_iters, psnr_stretched_history, 'g-', alpha=0.7)
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('PSNR (dB)')
                ax2.set_title('PSNR vs Noisy Input (Fit Indicator)')
                ax2.grid(True, alpha=0.3)
                ax2.axvline(x=best_iter, color='r', linestyle='--', label=f'Best (iter {best_iter})')
                ax2.legend()

                # ES-WMV variance (selection metric)
                if wmv.variance_history:
                    wmv_x, wmv_y = zip(*wmv.variance_history)
                    ax3.plot(wmv_x, wmv_y, 'm-', alpha=0.7)
                    ax3.axvline(x=best_iter, color='r', linestyle='--',
                                label=f'WMV min (iter {best_iter})')
                    ax3.legend()
                ax3.set_xlabel('Iteration')
                ax3.set_ylabel('Windowed moving variance')
                ax3.set_title('ES-WMV (Selection Metric)')
                ax3.set_yscale('log')
                ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            convergence_path = save_path.parent / f"{save_path.stem}_convergence.png"
            plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"   Convergence plot: {convergence_path}")

        except ImportError:
            pass

    results = {
        'denoised_image': final_out,
        'best_psnr_stretched': best_psnr_stretched,
        'best_iteration': best_iter,
        'total_time': total_time,
        'time_per_iter': total_time / final_iteration,
        'device': str(device),
        'metadata': metadata,
        'final_iteration': final_iteration,
        'stopped_early': stopped_early,
        'seed': seed,
        'psnr_iterations': psnr_iters,
        'psnr_stretched_history': psnr_stretched_history,
        'loss_history': loss_history,
        'loss_gat_history': loss_gat_history if use_gat else None,
        'loss_raw_history': loss_raw_history if use_gat else None,
        'wmv_variance_history': wmv.variance_history,
        'sensor_params': {
            'gain': gain,                # e-/ADU, as used by the GAT
            'read_noise': read_noise,    # ADU, as used by the GAT
            'adu_scale': adu_scale,      # file units -> ADU factor
            'offset_adu': offset_adu,    # p_low in ADU
            'range_adu': range_adu,      # percentile window width in ADU
            'origins': sensor_origins,   # provenance of gain/read_noise
        } if use_gat else None,
        'use_gat': use_gat
    }

    print(f"\n✅ V4 MSE Denoising complete!")
    print(f"📊 Final statistics:")
    print(f"   Best iteration (ES-WMV): {best_iter}")
    print(f"   PSNR vs input (fit indicator): {best_psnr_stretched:.2f} dB")
    print(f"   Total time: {total_time:.1f} seconds")
    print(f"   Final iteration: {final_iteration}" + (" (early stop)" if stopped_early else ""))

    if 'original_min' in metadata:
        print(f"   Dynamic range preserved: [{metadata['original_min']:.6g}, {metadata['original_max']:.6g}]")

    print(f"\n💡 V4 MSE METHOD:")
    print(f"   ✅ DIP applied on RAW image (no stretch)")
    if use_gat:
        print(f"   ✅ Loss = MSE(T(output), T(input)) with GAT stabilization in ADU")
        print(f"   ✅ Poisson-Gaussian noise correctly handled")
        print(f"   ✅ Sensor parameters: g={gain} e-/ADU, σ_r={read_noise:.3f} ADU, "
              f"ADU scale {adu_scale:.6g}")
    else:
        print(f"   ⚠️ GAT disabled - using standard MSE")
    print(f"   ✅ PSNR calculated in stretched space for visual guidance")
    print(f"   ✅ Output is RAW with preserved photometry")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='V4 MSE: DIP with GAT variance stabilization')
    parser.add_argument('image', help='Path to input astronomical image')
    parser.add_argument('-o', '--output', help='Output path')
    parser.add_argument('-i', '--iterations', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--reg-noise', type=float, default=0.04)
    parser.add_argument('--psnr-beta', type=float, default=0.05,
                        help='Beta for PSNR stretch (visual guidance)')
    parser.add_argument('--percentiles', nargs=2, type=float, default=[0.5, 99.5])
    parser.add_argument('--save-intermediate', action='store_true')
    parser.add_argument('--psnr-interval', type=int, default=10,
                        help='Compute the PSNR fit indicator every N iterations')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='Run all iterations (ES-WMV still selects the best output)')
    parser.add_argument('--es-window', type=int, default=100,
                        help='ES-WMV window size (arXiv:2112.06074)')
    parser.add_argument('--es-patience', type=int, default=1000,
                        help='Stop when the WMV minimum stagnates this many iterations')
    parser.add_argument('--es-min-start', type=int, default=0,
                        help='Accept no WMV minimum before this iteration '
                             '(guards against the warmup false dip; use '
                             '~1500 on structure-rich fields)')

    # V4 specific: sensor parameters
    parser.add_argument('--gain', type=float, default=None,
                        help='Camera gain in e-/ADU (for stacks: gain × number of frames)')
    parser.add_argument('--read-noise', type=float, default=None,
                        help='Read noise in ADU (if not provided, will estimate)')
    parser.add_argument('--adu-scale', type=float, default=None,
                        help='File-value to ADU factor (default auto: 65535 for '
                             'pre-normalized float files, 1 for integer FITS)')
    parser.add_argument('--no-gat', action='store_true',
                        help='Disable GAT (for comparison with V3)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: drawn from the OS and logged)')
    parser.add_argument('--norm-range', nargs=2, type=float, default=None,
                        metavar=('LOW', 'HIGH'),
                        help='Absolute normalization window in file units '
                             '(overrides --percentiles; used by tiled processing)')
    parser.add_argument('--wmv-space', choices=['raw', 'gat', 'loss'], default='raw',
                        help='Space for the ES-WMV selection variance: raw '
                             '(historical, bright-dominated) or gat/loss (the '
                             'loss-transformed space, noise-balanced)')
    parser.add_argument('--loss-space', choices=['gat', 'asinh', 'raw'], default=None,
                        help='Space of the training loss: gat (variance-'
                             'stabilized, rigorous), asinh (aggressive faint-'
                             'structure weighting), raw (plain MSE). Default: '
                             'gat, or raw with --no-gat')
    parser.add_argument('--asinh-beta', type=float, default=None,
                        help='Beta of the asinh loss stretch in normalized '
                             'units (default: 4 x background sigma)')
    parser.add_argument('--grad-clip', type=float, default=None,
                        help='Clip the global gradient norm (stabilizes '
                             'chronic loss explosions on bright compact '
                             'content; try 1.0)')

    args = parser.parse_args()

    if not args.output:
        input_path = Path(args.image)
        suffix = "_v4_mse" if not args.no_gat else "_v4_no_gat"
        args.output = input_path.parent / f"{input_path.stem}_denoised{suffix}.fit"

    results = denoise_astro_image_v4_mse(
        args.image,
        num_iter=args.iterations,
        learning_rate=args.lr,
        reg_noise_std=args.reg_noise,
        save_path=args.output,
        psnr_stretch_beta=args.psnr_beta,
        percentiles=tuple(args.percentiles),
        save_intermediate=args.save_intermediate,
        psnr_interval=args.psnr_interval,
        early_stop=not args.no_early_stop,
        es_window=args.es_window,
        es_patience=args.es_patience,
        es_min_start=args.es_min_start,
        gain=args.gain,
        read_noise=args.read_noise,
        use_gat=not args.no_gat,
        adu_scale=args.adu_scale,
        seed=args.seed,
        norm_range=tuple(args.norm_range) if args.norm_range else None,
        wmv_space=args.wmv_space,
        loss_space=args.loss_space,
        asinh_beta=args.asinh_beta,
        grad_clip=args.grad_clip
    )

    print(f"\n🎉 Results saved to: {args.output}")