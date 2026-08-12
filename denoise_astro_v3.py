#!/usr/bin/env python3
"""
Deep Image Prior - Astronomical Image Denoising V3
SOLUTION FINALE : DIP sur RAW + PSNR stretché pour guidage

Idée clé de Ben :
- DIP travaille sur image RAW (pas de stretch)
- On stretch SEULEMENT pour calculer le PSNR
- Le PSNR stretché guide l'optimization (early stopping)
- Résultat final = RAW avec dynamique préservée

Author: Ben & Claude
Date: 2025-11-10
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
from utils.validation import validate_common_args


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def stretch_arcsinh(img, beta=0.05):
    """Arcsinh stretch for PSNR calculation ONLY"""
    stretched = np.arcsinh(img / beta) / np.arcsinh(1.0 / beta)
    return np.clip(stretched, 0, 1)


def stf_autostretch(img, ref=None, target=0.25):
    """PixInsight-style STF autostretch for the diagnostic plots.

    Shadows at median - 2.8 MAD, midtones transfer function bringing the
    median to `target`. Stretch parameters come from `ref` (the noisy
    original) so that side-by-side panels share the exact same stretch.
    The arcsinh stretch used for the PSNR indicator is far too soft to
    display linear astro images: it is a metric, not a visualization.
    """
    if ref is None:
        ref = img
    med = float(np.median(ref))
    mad = float(np.median(np.abs(ref - med))) * 1.4826
    c0 = med - 2.8 * mad

    def mtf(x, m):
        return ((m - 1.0) * x) / (((2.0 * m - 1.0) * x) - m)

    x = np.clip((img - c0) / (1.0 - c0), 0, 1)
    pivot = (med - c0) / (1.0 - c0)
    if pivot <= 0 or pivot >= 1:
        return x
    m = (pivot * (target - 1.0)) / (pivot * (2.0 * target - 1.0) - target)
    return np.clip(mtf(x, m), 0, 1)


def calculate_psnr_stretched(img1, img2, beta=0.05):
    """
    Calculate PSNR in STRETCHED space (for guidance)

    This separates signal from noise for better quality metric
    """
    # Stretch both images
    img1_stretched = stretch_arcsinh(img1, beta)
    img2_stretched = stretch_arcsinh(img2, beta)

    # Calculate PSNR in stretched space
    mse = np.mean((img1_stretched - img2_stretched) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


def denoise_astro_image_v3(
    image_path: str,
    num_iter: int = 3000,
    learning_rate: float = 0.01,
    reg_noise_std: float = 0.04,
    save_path: Optional[str] = None,
    show_progress: bool = True,
    percentiles: Tuple[float, float] = (0.5, 99.5),
    preserve_dynamics: bool = True,
    psnr_stretch_beta: float = 0.05,  # Beta for PSNR calculation ONLY
    save_intermediate: bool = False,
    intermediate_interval: int = 500,
    psnr_interval: int = 10,
    early_stop: bool = True,
    es_window: int = 100,
    es_patience: int = 1000,
    es_min_start: int = 0,
    seed: Optional[int] = None,
    norm_range: Optional[Tuple[float, float]] = None,
    exp_weight: float = 0.99
) -> Dict:
    """
    V3: DIP on RAW image, PSNR calculated in stretched space for guidance

    KEY INNOVATION (Ben's idea):
    - DIP works on RAW image (no stretching of data)
    - PSNR is calculated in STRETCHED space (separates signal/noise)
    - Use stretched PSNR for early stopping guidance
    - Best iteration tracked by stretched PSNR max
    - Output is RAW with preserved dynamics
    """

    validate_common_args(num_iter, learning_rate, reg_noise_std, percentiles,
                         psnr_interval, es_window, es_patience, es_min_start,
                         exp_weight, norm_range)

    # Seed everything that draws random numbers: network init and input noise
    # (torch), regularization noise (torch), any numpy draws. The ES-WMV pixel
    # subsample keeps its own fixed seed so the variance curves of two runs
    # stay comparable regardless of the run seed. Note: MPS does not guarantee
    # bitwise-identical results across runs for every op; the trajectory is
    # reproducible in practice but small numerical drift is possible.
    if seed is None:
        seed = int.from_bytes(os.urandom(4), 'little')
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    print(f"🚀 Using device: {device}")
    print(f"🎲 Seed: {seed}")
    print(f"\n💡 V3 WORKFLOW:")
    print(f"   1. DIP on RAW image (no stretch)")
    print(f"   2. Best iteration selected by ES-WMV (output variance minimum)")
    print(f"   3. PSNR in STRETCHED space (β={psnr_stretch_beta}) logged as fit indicator")
    print(f"   4. Output = RAW float32 with dynamics preserved")

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
    print(f"   ✅ Image kept in RAW space (NO stretch applied)")

    # Convert to torch
    if len(img_np.shape) == 2:
        img_np = img_np[np.newaxis, ...]
    elif len(img_np.shape) == 3:
        img_np = img_np.transpose(2, 0, 1)

    img_torch = torch.from_numpy(img_np.astype(np.float32)).unsqueeze(0).to(device)
    n_channels = img_torch.shape[1]

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

    # Tracking. exp_weight controls the EMA of saved outputs: 0.99 averages
    # over ~100 iterations (smoother, the historical default), lower values
    # shorten the window (sharper, closer to the raw output, more grain).
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

    # Backtracking (recovery from sudden divergence)
    last_state = None
    psnr_stretched_last = 0

    # Create intermediate dir
    if save_intermediate and save_path:
        intermediate_dir = Path(save_path).parent / "intermediate_v3"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print(f"\n🎯 Starting optimization ({num_iter} iterations on RAW)...")
    print(f"   DIP on: RAW image")
    print(f"   Selection: ES-WMV window={es_window}, patience={es_patience}, "
          f"early stop {'ON' if early_stop else 'OFF'}")
    print(f"   PSNR (fit indicator) computed every {psnr_interval} iterations")

    # Store img_np for PSNR calculation
    img_np_for_psnr = img_np[0] if img_np.shape[0] == 1 else img_np

    # Optimization loop
    for i in range(num_iter):
        optimizer.zero_grad()

        # Regularization noise
        if reg_noise_std > 0:
            net_input = net_input_saved + (torch.randn_like(net_input_saved) * reg_noise_std)

        # Forward pass (on RAW image)
        out = net(net_input)

        # Exponential smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        # Loss (on RAW image)
        loss = mse(out, img_torch)
        loss.backward()
        optimizer.step()

        # ES-WMV: a new variance minimum marks the current best iteration,
        # snapshot the smoothed output
        if wmv.update(out, i):
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

            intermediate_path = intermediate_dir / f"iter_{i+1:04d}.tif"

            AstroImageHandler.save_astro_image(
                intermediate_out,
                metadata,
                intermediate_path,
                bit_depth=32
            )

        # Progress
        if show_progress and (i == 0 or (i + 1) % 100 == 0):
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

        write_run_sidecar(save_path, {
            'pipeline': 'dip-astro-v3',
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
                'exp_weight': exp_weight,
                'percentiles': list(percentiles),
                'psnr_stretch_beta': psnr_stretch_beta,
                'psnr_interval': psnr_interval,
                'early_stop': early_stop,
                'es_window': es_window,
                'es_patience': es_patience,
                'es_min_start': es_min_start,
            },
        })

        # Save comparison with both metrics
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

            # Row 2: STRETCHED (for visualization)
            axes[1, 0].imshow(img_vis, cmap='gray', vmin=0, vmax=1)
            axes[1, 0].set_title('Original STRETCHED')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(final_vis, cmap='gray', vmin=0, vmax=1)
            axes[1, 1].set_title(f'Denoised STF (PSNR fit: {best_psnr_stretched:.1f} dB)')
            axes[1, 1].axis('off')

            diff_vis = np.abs(img_vis - final_vis)
            im1 = axes[1, 2].imshow(diff_vis, cmap='hot', vmin=0, vmax=0.2)
            axes[1, 2].set_title('Noise STRETCHED')
            axes[1, 2].axis('off')
            plt.colorbar(im1, ax=axes[1, 2], fraction=0.046)

            plt.tight_layout()
            comparison_path = save_path.parent / f"{save_path.stem}_comparison.png"
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"   Comparison plot: {comparison_path}")

            # Convergence
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5))

            # Loss
            ax1.plot(loss_history, 'b-', alpha=0.7)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('MSE Loss (on RAW)')
            ax1.set_title('Loss on RAW Image')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            ax1.axvline(x=best_iter, color='r', linestyle='--', label=f'Best (iter {best_iter})')
            ax1.legend()

            # PSNR stretched (fit indicator)
            ax2.plot(psnr_iters, psnr_stretched_history, 'g-', alpha=0.7)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('PSNR (dB) in STRETCHED space')
            ax2.set_title('PSNR vs Noisy Input (Fit Indicator)')
            ax2.grid(True, alpha=0.3)
            ax2.axvline(x=best_iter, color='r', linestyle='--', label=f'Best (iter {best_iter})')
            ax2.legend()

            # ES-WMV variance (selection metric)
            if wmv.variance_history:
                wmv_x, wmv_y = zip(*wmv.variance_history)
                ax3.plot(wmv_x, wmv_y, 'm-', alpha=0.7)
                ax3.axvline(x=best_iter, color='r', linestyle='--', label=f'WMV min (iter {best_iter})')
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
        'wmv_variance_history': wmv.variance_history
    }

    print(f"\n✅ V3 Denoising complete!")
    print(f"📊 Final statistics:")
    print(f"   Best iteration (ES-WMV): {best_iter}")
    print(f"   PSNR vs input (fit indicator): {best_psnr_stretched:.2f} dB")
    print(f"   Total time: {total_time:.1f} seconds")
    print(f"   Final iteration: {final_iteration}" + (" (early stop)" if stopped_early else ""))

    if 'original_min' in metadata:
        print(f"   Dynamic range preserved: [{metadata['original_min']:.6g}, {metadata['original_max']:.6g}]")

    print(f"\n💡 V3 METHOD:")
    print(f"   ✅ DIP applied on RAW image (no stretch)")
    print(f"   ✅ Best iteration selected by ES-WMV variance minimum")
    print(f"   ✅ Early stopping when the variance minimum stagnates")
    print(f"   ✅ Output is RAW float32 with preserved photometry")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='V3: DIP on RAW, PSNR stretched for guidance')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', help='Output path')
    parser.add_argument('-i', '--iterations', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--reg-noise', type=float, default=0.04)
    parser.add_argument('--psnr-beta', type=float, default=0.05,
                        help='Beta for PSNR stretch (guidance only)')
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
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: drawn from the OS and logged)')
    parser.add_argument('--norm-range', nargs=2, type=float, default=None,
                        metavar=('LOW', 'HIGH'),
                        help='Absolute normalization window in file units '
                             '(overrides --percentiles; used by tiled processing)')
    parser.add_argument('--exp-weight', type=float, default=0.99,
                        help='EMA weight of the saved output (0.99 = ~100-iter '
                             'window; lower = sharper, more grain)')

    args = parser.parse_args()

    if not args.output:
        input_path = Path(args.image)
        # Recommandé : Export FITS (meilleure préservation de la dynamique)
        args.output = input_path.parent / f"{input_path.stem}_denoised_v3.fit"

    results = denoise_astro_image_v3(
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
        seed=args.seed,
        norm_range=tuple(args.norm_range) if args.norm_range else None,
        exp_weight=args.exp_weight
    )

    print(f"\n🎉 Results saved to: {args.output}")