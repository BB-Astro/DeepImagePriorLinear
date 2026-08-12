#!/usr/bin/env python3
"""
Deep Image Prior - Tiled driver for images too large for GPU memory.

One DIP training step needs ~8 GiB of GPU memory per megapixel (measured on
MPS, skip network 128ch/5 scales), so a 64 GB Apple Silicon machine tops out
around 5 Mpx per run. This driver splits the image into an overlapping grid,
runs the V3 pipeline on each tile, and blends the results.

Tile consistency rests on three mechanisms:
1. The percentile normalization window is computed ONCE on the full image and
   imposed on every tile (norm_range): all tiles share the same dynamics.
2. Every tile runs with the same seed: same network init, same input noise.
3. Overlap zones are blended with a linear crossfade, and the pre-blend
   disagreement between adjacent tiles is measured and reported in the
   sidecar (overlap_rms_diff, overlap_max_diff) so consistency is verified,
   not assumed.

Bright pixels above the high percentile are reinjected verbatim per tile
(both tiles hold identical original values in the overlaps, so blending
preserves them bit for bit).

Author: Ben & Claude
Date: 2026-07-31
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from denoise_astro_v3 import denoise_astro_image_v3
from denoise_astro_v4_mse import denoise_astro_image_v4_mse
from utils.astro_adapter import AstroImageHandler, HAS_ASTROPY
from utils.sidecar import write_run_sidecar
from utils.validation import validate_tiled_args

if HAS_ASTROPY:
    from astropy.io import fits


def tile_grid(H: int, W: int, ny: int, nx: int, overlap: int) -> List[Dict]:
    """Overlapping grid: core regions partition the image, extended windows
    add `overlap` pixels on interior edges."""
    ys = np.linspace(0, H, ny + 1).astype(int)
    xs = np.linspace(0, W, nx + 1).astype(int)
    tiles = []
    for i in range(ny):
        for j in range(nx):
            y0c, y1c = int(ys[i]), int(ys[i + 1])
            x0c, x1c = int(xs[j]), int(xs[j + 1])
            y0 = max(0, y0c - overlap) if i > 0 else 0
            y1 = min(H, y1c + overlap) if i < ny - 1 else H
            x0 = max(0, x0c - overlap) if j > 0 else 0
            x1 = min(W, x1c + overlap) if j < nx - 1 else W
            tiles.append({
                'index': (i, j),
                'core': (y0c, y1c, x0c, x1c),
                'ext': (y0, y1, x0, x1),
            })
    return tiles


def crossfade_weights(tile: Dict, overlap: int) -> np.ndarray:
    """Separable weight map: 1 in the core, linear ramp to ~0 across the
    overlap margin on edges that have a neighbor."""
    y0c, y1c, x0c, x1c = tile['core']
    y0, y1, x0, x1 = tile['ext']

    def axis_weights(length, lead, trail):
        w = np.ones(length, dtype=np.float64)
        if lead > 0:
            w[:lead] = np.linspace(0.0, 1.0, lead + 2)[1:-1]
        if trail > 0:
            w[length - trail:] = np.linspace(1.0, 0.0, trail + 2)[1:-1]
        return w

    wy = axis_weights(y1 - y0, y0c - y0, y1 - y1c)
    wx = axis_weights(x1 - x0, x0c - x0, x1 - x1c)
    return np.outer(wy, wx)


def denoise_astro_tiled(
    image_path: str,
    save_path: str,
    tiles: str = '2x2',
    overlap: int = 128,
    num_iter: int = 12000,
    seed: Optional[int] = None,
    percentiles: Tuple[float, float] = (0.5, 99.5),
    engine: str = 'v3',
    **engine_kwargs,
) -> Dict:
    if not HAS_ASTROPY:
        raise ImportError("Tiled processing writes FITS tiles and output: "
                          "astropy is required (pip install astropy)")

    image_path = Path(image_path)
    save_path = Path(save_path)
    if save_path.suffix.lower() not in ('.fit', '.fits'):
        raise ValueError(f"Tiled output must be FITS (.fit/.fits), got "
                         f"'{save_path.suffix}'. The assembled image is "
                         f"written directly as FITS.")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tiles_dir = save_path.parent / f"{save_path.stem}_tiles"
    tiles_dir.mkdir(exist_ok=True)

    ny, nx = (int(v) for v in tiles.lower().split('x'))
    if seed is None:
        seed = int.from_bytes(os.urandom(4), 'little')

    print(f"🧩 Tiled DIP: {ny}x{nx} tiles, overlap {overlap} px, seed {seed}")

    full, metadata = AstroImageHandler.load_raw(image_path)
    H, W = full.shape
    validate_tiled_args(H, W, ny, nx, overlap)
    p_low, p_high = np.percentile(full, percentiles)
    if p_high - p_low < 1e-12:
        raise ValueError("Degenerate global percentile window")
    print(f"   Image {W}x{H} ({H*W/1e6:.2f} Mpx), "
          f"global window [{p_low:.6g}, {p_high:.6g}]")

    grid = tile_grid(H, W, ny, nx, overlap)
    start = time.time()

    denoised_tiles = []
    tile_reports = []
    for k, tile in enumerate(grid):
        y0, y1, x0, x1 = tile['ext']
        i, j = tile['index']
        tile_path = tiles_dir / f"tile_{i}{j}.fit"
        out_path = tiles_dir / f"tile_{i}{j}_denoised.fit"

        # The skip network has 5 downsampling stages: tile dimensions must be
        # multiples of 32. Reflection-pad the bottom/right edge, crop it back
        # after denoising.
        th, tw = y1 - y0, x1 - x0
        pad_h = (-th) % 32
        pad_w = (-tw) % 32
        tdata = full[y0:y1, x0:x1]
        if pad_h or pad_w:
            tdata = np.pad(tdata, ((0, pad_h), (0, pad_w)), mode='reflect')

        hdu = fits.PrimaryHDU(tdata)
        hdu.header['TILEY0'] = y0
        hdu.header['TILEX0'] = x0
        hdu.writeto(tile_path, overwrite=True)

        print(f"\n🧩 Tile {k+1}/{len(grid)} [{i},{j}]: "
              f"y[{y0}:{y1}] x[{x0}:{x1}] ({th*tw/1e6:.2f} Mpx, "
              f"pad +{pad_h}/+{pad_w})")
        denoise_fn = denoise_astro_image_v3 if engine == 'v3' else denoise_astro_image_v4_mse
        results = denoise_fn(
            str(tile_path),
            num_iter=num_iter,
            save_path=str(out_path),
            seed=seed,
            norm_range=(float(p_low), float(p_high)),
            **engine_kwargs,
        )
        den = np.asarray(fits.getdata(out_path), dtype=np.float64)[:th, :tw]
        denoised_tiles.append(den)
        tile_reports.append({
            'tile': [i, j],
            'window': [y0, y1, x0, x1],
            'best_iteration': results['best_iteration'],
            'final_iteration': results['final_iteration'],
            'stopped_early': results['stopped_early'],
            'time_s': results['total_time'],
        })

    # Overlap disagreement BEFORE blending: the objective consistency check.
    overlap_stats = []
    for a in range(len(grid)):
        for b in range(a + 1, len(grid)):
            ya0, ya1, xa0, xa1 = grid[a]['ext']
            yb0, yb1, xb0, xb1 = grid[b]['ext']
            oy0, oy1 = max(ya0, yb0), min(ya1, yb1)
            ox0, ox1 = max(xa0, xb0), min(xa1, xb1)
            if oy0 >= oy1 or ox0 >= ox1:
                continue
            da = denoised_tiles[a][oy0 - ya0:oy1 - ya0, ox0 - xa0:ox1 - xa0]
            db = denoised_tiles[b][oy0 - yb0:oy1 - yb0, ox0 - xb0:ox1 - xb0]
            diff = da - db
            overlap_stats.append({
                'tiles': [grid[a]['index'], grid[b]['index']],
                'pixels': int(diff.size),
                'rms_diff': float(np.sqrt(np.mean(diff ** 2))),
                'max_abs_diff': float(np.abs(diff).max()),
            })

    # Blend
    acc = np.zeros((H, W), dtype=np.float64)
    wacc = np.zeros((H, W), dtype=np.float64)
    for tile, den in zip(grid, denoised_tiles):
        y0, y1, x0, x1 = tile['ext']
        w = crossfade_weights(tile, overlap)
        acc[y0:y1, x0:x1] += den * w
        wacc[y0:y1, x0:x1] += w
    assembled = (acc / wacc).astype(np.float32)

    # Reinjection safety net: blending preserves reinjected pixels only if
    # every overlapping tile reinjected them identically (it does, values are
    # verbatim). Enforce exactness anyway from the global mask.
    reinj_mask = full > p_high
    assembled[reinj_mask] = full[reinj_mask]

    total_time = time.time() - start

    hdu = fits.PrimaryHDU(assembled)
    # Source header first (same exclusion set as the adapter: structural
    # keys, BSCALE/BZERO which would corrupt a float32 output at read time,
    # checksums, and pipeline keys), then our cards so they always win.
    exclude = AstroImageHandler.FITS_HEADER_EXCLUDE | {'NTILESY', 'NTILESX', 'TILEOVLP'}
    src_header = metadata.get('header')
    if isinstance(src_header, dict):
        for key, value in src_header.items():
            if str(key).upper() not in exclude:
                try:
                    hdu.header[key] = value
                except Exception:
                    pass
    hdu.header['HISTORY'] = 'Deep Image Prior denoising applied (tiled)'
    hdu.header['ORIGMIN'] = (float(p_low), 'Global normalization window low')
    hdu.header['ORIGMAX'] = (float(p_high), 'Global normalization window high')
    hdu.header['PERCLOW'] = (percentiles[0], 'Low percentile used')
    hdu.header['PERCHIGH'] = (percentiles[1], 'High percentile used')
    hdu.header['NREINJ'] = (int(reinj_mask.sum()),
                            'Pixels above high percentile reinjected verbatim')
    hdu.header['NTILESY'] = ny
    hdu.header['NTILESX'] = nx
    hdu.header['TILEOVLP'] = overlap
    hdu.writeto(save_path, overwrite=True)
    print(f"\n💾 Assembled image saved: {save_path}")

    worst = max((s['rms_diff'] for s in overlap_stats), default=0.0)
    print(f"   Overlap consistency: worst RMS diff {worst:.3g} "
          f"(window width {p_high - p_low:.3g})")

    sidecar = {
        'pipeline': f'dip-astro-{engine}-tiled',
        'input': str(image_path),
        'output': str(save_path),
        'seed': seed,
        'tiles': f'{ny}x{nx}',
        'overlap_px': overlap,
        'global_window': [float(p_low), float(p_high)],
        'reinjected_pixels': int(reinj_mask.sum()),
        'total_time_s': total_time,
        'tile_runs': tile_reports,
        'overlap_consistency': overlap_stats,
        'params': {'num_iter': num_iter, 'percentiles': list(percentiles),
                   'engine': engine,
                   **{k: v for k, v in engine_kwargs.items()
                      if isinstance(v, (int, float, bool, str))}},
    }
    write_run_sidecar(save_path, sidecar)
    return sidecar


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tiled DIP denoising for images too large for GPU memory')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-o', '--output', help='Output path')
    parser.add_argument('-i', '--iterations', type=int, default=12000)
    parser.add_argument('--tiles', default='2x2',
                        help='Grid, e.g. 2x2, 3x2 (rows x cols)')
    parser.add_argument('--overlap', type=int, default=128,
                        help='Overlap margin in pixels on interior edges')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--percentiles', nargs=2, type=float, default=[0.5, 99.5])
    parser.add_argument('--es-patience', type=int, default=1000)
    parser.add_argument('--es-min-start', type=int, default=1500,
                        help='No WMV minimum before this iteration (warmup '
                             'false-dip guard; tiled default 1500)')
    parser.add_argument('--engine', choices=['v3', 'v4'], default='v3',
                        help='v3 = MSE loss, v4 = GAT variance-stabilized loss')
    parser.add_argument('--wmv-space', choices=['raw', 'gat', 'loss'], default='raw',
                        help='ES-WMV selection space (v4 engine only)')

    args = parser.parse_args()
    if not args.output:
        input_path = Path(args.image)
        args.output = input_path.parent / f"{input_path.stem}_denoised_tiled.fit"

    extra = {}
    if args.engine == 'v4':
        extra['wmv_space'] = args.wmv_space

    denoise_astro_tiled(
        args.image,
        str(args.output),
        tiles=args.tiles,
        overlap=args.overlap,
        num_iter=args.iterations,
        seed=args.seed,
        percentiles=tuple(args.percentiles),
        es_patience=args.es_patience,
        es_min_start=args.es_min_start,
        engine=args.engine,
        **extra,
    )
