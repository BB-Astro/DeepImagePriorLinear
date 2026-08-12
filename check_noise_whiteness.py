#!/usr/bin/env python3
"""
Background noise whiteness check for astronomical images.

Answers, in seconds, the question worth asking BEFORE hours of DIP: is the
background noise white (DIP will remove most of it) or spatially correlated
(drizzle residual, stripes: DIP will keep most of it, fix the preprocessing
first)?

Method: on the darkest windows of the image, after removing the best-fit
plane (sky gradient), measure the adjacent-pixel decorrelation ratio
    r = std(diff of adjacent pixels) / std
For white noise r = sqrt(2) = 1.414; correlated noise drives it toward 0.
Equivalent lag-1 autocorrelation rho = 1 - r^2/2 is reported too, and the
noise std is decomposed into its white and correlated components
(white = std(diff)/sqrt(2), correlated = sqrt(total^2 - white^2)).

Measured reference points (2026-07-31):
    Arp130 (HST, drizzled)   r = 1.07-1.13  -> strongly correlated
    DIP output on the same   r = 0.22-0.46  -> residual is pure low-frequency

The verdict thresholds are heuristics chosen from those measurements, not
theory: r >= 1.30 white, 1.10-1.30 moderately correlated, < 1.10 strongly
correlated.

Author: Ben & Claude
Date: 2026-07-31
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from utils.astro_adapter import AstroImageHandler


def detrend_plane(w: np.ndarray) -> np.ndarray:
    """Subtract the best-fit plane (removes the linear sky gradient while
    keeping the noise structure under test)."""
    h, wd = w.shape
    yy, xx = np.mgrid[0:h, 0:wd].astype(np.float64)
    A = np.stack([np.ones(w.size), yy.ravel(), xx.ravel()], axis=1)
    coef, *_ = np.linalg.lstsq(A, w.ravel().astype(np.float64), rcond=None)
    return w - (A @ coef).reshape(h, wd)


def darkest_windows(data: np.ndarray, size: int, count: int, margin: int = 0):
    """Greedy selection of non-overlapping windows with the lowest medians.
    `margin` excludes a border strip (mirror padding, stacking edges,
    vignetting all fake strong correlation)."""
    H, W = data.shape
    step = max(size // 2, 1)
    cands = []
    for y in range(margin, H - size - margin + 1, step):
        for x in range(margin, W - size - margin + 1, step):
            cands.append((float(np.median(data[y:y + size, x:x + size])), y, x))
    cands.sort()
    chosen = []
    for med, y, x in cands:
        if all(abs(y - cy) >= size or abs(x - cx) >= size for _, cy, cx in chosen):
            chosen.append((med, y, x))
        if len(chosen) == count:
            break
    return chosen


def analyze_window(w: np.ndarray) -> dict:
    wc = detrend_plane(w.astype(np.float64))
    total = wc.std()
    dx = np.diff(wc, axis=1).std()
    dy = np.diff(wc, axis=0).std()
    r = (dx + dy) / (2.0 * total) if total > 0 else float('nan')
    white = (dx + dy) / (2.0 * np.sqrt(2.0))
    corr = np.sqrt(max(total ** 2 - white ** 2, 0.0))
    return {
        'std': total,
        'ratio': r,
        'rho1': 1.0 - r ** 2 / 2.0,
        'white_std': white,
        'correlated_std': corr,
    }


def verdict(ratio: float) -> str:
    if ratio >= 1.30:
        return 'blanc'
    if ratio >= 1.10:
        return 'moderement correle'
    return 'fortement correle'


def main():
    parser = argparse.ArgumentParser(
        description='Is the background noise white enough for DIP to bite?')
    parser.add_argument('image', help='FITS/XISF/TIFF image (linear)')
    parser.add_argument('--window', type=int, default=128,
                        help='Analysis window size in pixels (default 128)')
    parser.add_argument('--n-windows', type=int, default=5,
                        help='Number of darkest windows to analyze (default 5)')
    parser.add_argument('--margin', type=int, default=0,
                        help='Exclude a border strip this many pixels wide '
                             '(mirror padding and stacking edges fake '
                             'correlation; try --margin 256 on drizzled data)')
    parser.add_argument('--json', action='store_true',
                        help='Machine-readable output on stdout')
    args = parser.parse_args()

    data, _ = AstroImageHandler.load_raw(args.image)
    H, W = data.shape
    if H < args.window + 2 * args.margin or W < args.window + 2 * args.margin:
        print(f"Image {W}x{H} smaller than window + margins",
              file=sys.stderr)
        sys.exit(1)

    windows = darkest_windows(data, args.window, args.n_windows, args.margin)
    reports = []
    for med, y, x in windows:
        rep = analyze_window(data[y:y + args.window, x:x + args.window])
        rep.update({'y': y, 'x': x, 'median': med})
        reports.append(rep)

    med_ratio = float(np.median([r['ratio'] for r in reports]))
    med_std = float(np.median([r['std'] for r in reports]))
    med_white = float(np.median([r['white_std'] for r in reports]))
    med_corr = float(np.median([r['correlated_std'] for r in reports]))
    v = verdict(med_ratio)

    if args.json:
        json.dump({
            'image': str(Path(args.image)),
            'window': args.window,
            'ratio_median': med_ratio,
            'ratio_white_reference': float(np.sqrt(2.0)),
            'verdict': v,
            'background_std': med_std,
            'white_std': med_white,
            'correlated_std': med_corr,
            'windows': [{k: (float(v_) if isinstance(v_, (int, float, np.floating))
                             else v_) for k, v_ in r.items()} for r in reports],
        }, sys.stdout, indent=2)
        print()
        return

    print(f"🔬 {Path(args.image).name}  ({W}x{H})")
    print(f"   {len(reports)} fenetres de {args.window}px les plus sombres, "
          f"plan de gradient retire")
    for r in reports:
        print(f"   y={r['y']:5d} x={r['x']:5d}: std={r['std']:.3g}  "
              f"ratio={r['ratio']:.2f}  rho1={r['rho1']:+.2f}  "
              f"blanc={r['white_std']:.3g}  correle={r['correlated_std']:.3g}")
    print(f"\n   Ratio median : {med_ratio:.2f}  (blanc pur = 1.41)")
    print(f"   Bruit de fond : {med_std:.3g} total = "
          f"{med_white:.3g} blanc + {med_corr:.3g} correle (quadrature)")
    print(f"   Verdict : {v.upper()}")
    if v == 'blanc':
        print("   → Le DIP retirera l'essentiel de ce bruit.")
    elif v == 'moderement correle':
        print("   → Le DIP retirera la composante blanche; le reste demande "
              "un pretraitement (drizzle, destriping).")
    else:
        print("   → Bruit domine par une composante correlee (drizzle, "
              "trames) : refaire le pretraitement avant de lancer des "
              "heures de DIP.")


if __name__ == '__main__':
    main()
