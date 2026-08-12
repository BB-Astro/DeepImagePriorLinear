"""
Early stopping for Deep Image Prior based on windowed moving variance (ES-WMV).

The variance of the DIP output sequence follows a U-shaped curve: it decreases
while the network learns the signal, reaches a minimum near the peak-quality
iteration, then rises again as the network starts fitting noise. Tracking the
running minimum of the windowed moving variance therefore gives a
ground-truth-free estimate of the best iteration, and a patience rule on that
minimum gives an early stopping criterion.

Reference:
    Wang et al., "Early Stopping for Deep Image Prior", TMLR 2023.
    arXiv:2112.06074

Author: Ben & Claude
Date: 2026-07-31
"""

import numpy as np
import torch


class ESWMV:
    """Windowed moving variance tracker for DIP outputs.

    The paper computes the variance over full reconstructions. To keep memory
    bounded on large images, the variance is estimated here on a fixed random
    subset of pixels: the location of the variance minimum is preserved, only
    its absolute scale changes.
    """

    def __init__(
        self,
        window_size: int = 100,
        patience: int = 1000,
        max_pixels: int = 65536,
        seed: int = 0,
        min_start: int = 0,
    ):
        """min_start: no variance minimum is accepted before this iteration.
        Guards against the warmup dip: in the first ~100-400 iterations the
        output barely moves, the windowed variance is artificially low, and
        on structure-rich content that false dip can hold for the whole
        patience and freeze the selection on an untrained output (observed
        twice: selections at iters 392 and 125 on galaxy-bearing tiles whose
        true valleys were past 6000). Real valleys on such content are at
        6000-16000, so a floor of ~1500 costs nothing there; keep 0 for tiny
        pure-background crops whose legitimate valley is ~130."""
        self.window_size = window_size
        self.patience = patience
        self.max_pixels = max_pixels
        self.seed = seed
        self.min_start = min_start

        self._sample_idx = None   # torch.LongTensor on the output device
        self._ring = None         # (window_size, n_pixels) float32 ring buffer
        self._count = 0           # number of outputs recorded so far

        self.min_variance = float('inf')
        self.min_iteration = -1
        self.frozen = False       # True once the minimum stagnated for `patience`
        self.variance_history = []  # list of (iteration, variance)

    def _init_buffers(self, flat_out: torch.Tensor) -> None:
        n = flat_out.numel()
        if n > self.max_pixels:
            rng = np.random.default_rng(self.seed)
            idx = np.sort(rng.choice(n, size=self.max_pixels, replace=False))
        else:
            idx = np.arange(n)
        self._sample_idx = torch.from_numpy(idx).to(flat_out.device)
        self._ring = np.empty((self.window_size, len(idx)), dtype=np.float32)

    def update(self, out: torch.Tensor, iteration: int) -> bool:
        """Record one raw network output.

        Returns True when this iteration sets a new variance minimum, i.e.
        the caller should snapshot its current output as the best one.
        """
        flat = out.detach().reshape(-1)
        if self._sample_idx is None:
            self._init_buffers(flat)

        sample = flat[self._sample_idx].float().cpu().numpy()
        self._ring[self._count % self.window_size] = sample
        self._count += 1

        if self._count < self.window_size:
            return False

        variance = float(self._ring.var(axis=0).mean())
        self.variance_history.append((iteration, variance))

        # Warmup guard: record the curve but accept no minimum yet. The
        # patience clock only starts at min_start (min_iteration floor).
        if iteration < self.min_start:
            return False
        if self.min_iteration < self.min_start:
            self.min_iteration = max(self.min_iteration, self.min_start)

        # Once the minimum has stagnated for `patience` iterations, freeze the
        # selection (the paper's ES point is the FIRST patience-confirmed
        # minimum). Without this, very long runs can drift below the early
        # valley late in training, when Adam has converged and the output
        # jitter shrinks even though the output is noise-fitted, and the
        # global minimum would then re-select an overfitted iteration.
        if not self.frozen and iteration - self.min_iteration >= self.patience:
            self.frozen = True

        if not self.frozen and variance < self.min_variance:
            self.min_variance = variance
            self.min_iteration = iteration
            return True
        return False

    def should_stop(self, iteration: int) -> bool:
        """True when the variance minimum has stagnated for `patience` iterations."""
        return self.frozen

    def get_state(self) -> dict:
        """Snapshot for backtracking: when the caller rewinds the network,
        the selection state must rewind too, otherwise the ring buffer keeps
        outputs from the abandoned trajectory and the freeze/min bookkeeping
        no longer describes the network being trained."""
        return {
            'ring': None if self._ring is None else self._ring.copy(),
            'count': self._count,
            'min_variance': self.min_variance,
            'min_iteration': self.min_iteration,
            'frozen': self.frozen,
            'history_len': len(self.variance_history),
        }

    def set_state(self, state: dict) -> None:
        """Restore a snapshot taken with get_state()."""
        self._ring = None if state['ring'] is None else state['ring'].copy()
        self._count = state['count']
        self.min_variance = state['min_variance']
        self.min_iteration = state['min_iteration']
        self.frozen = state['frozen']
        del self.variance_history[state['history_len']:]
