"""ES-WMV behavior tests: freeze, warmup guard, backtracking state."""
import torch

from utils.early_stopping import ESWMV


def synthetic_run(wmv, phases, seed=0):
    """phases: list of (n_iters, noise_amplitude). Returns stop iteration."""
    torch.manual_seed(seed)
    base = torch.rand(1, 1, 16, 16)
    i = 0
    for n, amp in phases:
        for _ in range(n):
            wmv.update(base + amp * torch.randn_like(base), i)
            if wmv.should_stop(i):
                return i
            i += 1
    return i


def test_freeze_at_first_confirmed_minimum():
    wmv = ESWMV(window_size=50, patience=200, max_pixels=256)
    stop = synthetic_run(wmv, [(300, 0.001), (600, 0.05)])
    assert wmv.frozen
    assert wmv.min_iteration < 300
    assert stop < 600


def test_warmup_guard_rejects_early_dip():
    wmv = ESWMV(window_size=50, patience=400, max_pixels=256, min_start=600)
    synthetic_run(wmv, [(200, 1e-6), (900, 0.05), (900, 0.001)])
    assert wmv.min_iteration >= 600, 'no minimum may be accepted before min_start'
    assert wmv.min_iteration >= 1100, 'selection must land in the true valley'


def test_state_snapshot_and_restore():
    torch.manual_seed(1)
    wmv = ESWMV(window_size=20, patience=500, max_pixels=256)
    base = torch.rand(1, 1, 16, 16)
    for i in range(100):
        wmv.update(base + 0.01 * torch.randn_like(base), i)
    snap = wmv.get_state()
    ref = (wmv.min_variance, wmv.min_iteration, wmv._count,
           len(wmv.variance_history))

    # Diverged trajectory pollutes the ring and the history
    for i in range(100, 160):
        wmv.update(base + 10.0 * torch.randn_like(base), i)
    assert len(wmv.variance_history) > ref[3]

    wmv.set_state(snap)
    assert (wmv.min_variance, wmv.min_iteration, wmv._count,
            len(wmv.variance_history)) == ref

    # The tracker keeps working after restore
    for i in range(100, 200):
        wmv.update(base + 0.01 * torch.randn_like(base), i)
    assert wmv._count == 200
