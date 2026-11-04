import pytest
import torch

from pck.cuda import GpuBlockedChecker, SideStreamChecker, gemv_coalesced
from pck.fp import tree_height

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

THREADS = 256


def setup(m: int, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, GpuBlockedChecker]:
    gen = torch.Generator().manual_seed(seed)
    W = (torch.randn(m, n, generator=gen) * 0.02).to(torch.bfloat16).cuda()
    x = torch.randn(n, generator=gen).to(torch.bfloat16).cuda()
    checker = GpuBlockedChecker(W, 64, tree_height(n, THREADS, vec=8), checks=2, generator=gen)
    return W, x, checker


def test_side_stream_matches_same_stream() -> None:
    W, x, checker = setup(512, 4096, seed=0)
    y = gemv_coalesced(W, x, THREADS)
    serial = checker.check(x, y)
    side = SideStreamChecker()
    result = side.submit(checker, x, y).wait()
    assert torch.equal(result.passed, serial.passed)
    assert torch.equal(result.rho, serial.rho)
    assert torch.equal(result.tau, serial.tau)
    assert result.verdict.all()


def test_check_sees_output_even_if_freed_and_overwritten() -> None:
    """record_stream keeps y alive until the queued check has read it."""
    W, x, checker = setup(1024, 4096, seed=1)
    side = SideStreamChecker()
    pending = []
    for _ in range(20):
        y = gemv_coalesced(W, x, THREADS)
        pending.append(side.submit(checker, x, y))
        del y
        # Allocations of the same size would reuse y's memory if it were released early.
        garbage = torch.full((1024,), float("nan"), device="cuda")
        garbage.add_(1)
    for p in pending:
        assert p.wait().verdict.all()


def test_many_layers_overlap_and_detect_a_fault() -> None:
    side = SideStreamChecker()
    layers = [setup(m, n, seed=s) for s, (m, n) in enumerate([(512, 4096), (2048, 512)] * 3)]
    pending = []
    for i, (W, x, checker) in enumerate(layers):
        y = gemv_coalesced(W, x, THREADS)
        if i == 3:
            y[7] += 1.0  # a large single-output fault
        pending.append(side.submit(checker, x, y))
    side.wait_all()
    verdicts = [p.wait().verdict for p in pending]
    for i, v in enumerate(verdicts):
        if i == 3:
            assert not v[0] and v[1:].all()
        else:
            assert v.all()
