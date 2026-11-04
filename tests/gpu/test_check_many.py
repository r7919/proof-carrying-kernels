import pytest
import torch

from pck.cuda import GpuBlockedChecker, check_many, gemv_coalesced
from pck.cuda._ext import extension
from pck.fp import tree_height

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

THREADS = 128
# (m, n, block, checks, segments): different shapes and settings in one batch.
CONFIGS = [
    (512, 4096, 64, 1, None),
    (1024, 512, 32, 2, 1),
    (256, 14336, 128, 4, 7),
    (64, 1000, 8, 1, 3),
    (2048, 4096, 256, 2, None),
]


def make(
    m: int, n: int, block: int, checks: int, segments: int | None, seed: int
) -> tuple[GpuBlockedChecker, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    W = (torch.randn(m, n, generator=gen) * 0.02).to(torch.bfloat16).cuda()
    x = torch.randn(n, generator=gen).to(torch.bfloat16).cuda()
    checker = GpuBlockedChecker(
        W, block, tree_height(n, THREADS, vec=8), checks=checks, generator=gen, segments=segments
    )
    return checker, x, gemv_coalesced(W, x, THREADS)


def test_batch_matches_individual_checks() -> None:
    items = [make(*cfg, seed=i) for i, cfg in enumerate(CONFIGS)]
    batched = check_many(items)
    for (checker, x, y), res in zip(items, batched, strict=True):
        alone = checker.check(x, y)
        assert torch.equal(res.passed, alone.passed)
        assert torch.equal(res.rho, alone.rho)
        assert torch.equal(res.tau, alone.tau)
        assert res.verdict.all()


def test_more_than_one_launch_and_fault_isolation() -> None:
    items = [make(*CONFIGS[i % len(CONFIGS)], seed=10 + i) for i in range(11)]  # 8 + 3
    checker, x, y = items[9]
    bad = y.clone()
    bad[3] = float("inf")
    items[9] = (checker, x, bad)
    results = check_many(items)
    assert len(results) == 11
    for i, res in enumerate(results):
        if i == 9:
            assert not res.verdict[0] and res.verdict[1:].all()
        else:
            assert res.verdict.all()


def test_rejects_same_checker_twice() -> None:
    item = make(*CONFIGS[0], seed=0)
    with pytest.raises(ValueError, match="once"):
        check_many([item, item])


def test_extension_rejects_bad_batches() -> None:
    checker, x, y = make(*CONFIGS[0], seed=0)
    args = [
        checker.c,
        checker.a,
        checker.r,
        x,
        y,
        checker._partial_cx,
        checker._partial_s,
        checker._counters,
        checker.coef,
        checker.geps,
        checker.tau_floor,
    ]
    with pytest.raises(RuntimeError, match="batch size"):
        extension().blocked_check(*([] for _ in args))
    with pytest.raises(RuntimeError, match="batch size"):
        extension().blocked_check(*([a] * 9 for a in args))
    lists = [[a] for a in args]
    lists[4] = [y, y]
    with pytest.raises(RuntimeError, match="same length"):
        extension().blocked_check(*lists)
