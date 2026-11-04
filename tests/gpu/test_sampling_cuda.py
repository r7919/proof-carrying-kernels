import math

import numpy as np
import pytest
import torch

from pck.cuda import (
    check_topk_gpu,
    check_topp_gpu,
    sharded_topk_gpu,
    topk_sample,
    topp_select,
)
from pck.fp import bf16
from pck.sampling import check_topk, check_topp, softmax64

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]


def logits(
    rows: int, vocab: int, seed: int, scale: float = 3.0, ties: bool = False
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((rows, vocab)) * scale).astype(np.float32)
    if ties:
        a = bf16(a)
    return torch.from_numpy(a).cuda()


@pytest.mark.parametrize(("vocab", "k"), [(1, 1), (7, 3), (1000, 50), (4096, 1), (151_936, 40)])
@pytest.mark.parametrize("ties", [False, True])
def test_topk_sampler_passes_and_matches_cpu(vocab: int, k: int, ties: bool) -> None:
    lg = logits(8, vocab, seed=vocab + k, ties=ties)
    selected, token = topk_sample(lg, k, generator=torch.Generator("cuda").manual_seed(0))
    res = check_topk_gpu(lg, selected, k)
    assert res.passed.all()
    assert (selected == token[:, None]).any(dim=1).all()
    for b in range(8):
        assert check_topk(lg[b].cpu().numpy(), selected[b].cpu().numpy(), k)


@pytest.mark.parametrize("seed", range(5))
def test_random_swaps_match_cpu_verdict(seed: int) -> None:
    rows, vocab, k = 8, 2000, 20
    lg = logits(rows, vocab, seed=seed, ties=True)
    selected, _ = topk_sample(lg, k)
    rng = np.random.default_rng(seed)
    sel = selected.clone()
    for b in range(rows):  # replace one selected index by a random other index
        rejected = np.setdiff1d(np.arange(vocab), sel[b].cpu().numpy())
        sel[b, rng.integers(k)] = int(rng.choice(rejected))
    gpu = check_topk_gpu(lg, sel, k).passed.cpu().numpy()
    cpu = [check_topk(lg[b].cpu().numpy(), sel[b].cpu().numpy(), k) for b in range(rows)]
    np.testing.assert_array_equal(gpu, cpu)


def test_sharded_fault_matches_cpu_and_is_caught_when_clustered() -> None:
    rows, vocab, k = 8, 4096, 16
    lg = logits(rows, vocab, seed=3)
    lg[0, :k] = 100.0 + torch.arange(k, device="cuda", dtype=torch.float32)  # cluster in shard 0
    got = sharded_topk_gpu(lg, k, shards=8, per_shard=4)
    gpu = check_topk_gpu(lg, got, k).passed.cpu().numpy()
    cpu = [check_topk(lg[b].cpu().numpy(), got[b].cpu().numpy(), k) for b in range(rows)]
    np.testing.assert_array_equal(gpu, cpu)
    assert not gpu[0]
    assert check_topk_gpu(lg, sharded_topk_gpu(lg, k, 8, per_shard=k), k).passed.all()


@pytest.mark.parametrize("fault", ["duplicate", "out_of_range", "negative", "nan"])
def test_malformed_or_nan_fails_only_its_row(fault: str) -> None:
    lg = logits(4, 100, seed=4)
    selected, _ = topk_sample(lg, 5)
    if fault == "duplicate":
        selected[2, 1] = selected[2, 0]
    elif fault == "out_of_range":
        selected[2, 1] = 100
    elif fault == "negative":
        selected[2, 1] = -1
    else:
        lg[2, 50] = float("nan")
    np.testing.assert_array_equal(
        check_topk_gpu(lg, selected, 5).passed.cpu().numpy(), [True, True, False, True]
    )


@pytest.mark.parametrize("vocab", [10, 1000, 151_936])
@pytest.mark.parametrize(("p", "temp"), [(0.5, 1.0), (0.9, 0.7), (0.95, 1.5), (1.0, 1.0)])
def test_fp32_topp_sampler_passes_and_masses_match(vocab: int, p: float, temp: float) -> None:
    lg = logits(8, vocab, seed=vocab, scale=2.0)
    selected, lengths = topp_select(lg, p, temp)
    res = check_topp_gpu(lg, selected, lengths, p, temp, tol=1e-5)
    assert res.passed.all()
    for b in range(8):
        a = lg[b].cpu().numpy()
        sel = selected[b, : lengths[b]].cpu().numpy()
        assert check_topp(a, sel, p, temp, tol=1e-5)
        mass = float(softmax64(a, temp)[sel].sum())
        assert math.isclose(float(res.mass[b]), mass, rel_tol=16 * 2.0**-24)  # FP32 expf


@pytest.mark.parametrize("delta", [-1, 1])
def test_topp_off_by_one_matches_cpu(delta: int) -> None:
    rows, vocab, p = 8, 500, 0.8
    lg = logits(rows, vocab, seed=11)
    selected, lengths = topp_select(lg, p)
    lengths = (lengths + delta).clamp(1, vocab)
    gpu = check_topp_gpu(lg, selected, lengths, p, tol=0.0).passed.cpu().numpy()
    cpu = [
        check_topp(lg[b].cpu().numpy(), selected[b, : lengths[b]].cpu().numpy(), p, tol=0.0)
        for b in range(rows)
    ]
    np.testing.assert_array_equal(gpu, cpu)


def test_rejects_bad_inputs() -> None:
    lg = logits(2, 10, seed=0)
    sel, lengths = topp_select(lg, 0.5)
    with pytest.raises(ValueError, match="p must"):
        check_topp_gpu(lg, sel, lengths, 0.0)
    with pytest.raises(ValueError, match="tol"):
        check_topp_gpu(lg, sel, lengths, 0.5, tol=-1.0)
    with pytest.raises(RuntimeError, match="segments"):
        check_topk_gpu(lg, topk_sample(lg, 2)[0], 2, segments=11)
    with pytest.raises(RuntimeError, match="selected must be"):
        check_topk_gpu(lg, sel.int(), 1)
    with pytest.raises(ValueError, match="vocab % shards"):
        sharded_topk_gpu(lg, 2, shards=3, per_shard=1)


@pytest.mark.parametrize("segments", [1, 2, 7, 38, 151_936])
def test_segment_count_does_not_change_verdicts(segments: int) -> None:
    lg = logits(4, 151_936, seed=21, scale=2.0)
    selected, _ = topk_sample(lg, 30)
    bad = selected.clone()
    bad[1, 0] = selected[1, 1]  # duplicate in row 1
    ref = check_topk_gpu(lg, bad, 30, segments=1)
    got = check_topk_gpu(lg, bad, 30, segments=segments)
    assert torch.equal(got.passed, ref.passed) and ref.passed.tolist() == [True, False, True, True]
    assert torch.equal(got.min_selected, ref.min_selected)
    assert torch.equal(got.max_rejected, ref.max_rejected)

    nucleus, lengths = topp_select(lg, 0.9)
    ref_p = check_topp_gpu(lg, nucleus, lengths, 0.9, tol=1e-5, segments=1)
    got_p = check_topp_gpu(lg, nucleus, lengths, 0.9, tol=1e-5, segments=segments)
    assert got_p.passed.all() and ref_p.passed.all()
    # Segment maxima change each expf argument, so masses agree to FP32 rounding only.
    torch.testing.assert_close(got_p.mass, ref_p.mass, rtol=16 * 2.0**-24, atol=0)


def test_vocabulary_beyond_shared_memory_bitmap() -> None:
    lg = logits(2, 500_000, seed=22)
    selected, _ = topk_sample(lg, 64)
    assert check_topk_gpu(lg, selected, 64).passed.all()
    nucleus, lengths = topp_select(lg, 0.5)
    assert check_topp_gpu(lg, nucleus, lengths, 0.5, tol=1e-5).passed.all()
