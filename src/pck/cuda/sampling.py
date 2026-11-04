"""Trusted top-k / top-p sampling checker on the GPU, and untrusted GPU samplers to test it.

The checker (``check_topk_gpu``, ``check_topp_gpu``) mirrors ``pck.sampling``: exact FP32
comparisons for the top-k prefix property, FP64 masses for top-p minimality. Selections are
``(B, L)`` int64 tensors with per-row lengths, so top-p nuclei of different sizes share one
launch. The samplers (``topk_sample``, ``topp_select``, ``sharded_topk_gpu``) are the code
under test, not trusted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from pck.cuda._ext import extension
from pck.fp import U64, gamma


@dataclass(frozen=True)
class SamplingCheck:
    """Per-row checker outputs, each of shape (B,)."""

    passed: torch.Tensor
    min_selected: torch.Tensor
    max_rejected: torch.Tensor
    mass: torch.Tensor
    mass_without_smallest: torch.Tensor


def _lengths(selected: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    if lengths is None:
        return torch.full(
            (selected.shape[0],), selected.shape[1], dtype=torch.int64, device=selected.device
        )
    return lengths


def check_topk_gpu(logits: torch.Tensor, selected: torch.Tensor, k: int) -> SamplingCheck:
    """Check that each row of ``selected`` (B, k) is a valid top-k set of ``logits`` (B, V)."""
    out = extension().sampling_check(
        logits, selected, _lengths(selected, None), k, False, 1.0, 1.0, 0.0
    )
    return SamplingCheck(*out)


def check_topp_gpu(
    logits: torch.Tensor,
    selected: torch.Tensor,
    lengths: torch.Tensor,
    p: float,
    temperature: float = 1.0,
    tol: float = 1e-6,
) -> SamplingCheck:
    """Check that ``selected[b, :lengths[b]]`` is a valid, minimal top-p nucleus of row b.

    ``tol`` is the sampler's declared probability error; the checker adds its own FP64
    rounding slack (CUDA's FP64 exp is within 1 ulp, so each mass is within
    gamma_{V+8}(u64) of exact).
    """
    if not 0 < p <= 1:
        raise ValueError(f"p must be in (0, 1], got {p}")
    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol}")
    slack = math.nextafter(tol + gamma(logits.shape[1] + 8, U64), math.inf)
    out = extension().sampling_check(logits, selected, lengths, 0, True, p, temperature, slack)
    return SamplingCheck(*out)


def topk_sample(
    logits: torch.Tensor, k: int, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Untrusted sampler: torch.topk, then sample within the set. Returns (selected, token)."""
    values, idx = torch.topk(logits, k, dim=-1)
    probs = torch.softmax(values, dim=-1)
    pick = torch.multinomial(probs, 1, generator=generator)
    return idx, idx.gather(-1, pick).squeeze(-1)


def topp_select(
    logits: torch.Tensor, p: float, temperature: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Untrusted top-p: FP32 softmax, sort, then the smallest prefix with mass >= p.

    The running mass is accumulated in FP64 so the sampler's own error is that of its FP32
    probabilities (about 1e-6 at most), within the checker's usual ``tol``. Returns
    (selected (B, V) sorted indices, lengths (B,)); only the first lengths[b] entries of
    row b are the nucleus.
    """
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, order = torch.sort(probs, dim=-1, descending=True, stable=True)
    mass = torch.cumsum(sorted_probs.double(), dim=-1)
    lengths = (mass < p).sum(dim=-1) + 1
    return order.contiguous(), lengths.clamp(max=logits.shape[1]).to(torch.int64)


def sharded_topk_gpu(logits: torch.Tensor, k: int, shards: int, per_shard: int) -> torch.Tensor:
    """Untrusted approximate top-k (the buggy kind): top ``per_shard`` per shard, then top k.

    Same fault model as ``pck.faults.sharded_topk``; the vocabulary must split evenly.
    """
    rows, vocab = logits.shape
    if vocab % shards or per_shard > vocab // shards or k > shards * per_shard:
        raise ValueError("need vocab % shards == 0, per_shard <= vocab / shards, k <= pool")
    blocks = logits.view(rows, shards, vocab // shards)
    _, local = torch.topk(blocks, per_shard, dim=-1)
    offsets = torch.arange(shards, device=logits.device).view(1, shards, 1) * (vocab // shards)
    pool = (local + offsets).view(rows, -1)
    _, pick = torch.topk(logits.gather(-1, pool), k, dim=-1)
    return pool.gather(-1, pick)
