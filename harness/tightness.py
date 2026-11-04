"""Tightness study: how far the proven thresholds sit above the actual rounding error.

Uses the bit-exact CPU model of the coalesced CUDA GEMV (`emulate_gemv_coalesced`, 256
threads, 16-byte vectors), so the errors measured are the GPU kernel's errors. Reports:

1. per-row tightness:   gamma_h * sum_j |W_ij x_j|  /  |y_i - exact_i|
2. per-block tightness: tau_beta / |rho_beta| for the blocked checker at b = 64 and 128
   (the number to set beside V-ABFT's reported 48-158x in BF16)
3. a probabilistic tier (Higham & Mary 2019): gamma~_h(lambda) = exp(lambda sqrt(h) u +
   h u^2 / (1 - u)) - 1, with lambda chosen so that the bound fails with probability at
   most 1e-6 per row under their model of independent, mean-zero rounding errors. This is a
   model of the hardware, not a proof; it is reported separately from the proven bound.
4. "why not tests": a per-element tolerance calibrated on Gaussian activations (10x the
   largest error seen) applied to heavy-tailed and outlier activations, against the proven
   threshold.

Activations are synthetic (Gaussian, Student-t with 3 dof, LLM-like outliers where 1% of
channels are 100x larger, mixed scale); weights are N(0, 0.02^2) in BF16.

Run: uv run python harness/tightness.py   (about a minute on CPU)
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from pck import (
    U32,
    abs_row_sums,
    bf16,
    emulate_gemv_coalesced,
    gamma,
    reference_gemv,
    round_up_f32,
    tree_height,
)

M = N = 4096
THREADS = 256
TRIALS = 4
DISTS = ["gauss", "student-t", "outliers", "mixed-scale"]


def activations(kind: str, rng: np.random.Generator) -> NDArray[np.float32]:
    if kind == "gauss":
        x = rng.standard_normal(N)
    elif kind == "student-t":
        x = rng.standard_t(3, N)
    elif kind == "outliers":
        x = rng.standard_normal(N)
        x[rng.choice(N, N // 100, replace=False)] *= 100.0
    else:
        x = rng.standard_normal(N) * 2.0 ** rng.integers(-20, 21, N)
    return bf16(x)


def hm_gamma(h: int, u: float, fail: float) -> float:
    """Higham-Mary probabilistic constant for h roundings, failure probability <= fail."""
    lam = math.sqrt(2 * math.log(2 * h / fail)) / (1 - u)
    return math.expm1(lam * math.sqrt(h) * u + h * u * u / (1 - u))


def block_ratios(
    W: NDArray[np.float32],
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    b: int,
    h: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """tau / |rho| per block, one random sign vector per block (checker arithmetic)."""
    m, n = W.shape
    nb = m // b
    Wb = W.reshape(nb, b, n).astype(np.float64)
    x64 = x.astype(np.float64)
    r = rng.choice([-1.0, 1.0], (nb, b))
    c = np.einsum("kb,kbn->kn", r, Wb).astype(np.float32).astype(np.float64)
    a = round_up_f32(np.abs(Wb).sum(axis=1))
    s = a @ np.abs(x64)
    yb = y.astype(np.float64).reshape(nb, b)
    rho = c @ x64 - (r * yb).sum(axis=1)
    tau = (gamma(h) + U32) * s
    ratio: NDArray[np.float64] = tau / np.maximum(np.abs(rho), 1e-300)
    return ratio


def summary(v: NDArray[np.float64]) -> str:
    return f"median {np.median(v):9.0f}x   p1 {np.percentile(v, 1):9.0f}x   min {np.min(v):9.0f}x"


def main() -> None:
    rng = np.random.default_rng(0)
    W = bf16(rng.standard_normal((M, N)) * 0.02)
    h = tree_height(N, THREADS, vec=8)
    g = gamma(h)
    g_hm = hm_gamma(h, U32, 1e-6)
    print(f"m = n = {N}, coalesced kernel ({THREADS} threads), tree height h = {h}")
    print(f"gamma_h = {g:.3e} (proven)   Higham-Mary gamma~ = {g_hm:.3e} (probabilistic)")

    errs: dict[str, list[NDArray[np.float64]]] = {d: [] for d in DISTS}
    rel: dict[str, list[NDArray[np.float64]]] = {d: [] for d in DISTS}
    blocks: dict[tuple[str, int], list[NDArray[np.float64]]] = {}
    bounds: dict[str, list[NDArray[np.float64]]] = {d: [] for d in DISTS}
    for dist in DISTS:
        for _ in range(TRIALS):
            x = activations(dist, rng)
            y = emulate_gemv_coalesced(W, x, THREADS)
            exact = reference_gemv(W, x)
            err = np.abs(y.astype(np.float64) - exact)
            s_row = abs_row_sums(W, x)
            assert np.all(err <= g * s_row), "proven bound violated"
            errs[dist].append(err)
            bounds[dist].append(s_row)
            rel[dist].append(err / np.maximum(np.abs(exact), 1e-300))
            for b in (64, 128):
                blocks.setdefault((dist, b), []).append(block_ratios(W, x, y, b, h, rng))

    print("\n1. Per-row tightness, proven bound / actual error (rows with nonzero error)")
    for dist in DISTS:
        e, s = np.concatenate(errs[dist]), np.concatenate(bounds[dist])
        nz = e > 0
        print(f"  {dist:>11s}: {summary(g * s[nz] / e[nz])}   ({(~nz).mean():.1%} rows exact)")

    print("\n2. Per-block tightness, tau / |rho| (one sign vector per block)")
    for b in (64, 128):
        for dist in DISTS:
            print(f"  b = {b:3d} {dist:>11s}: {summary(np.concatenate(blocks[(dist, b)]))}")
    print("  V-ABFT (reported, BF16): 48x to 158x")

    print("\n3. Probabilistic tier (Higham-Mary, P(fail) <= 1e-6 per row; not proven)")
    for dist in DISTS:
        e, s = np.concatenate(errs[dist]), np.concatenate(bounds[dist])
        nz = e > 0
        viol = int((e > g_hm * s).sum())
        print(f"  {dist:>11s}: {summary(g_hm * s[nz] / e[nz])}   violations {viol}")

    print("\n4. Why not tests: per-element tolerance calibrated on Gaussian x (10x max error)")
    atol = 10 * float(np.max(np.concatenate(errs["gauss"])))
    rtol = 10 * float(np.max(np.concatenate(rel["gauss"])))
    print(f"  atol = {atol:.3e}, rtol = {rtol:.3e}")
    print(f"  {'x':>11s} {'atol alarms':>14s} {'rtol alarms':>14s} {'proven alarms':>14s}")
    for dist in DISTS:
        e, r_ = np.concatenate(errs[dist]), np.concatenate(rel[dist])
        s = np.concatenate(bounds[dist])
        print(
            f"  {dist:>11s} {(e > atol).mean():14.2%} {(r_ > rtol).mean():14.2%}"
            f" {(e > g * s).mean():14.2%}"
        )


if __name__ == "__main__":
    main()
