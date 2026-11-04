"""Numerics study: BF16 GEMV with FP32 tree accumulation against blocked checksum checkers.

Reports, under Gaussian and heavy-tailed (Student-t, 3 dof) activations:
  * which planted exploits each block size catches, next to a KernelBench-style
    atol = rtol = 1e-2 tolerance;
  * false alarms on the legit kernel;
  * bit-flip detection by bit position, including the NaN-unsafe "rho > tau" test;
  * per-row tightness and the single-output detection floor.

Run: uv run python harness/numerics_sim.py   (about 7 minutes on CPU)
"""

from __future__ import annotations

import numpy as np

from pck import (
    BlockedChecker,
    abs_row_sums,
    bf16,
    emulate_gemv,
    flip_bit,
    gamma,
    reference_gemv,
    tree_height,
)

BITS = [
    (12, "mantissa"),
    (14, "mantissa"),
    (16, "mantissa"),
    (18, "mantissa"),
    (20, "mantissa"),
    (22, "mantissa"),
    (23, "exponent"),
    (27, "exponent"),
    (30, "exponent"),
    (31, "sign"),
]


def run(xdist: str, rng: np.random.Generator) -> None:
    m, n, T = 4096, 4100, 256
    L = -(-n // T)
    h = tree_height(n, T)
    W = bf16(rng.standard_normal((m, n)) * 0.02)
    x = bf16(rng.standard_normal(n) if xdist == "gauss" else rng.standard_t(3, n))
    y = reference_gemv(W, x)
    s_row = abs_row_sums(W, x)
    legit = emulate_gemv(W, x, T)

    print(f"\n=== x ~ {xdist}; m={m}, n={n}, L={L}, T={T}, tree height h={h} ===")
    print(
        f"gamma_(n-1)={gamma(n - 1):.2e}  gamma_h={gamma(h):.2e}  "
        f"ratio={gamma(n - 1) / gamma(h):.0f}x"
    )

    cases = {
        "legit FP32 kernel": legit,
        "skip_tail (drop 4)": emulate_gemv(W, x, T, drop_tail=4),
        "BF16 accumulators": emulate_gemv(W, x, T, acc_bf16=True),
        "BF16 partials before tree": emulate_gemv(W, x, T, partial_bf16=True),
        "BF16 output (downcast)": bf16(legit),
    }
    blocks = (8, 64, 128, m)
    checkers = {b: BlockedChecker(W, x, b, h) for b in blocks}
    print(
        f"{'case':30s}"
        + "".join(f"{'b=' + str(b) + ' (K=8)':>16s}" for b in blocks)
        + f"{'atol=rtol=1e-2':>16s}{'spec viol.':>12s}"
    )
    for name, yk in cases.items():
        e = yk.astype(np.float64) - y
        row = f"{name:30s}"
        for b in blocks:
            caught = not checkers[b].verdict(yk, rng, checks=8).all()
            row += f"{('CAUGHT' if caught else 'pass'):>16s}"
        kb = np.all(np.abs(e) <= 1e-2 + 1e-2 * np.abs(y))
        spec = np.any(np.abs(e) > gamma(h) * s_row)
        row += f"{('pass' if kb else 'CAUGHT'):>16s}{('yes' if spec else 'no'):>12s}"
        print(row)

    c64 = checkers[64]
    fa = sum(int((~c64.verdict(legit, rng, checks=1)).sum()) for _ in range(200))
    print(f"false alarms, legit kernel, b=64: {fa} of {200 * (m // 64)} block checks")

    rows = rng.choice(m, 200, replace=False)
    print("bit-flip detection at b=64 (fraction of 200 random rows):")
    print(f"  {'bit':>4s} {'kind':>10s} {'K=1':>7s} {'K=4':>7s}   naive 'rho>tau' test (K=1)")
    for bit, kind in BITS:
        k1 = k4 = naive = 0
        for i in rows:
            f = flip_bit(legit, i, bit)
            blk = i // 64
            k1 += not c64.verdict(f, rng, checks=1)[blk]
            k4 += not c64.verdict(f, rng, checks=4)[blk]
            # Naive test: alarm only if |rho| > tau, so a NaN residual slips through.
            r = rng.choice([-1.0, 1.0], c64.block)
            cc = (r @ c64.W_blocks[blk]).astype(np.float32).astype(np.float64)
            with np.errstate(invalid="ignore", over="ignore"):
                rho = cc @ c64.x - r @ f[blk * 64 : (blk + 1) * 64].astype(np.float64)
                naive += bool(np.abs(rho) > c64.tau_base[blk] * 1.001)
        N = len(rows)
        print(f"  {bit:>4d} {kind:>10s} {k1 / N:7.1%} {k4 / N:7.1%}   {naive / N:7.1%}")

    tight = gamma(h) * s_row / np.maximum(np.abs(legit - y), 1e-300)
    print(
        f"per-row tightness (gamma_h*s / actual err): "
        f"median {np.median(tight):.0f}x, min {np.min(tight):.0f}x"
    )
    for b in (64, 128):
        floor = np.median(2 * checkers[b].tau_base) / np.median(np.abs(y))
        print(
            f"b={b}: single-output detection floor 2*tau/median|y| = {floor:.3f};"
            f" extra traffic (FP32 c per b BF16 rows) = {2 / b:.1%}"
        )


def main() -> None:
    rng = np.random.default_rng(1)
    for d in ("gauss", "student-t(3)"):
        run(d, rng)


if __name__ == "__main__":
    main()
