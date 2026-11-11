"""Phase 1 campaign on the GPU: false alarms over >= 10^7 clean checks, and bit-flip detection.

Part 1 (false alarms). Runs one fused Llama-3-8B decode layer (qkv, o, gate_up, down) with
fresh activations every trial and both GEMV kernels, checking every output block with K
sign vectors. Activations cycle through four distributions: Gaussian, Student-t (3 dof),
mixed-scale (magnitudes over 2^-20..2^20) and LLM-like outliers (1% of channels x100).
Secret signs are redrawn every --redraw trials. A block check is one (sign vector, block)
pair; the target is >= 10^7 of them with zero alarms. Also reports the largest
|rho| / tau seen, i.e. how close a legit output ever came to the threshold.

Part 2 (bit flips). On a 4096 x 4096 GEMV, flips one bit in one random output of every
block per trial and records the share of blocks that raise the alarm, by bit position,
block size and activation distribution (the GPU counterpart of harness/numerics_sim.py).

Run: uv run python harness/campaign.py            (about 10-20 minutes)
     uv run python harness/campaign.py --quick    (smoke test, ~1 minute)
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch

from pck.cuda import DEFAULT_BLOCK, GpuBlockedChecker, check_many, gemv_coalesced, gemv_naive
from pck.cuda.timing import device_name
from pck.fp import tree_height

THREADS = 256
LAYER = [("qkv", 6144, 4096), ("o", 4096, 4096), ("gate_up", 28672, 4096), ("down", 4096, 14336)]
BITS = [(12, "mantissa"), (14, "mantissa"), (16, "mantissa"), (18, "mantissa"),
        (20, "mantissa"), (22, "mantissa"), (23, "exponent"), (27, "exponent"),
        (30, "exponent"), (31, "sign")]  # fmt: skip


def activations(kind: str, n: int, gen: torch.Generator) -> torch.Tensor:
    if kind == "gauss":
        x = torch.randn(n, generator=gen)
    elif kind == "student-t":
        x = torch.distributions.StudentT(3.0).sample((n,))
    elif kind == "mixed-scale":
        x = torch.randn(n, generator=gen) * 2.0 ** torch.randint(-20, 21, (n,), generator=gen)
    elif kind == "outliers":
        x = torch.randn(n, generator=gen)
        idx = torch.randperm(n, generator=gen)[: max(1, n // 100)]
        x[idx] *= 100.0
    else:
        raise ValueError(kind)
    return x.to(torch.bfloat16).cuda()


DISTS = ["gauss", "student-t", "mixed-scale", "outliers"]
Gemv = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
KERNELS: dict[str, tuple[Gemv, int]] = {"coalesced": (gemv_coalesced, 8), "naive": (gemv_naive, 1)}


def false_alarms(block: int, checks: int, target: int, redraw: int, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)  # StudentT samples from the global generator
    weights = [
        (torch.randn(m, n, generator=gen) * 0.02).to(torch.bfloat16).cuda() for _, m, n in LAYER
    ]
    per_trial = checks * sum(m // block for _, m, _ in LAYER)
    print(
        f"\n== false alarms: b = {block}, K = {checks};"
        f" {per_trial} block checks per kernel per trial"
    )
    print(
        f"{'kernel':>10s} {'block checks':>14s} {'alarms':>7s} {'max |rho|/tau':>14s}"
        "   by distribution (max |rho|/tau)"
    )
    for kname, (gemv, vec) in KERNELS.items():
        done = alarms = 0
        worst = 0.0
        worst_by: dict[str, float] = dict.fromkeys(DISTS, 0.0)
        trial = 0
        checkers: list[GpuBlockedChecker] = []
        t0 = time.time()
        while done < target:
            if trial % redraw == 0:  # fresh secret signs
                checkers = [
                    GpuBlockedChecker(
                        W,
                        block,
                        tree_height(W.shape[1], THREADS, vec),
                        checks=checks,
                        generator=gen,
                    )
                    for W in weights
                ]
            dist = DISTS[trial % len(DISTS)]
            items = []
            for W, checker in zip(weights, checkers, strict=True):
                x = activations(dist, W.shape[1], gen)
                items.append((checker, x, gemv(W, x, THREADS)))
            for res in check_many(items):
                alarms += int((~res.passed).sum())
                ratio = float((res.rho.abs() / res.tau).max())
                worst = max(worst, ratio)
                worst_by[dist] = max(worst_by[dist], ratio)
            done += per_trial
            trial += 1
        by = ", ".join(f"{d} {v:.2e}" for d, v in worst_by.items())
        print(
            f"{kname:>10s} {done:14,d} {alarms:7d} {worst:14.2e}"
            f"   {by}   ({time.time() - t0:.0f} s)"
        )


def bit_flips(trials: int, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    m = n = 4096
    W = (torch.randn(m, n, generator=gen) * 0.02).to(torch.bfloat16).cuda()
    height = tree_height(n, THREADS, vec=8)
    print(f"\n== bit flips: {m} x {n}, one flip per block per trial, {trials} trials per cell")
    for block in (64, 128):
        nb = m // block
        for dist in ("gauss", "student-t", "outliers"):
            row = []
            for checks in (1, 4):
                checker = GpuBlockedChecker(W, block, height, checks=checks, generator=gen)
                rates = []
                for bit, _ in BITS:
                    caught = 0
                    for _ in range(trials):
                        x = activations(dist, n, gen)
                        y = gemv_coalesced(W, x, THREADS)
                        offsets = torch.randint(0, block, (nb,), generator=gen).cuda()
                        idx = torch.arange(nb, device="cuda") * block + offsets
                        bits = y.view(torch.int32)
                        bits[idx] ^= -(2**31) if bit == 31 else 1 << bit
                        caught += int((~checker.verdict(x, y)).sum())
                    rates.append(caught / (trials * nb))
                row.append(rates)
            print(f"\n  b = {block}, x ~ {dist}: share of flips detected")
            print(f"    {'bit':>4s} {'kind':>9s} {'K=1':>7s} {'K=4':>7s}")
            for (bit, kind), k1, k4 in zip(BITS, row[0], row[1], strict=True):
                print(f"    {bit:4d} {kind:>9s} {k1:7.1%} {k4:7.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="False-alarm and bit-flip campaign.")
    parser.add_argument("--target", type=int, default=10_000_000, help="block checks per kernel")
    parser.add_argument("--checks", type=int, default=8, help="K for the false-alarm part")
    parser.add_argument("--redraw", type=int, default=100, help="trials between sign redraws")
    parser.add_argument("--flip-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="small smoke-test run")
    args = parser.parse_args()
    if args.quick:
        args.target, args.redraw, args.flip_trials = 20_000, 5, 5

    print(f"device: {device_name()}; torch {torch.__version__}; seed {args.seed}")
    for block in sorted({DEFAULT_BLOCK, 64}, reverse=True):
        false_alarms(block, args.checks, args.target, args.redraw, args.seed)
    bit_flips(args.flip_trials, args.seed)


if __name__ == "__main__":
    main()
