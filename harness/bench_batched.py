"""Is a precision shortcut faster at batch > 1? A diagnostic benchmark (needs a GPU).

At batch 1 a BF16 GEMV is bandwidth-bound, and the template sweep found no speedup from
rounding to BF16 (harness/sweep_template_out.txt). With B activation vectors per weight
load the kernel does B times the arithmetic per byte, so the multiply-accumulate may start
to matter. This times three inner loops (harness/csrc/batched_gemv.cu) at B = 1, 2, 4, 8:
FP32 multiply then add (the template's), FP32 FMA (bitwise the same result) and packed
BF16 FMA (two lanes per instruction, the shortcut), against cuBLAS (torch.mm). It also
reports the share of outputs the shortcut gets within atol = rtol = 1e-2 of FP64, and how
many rows break the proven bound gamma_h * sum |W x|.

Run: uv run python harness/bench_batched.py
"""

from __future__ import annotations

import functools
from functools import partial
from pathlib import Path
from typing import Any

import torch

from pck.cuda.timing import ColdL2Timer, device_name
from pck.fp import gamma, tree_height

SHAPES = [(896, 896), (4096, 4096), (14336, 4096), (4096, 14336)]
BATCHES = [1, 2, 4, 8]
MODES = {0: "fp32 mul+add", 1: "fp32 fma", 2: "bf16x2 fma"}
THREADS = 256


@functools.cache
def variants() -> Any:
    from torch.utils.cpp_extension import load

    csrc = Path(__file__).parent / "csrc"
    return load(
        name="pck_batched_variants",
        sources=[str(csrc / "batched_gemv.cpp"), str(csrc / "batched_gemv.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--fmad=false"],
    )


def main() -> None:
    timer = ColdL2Timer(10, 100)
    torch.manual_seed(0)
    print(f"device: {device_name()}; {THREADS} threads, times in us (median)")
    print(
        f"{'shape':>11s} {'B':>2s} {'cuBLAS':>8s}"
        + "".join(f"{v:>14s}" for v in MODES.values())
        + f"{'bf16 in 1e-2':>14s}{'bf16 > bound':>14s}"
    )
    for m, n in SHAPES:
        W = (torch.randn(m, n, device="cuda") * 0.02).to(torch.bfloat16)
        h = tree_height(n, THREADS, vec=8)
        for B in BATCHES:
            X = torch.randn(B, n, device="cuda").to(torch.bfloat16)
            cublas = timer(partial(torch.mm, X, W.t())).median_ms * 1e3
            times = [
                timer(partial(variants().run, mode, W, X, THREADS)).median_ms * 1e3
                for mode in MODES
            ]
            ref = X.double() @ W.double().t()
            got = variants().run(2, W, X, THREADS).double()
            err = (got - ref).abs()
            within = (err <= 1e-2 + 1e-2 * ref.abs()).double().mean().item()
            bound = gamma(h) * (X.double().abs() @ W.double().abs().t())
            over = (err > bound).double().mean().item()
            row = "".join(f"{t:14.1f}" for t in times)
            print(f"{m:>5d}x{n:<5d} {B:2d} {cublas:8.1f}{row}{within:14.1%}{over:14.1%}")


if __name__ == "__main__":
    main()
