"""Cost of the trusted sampling checker next to the (untrusted) samplers it checks.

Times, per decode step and on a cold L2, the top-k sampler (torch.topk + multinomial) and
the top-p selection (softmax + sort + cumsum) against the GPU checkers, at batch 1 and 8
with a 151,936-token vocabulary (Qwen-sized).

Run: uv run python harness/bench_sampling.py
"""

from __future__ import annotations

import torch

from pck.cuda import check_topk_gpu, check_topp_gpu, topk_sample, topp_select
from pck.cuda.timing import ColdL2Timer

VOCAB = 151_936
K, P = 50, 0.9


def main() -> None:
    timer = ColdL2Timer(20, 200)
    print(f"device: {torch.cuda.get_device_name()}; V = {VOCAB}, k = {K}, p = {P}")
    print(f"{'batch':>5s} {'step':>22s} {'median ms':>10s} {'p90 ms':>8s}")
    for batch in (1, 8):
        logits = torch.randn(batch, VOCAB, device="cuda") * 3
        selected, _ = topk_sample(logits, K)
        nucleus, lengths = topp_select(logits, P)
        rows = {
            "top-k sampler": lambda: topk_sample(logits, K),  # noqa: B023
            "top-k checker": lambda: check_topk_gpu(logits, selected, K),  # noqa: B023
            "top-p selection": lambda: topp_select(logits, P),  # noqa: B023
            "top-p checker": lambda: check_topp_gpu(logits, nucleus, lengths, P),  # noqa: B023
        }
        assert check_topk_gpu(logits, selected, K).passed.all()
        assert check_topp_gpu(logits, nucleus, lengths, P, tol=1e-5).passed.all()
        for name, fn in rows.items():
            t = timer(fn)
            print(f"{batch:5d} {name:>22s} {t.median_ms:10.4f} {t.p90_ms:8.4f}")


if __name__ == "__main__":
    main()
