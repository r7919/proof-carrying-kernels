"""Check every token vLLM samples: a vLLM general plugin (off unless enabled).

vLLM loads entry points of the ``vllm.general_plugins`` group in every process; this one
(`register`) does nothing unless ``PCK_CHECK_SAMPLING=1``. Then it wraps the sampler of
vLLM's V2 model runner (``vllm.v1.worker.gpu.sample.sampler.Sampler._sample_random``,
vLLM 0.33): after each sampling step it checks every sampled token against its row of
logits and the request's top-k / top-p / greedy setting (`pck.member.check_members`,
proven in lean/PCK/TopMember.lean), on the GPU, accumulating counts on the device so the
decode loop never synchronizes. Every ``PCK_CHECK_EVERY`` steps it appends a JSON line
of running totals to ``PCK_CHECK_LOG``.

Environment:
  PCK_CHECK_SAMPLING=1   enable the check
  PCK_CHECK_LOG          JSON-lines file (default: pck_sampling_check.jsonl)
  PCK_CHECK_EVERY        steps between lines (default 16)
  PCK_CHECK_TOL          the sampler's own top-p tolerance (default 1e-4)
  PCK_FAULT_RATE         fault injection: replace this fraction of the sampled tokens of
                         constrained rows (top-k, top-p or greedy) with the row's lowest-
                         logit token, which no correct sampler returns (default 0)

Install into vLLM's environment:  uv tool install vllm --with-editable <this repo>
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from pck.member import check_members


class CheckStats:
    """Running totals, kept as device tensors until written."""

    FIELDS = ("rows", "rejected", "injected", "caught", "false_alarms")

    def __init__(self, path: Path, every: int) -> None:
        self.path, self.every = path, every
        self.steps = 0
        self.totals: Any = None
        self.max_excess: Any = None
        self.start = time.time()

    def add(self, ok: Any, injected: Any, excess: Any) -> None:
        import torch

        rejected = ~ok
        counts = torch.stack(
            [
                torch.ones_like(ok).sum(),
                rejected.sum(),
                injected.sum(),
                (rejected & injected).sum(),
                (rejected & ~injected).sum(),
            ]
        )
        step_excess = excess.max() if excess.numel() else excess.new_tensor(-1.0)
        if self.totals is None:
            self.totals, self.max_excess = counts, step_excess
        else:
            self.totals = self.totals + counts
            self.max_excess = torch.maximum(self.max_excess, step_excess)
        self.steps += 1
        if self.steps % self.every == 0:
            self.write()

    def write(self) -> None:
        if self.totals is None:
            return
        row: dict[str, float] = dict(
            zip(self.FIELDS, (int(v) for v in self.totals.tolist()), strict=True)
        )
        row |= {
            "steps": self.steps,
            "max_excess": float(self.max_excess),  # max r - p over accepted top-p rows
            "seconds": round(time.time() - self.start, 1),
        }
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def install(
    sampler_cls: Any, log: Path, every: int = 16, tol: float = 1e-4, fault_rate: float = 0.0
) -> CheckStats:
    """Wrap ``sampler_cls._sample_random`` (vLLM's V2 sampler, or a stand-in in tests)."""
    import torch

    original = sampler_cls._sample_random
    stats = CheckStats(log, every)

    def checked(
        self: Any,
        processed_logits: Any,
        expanded_idx_mapping: Any,
        idx_mapping_np: Any,
        pos: Any,
        top_k: Any,
        top_p: Any,
        has_topp_only_rows: bool,
        use_fused_sampler: bool,
    ) -> tuple[Any, Any]:
        logits = processed_logits.clone()  # vLLM's non-fused path masks it in place
        sampled, out = original(
            self,
            processed_logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            top_k,
            top_p,
            has_topp_only_rows,
            use_fused_sampler,
        )
        temperature = self.sampling_states.temperature.gpu[expanded_idx_mapping]
        greedy = temperature == 0
        rows, vocab = logits.shape
        constrained = greedy.clone()
        if top_k is not None:
            constrained |= top_k < vocab
        if top_p is not None:
            constrained |= top_p < 1.0
        injected = torch.zeros(rows, dtype=torch.bool, device=logits.device)
        if fault_rate > 0:
            draw = torch.rand(rows, device=logits.device) < fault_rate
            injected = draw & constrained
            worst = logits.argmin(dim=1).to(sampled.dtype)
            sampled = torch.where(injected, worst, sampled.reshape(rows)).reshape(sampled.shape)
        verdict = check_members(logits, sampled.reshape(rows), top_k, top_p, greedy, tol)
        # how close accepted top-p samples come to the boundary: max of r - p
        if top_p is not None:
            in_nucleus = verdict["ok"] & (top_p < 1.0) & ~greedy
            excess = (verdict["mass_above"] - top_p.to(torch.float64))[in_nucleus]
        else:
            excess = logits.new_zeros(0, dtype=torch.float64)
        stats.add(verdict["ok"], injected, excess)
        return sampled, out

    sampler_cls._sample_random = checked
    return stats


def register() -> None:
    """vLLM's plugin entry point: install the check if ``PCK_CHECK_SAMPLING=1``."""
    if os.environ.get("PCK_CHECK_SAMPLING") != "1":
        return
    from vllm.v1.worker.gpu.sample import sampler as vllm_sampler

    install(
        vllm_sampler.Sampler,
        Path(os.environ.get("PCK_CHECK_LOG", "pck_sampling_check.jsonl")),
        every=int(os.environ.get("PCK_CHECK_EVERY", "16")),
        tol=float(os.environ.get("PCK_CHECK_TOL", "1e-4")),
        fault_rate=float(os.environ.get("PCK_FAULT_RATE", "0")),
    )
