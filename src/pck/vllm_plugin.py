"""Check every token vLLM samples: a vLLM general plugin (off unless enabled).

vLLM loads entry points of the ``vllm.general_plugins`` group in every process; this one
(`register`) does nothing unless ``PCK_CHECK_SAMPLING=1``. Then it wraps the sampler of
vLLM's V2 model runner (``vllm.v1.worker.gpu.sample.sampler.Sampler._sample_random``,
vLLM 0.33): after each sampling step it checks every sampled token against its row of
logits and the request's top-k / top-p / greedy setting (`pck.member.check_members`,
proven in lean/PCK/TopMember.lean), on the GPU, accumulating counts on the device and
reading top-k from vLLM's host copy, so the decode loop never synchronizes. Every
``PCK_CHECK_EVERY`` steps it appends a JSON line of running totals (with the process id
and rank) to ``PCK_CHECK_LOG``, one interval late so it never waits for the GPU, and a
final line at exit.

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

import atexit
import json
import os
import time
from pathlib import Path
from typing import Any

from pck.member import check_members


class CheckStats:
    """Running totals, kept as device tensors until written.

    Writing never waits for the GPU: every ``every`` steps the totals are copied to pinned
    host memory without blocking, and that copy is written at the next interval (or at
    `flush`, which `install` registers to run at exit, so the last window is not lost).
    Each line carries the process id and rank, since every vLLM worker that samples
    appends to the same file."""

    FIELDS = ("rows", "rejected", "injected", "caught", "false_alarms")

    def __init__(self, path: Path, every: int) -> None:
        self.path, self.every = path, every
        self.steps = 0
        self.written = 0
        self.totals: Any = None
        self.max_excess: Any = None
        self.pending: tuple[Any, Any, int, float] | None = None
        self.start = time.time()
        self.rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK"))

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
        step_excess = excess.max()
        if self.totals is None:
            self.totals, self.max_excess = counts, step_excess
        else:
            self.totals = self.totals + counts
            self.max_excess = torch.maximum(self.max_excess, step_excess)
        self.steps += 1
        if self.steps % self.every == 0:
            self._write_ready(wait=False)
            self.pending = self._snapshot()
            if self.pending[1] is None:  # on the CPU the copy is already done
                self._write_ready(wait=True)

    def _snapshot(self) -> tuple[Any, Any, int, float]:
        """Start copying the totals to the host; (host tensor, event or None, steps, time)."""
        import torch

        vals = torch.cat([self.totals.to(torch.float64), self.max_excess.reshape(1).double()])
        if not vals.is_cuda:
            return vals.clone(), None, self.steps, time.time()
        host = torch.empty(vals.shape, dtype=vals.dtype, pin_memory=True)
        host.copy_(vals, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        return host, event, self.steps, time.time()

    def _write_ready(self, wait: bool) -> None:
        """Write the pending copy if it has arrived (or, with ``wait``, once it does)."""
        if self.pending is None:
            return
        host, event, steps, now = self.pending
        if event is not None and not event.query():
            if not wait:
                return  # still in flight: the next snapshot replaces it
            event.synchronize()
        values = host.tolist()
        row: dict[str, Any] = dict(zip(self.FIELDS, (int(v) for v in values[:-1]), strict=True))
        row |= {
            "steps": steps,
            "max_excess": values[-1],  # max r - p over accepted top-p rows (-1: none)
            "seconds": round(now - self.start, 1),
            "pid": os.getpid(),
            "rank": self.rank,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.pending, self.written = None, steps

    def write(self) -> None:
        """Write the current totals now (synchronizes; for the end of a run)."""
        if self.totals is not None and self.written != self.steps:
            self.pending = self._snapshot()
            self._write_ready(wait=True)

    def flush(self) -> None:
        """Write anything not yet written: the pending copy, then the current totals."""
        self._write_ready(wait=True)
        self.write()


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
        host = getattr(self.sampling_states, "top_k", None)  # vLLM's host copy: no sync
        top_k_host = host.np[idx_mapping_np] if host is not None and top_k is not None else None
        injected = torch.zeros(rows, dtype=torch.bool, device=logits.device)
        if fault_rate > 0:
            draw = torch.rand(rows, device=logits.device) < fault_rate
            worst = logits.argmin(dim=1).to(sampled.dtype)
            # a fault only where the lowest-logit token is really outside the allowed set
            # (not, e.g., top-p within tol of 1, or ties at the bottom of the top-k)
            legit = check_members(logits, worst, top_k, top_p, greedy, tol, top_k_host)["ok"]
            injected = draw & constrained & ~legit
            sampled = torch.where(injected, worst, sampled.reshape(rows)).reshape(sampled.shape)
        verdict = check_members(
            logits, sampled.reshape(rows), top_k, top_p, greedy, tol, top_k_host
        )
        # how close accepted top-p samples come to the boundary: max of r - p, -1 where
        # there is none (r - p >= -1); a where, not a mask, since a mask synchronizes
        excess = torch.full((rows,), -1.0, dtype=torch.float64, device=logits.device)
        if top_p is not None:
            in_nucleus = verdict["ok"] & (top_p < 1.0) & ~greedy
            excess = torch.where(in_nucleus, verdict["mass_above"] - top_p.double(), excess)
        stats.add(verdict["ok"], injected, excess)
        return sampled, out

    sampler_cls._sample_random = checked
    atexit.register(stats.flush)
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
