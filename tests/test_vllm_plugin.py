"""The vLLM plugin against a stand-in sampler with vLLM 0.33's method signature."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pck.vllm_plugin import install, register  # noqa: E402


class FakeSampler:
    """Samples a correct token per row: the argmax (greedy rows), else a uniformly drawn
    token among those vLLM's top-k keeps; masks the logits in place as vLLM's non-fused
    path does."""

    def __init__(self, temperature: Any) -> None:
        self.sampling_states = SimpleNamespace(temperature=SimpleNamespace(gpu=temperature))

    def _sample_random(
        self, logits: Any, idx: Any, idx_np: Any, pos: Any, top_k: Any, top_p: Any, *_: Any
    ) -> tuple[Any, Any]:
        k = top_k if top_k is not None else torch.full((logits.shape[0],), logits.shape[1])
        kth = logits.topk(int(k.max()), dim=1).values.gather(1, (k - 1).reshape(-1, 1))
        logits.masked_fill_(logits < kth, -float("inf"))  # in place, like vLLM
        greedy = self.sampling_states.temperature.gpu[idx] == 0
        noise = torch.rand(logits.shape)
        drawn = torch.where(torch.isfinite(logits), noise, -1.0).argmax(dim=1)
        return torch.where(greedy, logits.argmax(dim=1), drawn), logits


def batch(rows: int = 32, vocab: int = 200) -> tuple[Any, ...]:
    rng = np.random.default_rng(0)
    logits = torch.tensor(rng.standard_normal((rows, vocab)), dtype=torch.float32)
    idx = torch.arange(rows)
    top_k = torch.tensor(rng.choice([5, 20, vocab], rows))
    return logits, idx, idx.numpy(), idx, top_k, None, False, True


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_correct_samples_pass(tmp_path: Path) -> None:
    temperature = torch.tensor([0.0] * 4 + [1.0] * 28)
    cls = type("S", (FakeSampler,), {})
    log = tmp_path / "check.jsonl"
    install(cls, log, every=1)
    for _ in range(3):
        cls(temperature)._sample_random(*batch())
    last = lines(log)[-1]
    assert last["steps"] == 3 and last["rows"] == 96 and last["rejected"] == 0


def test_injected_faults_are_all_caught_and_nothing_else(tmp_path: Path) -> None:
    temperature = torch.ones(32)
    cls = type("S", (FakeSampler,), {})
    log = tmp_path / "check.jsonl"
    install(cls, log, every=1, fault_rate=0.5)
    for _ in range(4):
        cls(temperature)._sample_random(*batch())
    last = lines(log)[-1]
    assert last["injected"] > 0 and last["caught"] == last["injected"]
    assert last["false_alarms"] == 0 and last["rejected"] == last["injected"]


def test_top_p_rows_and_the_host_top_k_copy(tmp_path: Path) -> None:
    """With vLLM's host copy of top-k (no synchronization) and top-p on every row: honest
    samples pass, faults are caught, and the logged boundary distance is below 0."""
    temperature = torch.ones(32)
    cls = type("S", (FakeSampler,), {})
    log = tmp_path / "check.jsonl"
    install(cls, log, every=1, fault_rate=0.3)
    logits, idx, idx_np, pos, top_k, _, *rest = batch()
    # loose enough that every token in the top k passes; none on rows without a top-k
    top_p = torch.where(top_k < logits.shape[1], 0.999, 1.0)
    sampler = cls(temperature)
    sampler.sampling_states.top_k = SimpleNamespace(np=top_k.numpy())
    for _ in range(4):
        sampler._sample_random(logits.clone(), idx, idx_np, pos, top_k, top_p, *rest)
    last = lines(log)[-1]
    assert last["injected"] > 0 and last["caught"] == last["injected"]
    assert last["false_alarms"] == 0 and -1 <= last["max_excess"] < 0


def test_the_plugin_is_off_unless_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PCK_CHECK_SAMPLING", raising=False)
    register()  # must not import vLLM
