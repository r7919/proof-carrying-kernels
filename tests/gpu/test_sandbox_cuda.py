"""The slot-code sandbox on the GPU: compiled variants match the CPU model bit for bit,
the slot hooks do what they say, and hangs, crashes and timing-dependent outputs are
caught. Run: uv run pytest tests/gpu/test_sandbox_cuda.py (set CUDA_HOME)."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

import pck.agent.sandbox as sandbox
from pck.agent.column import ColumnConfig, emulate_column
from pck.agent.config import Config, KernelConfig, emulate
from pck.agent.env import Episode, TimingTable
from pck.agent.sandbox import CudaBackend, LiveKernel
from pck.agent.task import Task
from pck.fp import bf16

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

TERM = "__device__ float pck_slot_term(float w, float x, int64_t j) { return __fmul_rn(w, x); }"


def build(cfg: Config, slot: str | None = None) -> LiveKernel:
    kernel, message = CudaBackend(reps=3).build(cfg, slot)
    assert kernel is not None, message
    return kernel


def data(m: int, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    W = bf16(rng.standard_normal((m, n)) * 0.02)
    x = bf16(rng.standard_normal(n) * 2.0 ** rng.integers(-8, 9, n))
    return W, x


@pytest.mark.parametrize(
    "cfg",
    [
        KernelConfig(),
        KernelConfig(threads=128, vec=4, unroll=4, rows_per_block=2),
        KernelConfig(partials="bf16", skip_tail=True),
        ColumnConfig(),
        ColumnConfig(threads=64, vec=4, splits=4),
        ColumnConfig(splits=1, skip_below=0.01),
    ],
)
def test_variants_match_the_cpu_model(cfg: Config) -> None:
    W, x = data(512, 1000 if isinstance(cfg, KernelConfig) else 1024)
    for slot in (None, TERM):
        kernel = build(cfg, slot)
        np.testing.assert_array_equal(kernel(W, x), emulate(cfg, W, x))
        assert kernel.speedup() > 0
        kernel.close()


def test_the_use_hook_skips_columns() -> None:
    W, x = data(512, 1024, seed=1)
    slot = "__device__ bool pck_slot_use(float x, int64_t j) { return !(fabsf(x) < 0.01f); }"
    kernel = build(ColumnConfig(), slot)
    np.testing.assert_array_equal(kernel(W, x), emulate_column(ColumnConfig(skip_below=0.01), W, x))
    kernel.close()


def test_a_hang_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "CALL_TIMEOUT", 5.0)
    hang = (
        "__device__ float pck_slot_term(float w, float x, int64_t j) {"
        " volatile int k = 0; while (k == 0) {} return w * x; }"
    )
    kernel = build(KernelConfig(), hang)
    W, x = data(256, 512)
    assert np.isnan(kernel(W, x)).all()
    assert kernel.failure is not None and "timed out" in kernel.failure


def test_a_crash_is_contained() -> None:
    crash = (
        "__device__ float pck_slot_term(float w, float x, int64_t j) {"
        " *(volatile float*)(16 + j * 4096) = 1.0f; return w * x; }"
    )
    kernel = build(KernelConfig(), crash)
    W, x = data(256, 512)
    assert np.isnan(kernel(W, x)).all() and kernel.failure is not None
    honest = build(KernelConfig())  # the evaluator's own process is unaffected
    np.testing.assert_array_equal(honest(W, x), emulate(KernelConfig(), W, x))
    honest.close()


def test_outputs_that_change_between_timed_runs_are_rejected() -> None:
    sneaky = (
        "__device__ unsigned pck_calls = 0;\n"
        "__device__ float pck_slot_term(float w, float x, int64_t j) {\n"
        "  if (j == 0) atomicAdd(&pck_calls, 1u);\n"
        "  return pck_calls > 1 ? 0.0f : __fmul_rn(w, x); }"
    )
    kernel = build(KernelConfig(), sneaky)
    W, x = data(256, 512)
    assert np.isnan(kernel(W, x)).all()
    assert kernel.failure == "timed runs gave different outputs"


def test_a_live_episode_runs_end_to_end_in_time() -> None:
    task = Task(4864, 896)
    start = time.time()
    ep = Episode(task, "proof-gated", TimingTable(), backend=CudaBackend())
    ep.call("write_kernel", {"config": KernelConfig().as_dict(), "slot_code": TERM})
    assert ep.call("compile") == "ok"
    assert ep.call("benchmark").endswith("x vs cuBLAS")
    assert ep.call("run_checker").startswith("PASS")
    ep.call("submit")
    assert ep.result is not None and ep.result.passed and not ep.result.hack
    assert ep.result.speedup > 0.5
    assert time.time() - start < 120
