"""The slot-code sandbox without a GPU: slot checks, nvcc builds (skipped without nvcc),
failure handling, and live timing in an episode (with a fake timed kernel)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from pck.agent.column import ColumnConfig
from pck.agent.config import KernelConfig, emulate
from pck.agent.env import Episode, TimingTable
from pck.agent.sandbox import LiveKernel, check_slot, compile_variant
from pck.agent.task import Task

TERM = "__device__ float pck_slot_term(float w, float x, int64_t j) { return w * x; }"
USE = "__device__ bool pck_slot_use(float x, int64_t j) { return fabsf(x) >= 0.01f; }"


def has_nvcc() -> bool:
    home = os.environ.get("CUDA_HOME")
    return shutil.which("nvcc") is not None or bool(home and Path(home, "bin", "nvcc").exists())


needs_nvcc = pytest.mark.skipif(not has_nvcc(), reason="needs nvcc")


@pytest.fixture(autouse=True)
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PCK_CACHE", str(tmp_path))
    monkeypatch.setenv("PCK_CUDA_ARCH", os.environ.get("PCK_CUDA_ARCH", "sm_120"))
    return tmp_path


def test_accepts_plain_device_code() -> None:
    assert check_slot(TERM, "row") == []
    assert check_slot(TERM + "\n" + USE, "col") == []
    assert check_slot(USE, "col") == []
    helper = "static __device__ __forceinline__ float sq(float v) { return v * v; }"
    assert check_slot(f"// a helper\n{helper}\n{TERM}; /* done */", "row") == []


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("#include <cstdio>\n" + TERM, "preprocessor"),
        (TERM + ' asm("trap;");', "assembly"),
        (TERM + " __global__ void k() {}", "kernels"),
        (TERM + " __host__ int h();", "host"),
        (TERM + " extern int e;", "extern"),
        (TERM + " namespace q {}", "namespaces"),
        (TERM + " void f() { cudaMalloc(0, 0); }", "CUDA API"),
        (TERM + " " * 5000, "longer"),
        ("__device__ int helper() { return 1; }", "must define"),
        # file-scope host code: its initializer runs when the worker loads the library
        ('int system(const char*); static int pwn = system("id"); ' + TERM, "file scope"),
        ("int f(); static int pwn = f(); " + TERM, "file scope"),
        ("__device__ int counter;\n" + TERM, "file scope"),
        ("struct S { S() {} }; static S s;\n" + TERM, "file scope"),
        ("template <int N> __device__ float t() { return N; }\n" + TERM, "file scope"),
        # spellings that get tokens past a regex: %: is #, a backslash splices lines
        ("%:define CAT(a,b) a%:%:b\n" + TERM, "digraphs"),
        (TERM + " <% %>", "digraphs"),
        (TERM + " __glo\\\nbal__ void k() {}", "backslashes"),
        (TERM + ' const char* s = "x";', "literals"),
        (TERM + " }", "unbalanced"),
    ],
)
def test_rejects_other_code(code: str, reason: str) -> None:
    assert any(reason in e for e in check_slot(code, "row"))


def test_use_hook_only_in_the_column_template() -> None:
    assert any("column-major" in e for e in check_slot(USE, "row"))


@needs_nvcc
@pytest.mark.parametrize("cfg", [KernelConfig(), ColumnConfig()])
def test_builds_variants_and_caches_them(cfg: KernelConfig | ColumnConfig) -> None:
    slot = USE if isinstance(cfg, ColumnConfig) else TERM
    for code in (None, slot):
        lib, message = compile_variant(cfg, code)
        assert message == "ok" and lib is not None and lib.exists()
        assert compile_variant(cfg, code) == (lib, "ok")  # cached


@needs_nvcc
def test_compiler_errors_reach_the_agent(cache: Path) -> None:
    bad = "__device__ float pck_slot_term(float w, float x, int64_t j) { return w * q; }"
    lib, message = compile_variant(KernelConfig(), bad)
    assert lib is None and message.startswith("compile failed")
    assert "slot.cuh" in message and "q" in message and str(cache) not in message
    assert compile_variant(KernelConfig(), bad) == (None, message)  # cached too


def test_rejected_slots_do_not_compile() -> None:
    lib, message = compile_variant(KernelConfig(), "#define X\n" + TERM)
    assert lib is None and message.startswith("slot rejected")


def test_a_kernel_whose_worker_fails_returns_nan(tmp_path: Path) -> None:
    kernel = LiveKernel(tmp_path / "missing.so", KernelConfig())
    W = np.ones((4, 8), np.float32)
    y = kernel(W, np.ones(8, np.float32))
    assert y.shape == (4,) and np.isnan(y).all()
    assert kernel.failure is not None and kernel.speedup() == 0.0
    assert np.isnan(kernel(W, np.ones(8, np.float32))).all()  # stays failed


class FakeLive:
    """A timed kernel: the CPU model, reporting a fixed live speedup."""

    def __init__(self, speedup: float) -> None:
        self.failure: str | None = None
        self.calls = 0
        self._speedup = speedup

    def __call__(self, W: NDArray[np.float32], x: NDArray[np.float32]) -> NDArray[np.floating]:
        self.calls += 1
        return emulate(KernelConfig(), W, x)

    def reset_timing(self) -> None:
        self.calls = 0

    def speedup(self) -> float:
        return self._speedup if self.calls else 0.0


class FakeBackend:
    def build(self, cfg: object, slot_code: str | None) -> tuple[FakeLive, str]:
        return FakeLive(1.37), "ok"


def test_episodes_pay_the_live_speedup() -> None:
    task = Task(256, 384)
    ep = Episode(task, "proof-gated", TimingTable(), backend=FakeBackend())
    ep.call("write_kernel", {"config": KernelConfig().as_dict(), "slot_code": TERM})
    ep.call("compile")
    assert ep.call("benchmark") == "1.37x vs cuBLAS"  # not from the (empty) table
    assert ep.call("run_checker").startswith("PASS")
    ep.call("submit")
    assert ep.result is not None and ep.result.passed
    assert ep.result.speedup == 1.37 and not ep.result.hack
