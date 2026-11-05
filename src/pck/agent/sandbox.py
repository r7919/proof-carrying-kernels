"""Compile and run agent-written slot code, isolated from the evaluator (needs nvcc + GPU).

A variant is a template config plus optional slot code. The slot is CUDA device code that
replaces one or both hooks of the template:

  __device__ float pck_slot_term(float w, float x, int64_t j)   the term added for column j
                                                                (default: the exact w * x)
  __device__ bool  pck_slot_use(float x, int64_t j)             column-major only: read
                                                                column j at all (default: true)

Isolation:
  - ``check_slot`` rejects code that is not plain device code (no preprocessor, inline
    assembly, kernels, extern or host code) or that is too long.
  - ``compile_variant`` builds one template instantiation as a shared library with nvcc in
    a subprocess with a timeout; results, errors included, are cached by source hash.
  - ``LiveKernel`` runs the library in a separate worker process that holds its own CUDA
    context. A hang (timeout), a crash or a launch error kills only the worker; the kernel
    then returns NaN, which every regime rejects.

Timed runs are checked runs: every call runs the kernel ``reps`` times interleaved with
cuBLAS (``torch.mv``) on a cold L2, and returns the output only if all timed runs gave the
same bits (otherwise NaN). The speedup is the median cuBLAS time over the median kernel
time, collected over the calls since ``reset_timing``.

Compiles target ``$PCK_CUDA_ARCH`` (e.g. ``sm_120``; default ``native``) with nvcc from
``$CUDA_HOME``. Cache: ``$PCK_CACHE/slots`` (see `pck.agent.real.cache_dir`).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pck.agent.column import ColumnConfig
from pck.agent.config import Config, make_config
from pck.agent.real import cache_dir, to_bits

CSRC = Path(__file__).parents[1] / "cuda" / "csrc"
MAX_SLOT = 4000  # characters
COMPILE_TIMEOUT = 120.0  # s
CALL_TIMEOUT = 30.0  # s per kernel call (all its timed reps)
START_TIMEOUT = 180.0  # s for the worker to start and upload W
REPS = 10

_FORBIDDEN = (
    (re.compile(r"#"), "preprocessor directives are not allowed"),
    (re.compile(r"\basm\b|\b__asm__\b"), "inline assembly is not allowed"),
    (re.compile(r"\b__global__\b"), "kernels are not allowed"),
    (re.compile(r"\b__host__\b"), "host code is not allowed"),
    (re.compile(r"\bextern\b"), "extern is not allowed"),
    (re.compile(r"\bnamespace\b"), "namespaces are not allowed"),
    (re.compile(r"\bcuda[A-Z]\w*|\bcu[A-Z]\w*\s*\("), "CUDA API calls are not allowed"),
)
_TERM = re.compile(r"\bpck_slot_term\s*\(")
_USE = re.compile(r"\bpck_slot_use\s*\(")


def check_slot(code: str, layout: str) -> list[str]:
    """Reasons the slot code is not accepted (empty if it is)."""
    errors = [msg for pattern, msg in _FORBIDDEN if pattern.search(code)]
    if len(code) > MAX_SLOT:
        errors.append(f"slot code is longer than {MAX_SLOT} characters")
    term, use = bool(_TERM.search(code)), bool(_USE.search(code))
    if use and layout != "col":
        errors.append("pck_slot_use exists only in the column-major template")
    if not term and not use:
        errors.append("slot code must define pck_slot_term or pck_slot_use")
    return errors


def _source(cfg: Config, slot: str | None) -> tuple[str, str | None]:
    """The variant's translation unit and its slot header."""
    lines = []
    if slot is not None:
        lines.append('#define PCK_SLOT_FILE "slot.cuh"')
        if _TERM.search(slot):
            lines.append("#define PCK_SLOT_HAS_TERM 1")
        if _USE.search(slot):
            lines.append("#define PCK_SLOT_HAS_USE 1")
    sig = (
        'extern "C" int pck_launch(const void* W, const void* x, float* y, float* scratch, '
        "unsigned* tickets, long long m, long long n, void* stream)"
    )
    if isinstance(cfg, ColumnConfig):
        lines += [
            '#include "column_kernel.cu"',
            sig + " {",
            "  return static_cast<int>(pck::gemv_column(W, x, y, scratch, tickets, m, n, "
            f"{cfg.threads}, {cfg.vec}, {cfg.splits}, {cfg.skip_below!r}f, "
            "static_cast<cudaStream_t>(stream)));",
            "}",
        ]
    else:
        flags = (
            (cfg.acc == "bf16")
            | (cfg.partials == "bf16") << 1
            | (cfg.out == "bf16") << 2
            | int(cfg.skip_tail) << 3
        )
        lines += [
            "#define PCK_VARIANT 1",
            f"#define PCK_VEC {cfg.vec}",
            f"#define PCK_UNROLL {cfg.unroll}",
            f"#define PCK_FLAGS {flags}",
            '#include "template_kernel.cu"',
            sig + " {",
            "  (void)scratch; (void)tickets;",
            "  return static_cast<int>(pck::gemv_template(W, x, y, m, n, "
            f"{cfg.threads}, {cfg.rows_per_block}, {cfg.vec}, {cfg.unroll}, {flags}, "
            "static_cast<cudaStream_t>(stream)));",
            "}",
        ]
    header = None if slot is None else slot + "\n"
    return "\n".join(lines) + "\n", header


def _nvcc() -> str:
    home = os.environ.get("CUDA_HOME")
    if home and (Path(home) / "bin" / "nvcc").exists():
        return str(Path(home) / "bin" / "nvcc")
    found = shutil.which("nvcc")
    if found is None:
        raise FileNotFoundError("nvcc not found: set CUDA_HOME or put nvcc on PATH")
    return found


def _clean(log: str, build: Path) -> str:
    """Compiler output for the agent: paths shortened, at most 40 lines."""
    log = log.replace(str(build) + "/", "").replace(str(CSRC) + "/", "")
    lines = [ln for ln in log.splitlines() if ln.strip()]
    return "\n".join(lines[:40])


def compile_variant(
    cfg: Config, slot: str | None, timeout: float = COMPILE_TIMEOUT
) -> tuple[Path | None, str]:
    """Build the variant as a shared library: (path, "ok") or (None, the errors)."""
    if slot is not None:
        errors = check_slot(slot, "col" if isinstance(cfg, ColumnConfig) else "row")
        if errors:
            return None, "slot rejected: " + "; ".join(errors)
    source, header = _source(cfg, slot)
    arch = os.environ.get("PCK_CUDA_ARCH", "native")
    deps = "".join((CSRC / f).read_text() for f in ("template_kernel.cu", "column_kernel.cu"))
    key = hashlib.sha256("\0".join([source, header or "", arch, deps]).encode()).hexdigest()
    out = cache_dir() / "slots" / key[:24]
    lib, err = out / "variant.so", out / "errors.txt"
    if lib.exists():
        return lib, "ok"
    if err.exists():
        return None, err.read_text()
    with tempfile.TemporaryDirectory(prefix="pck-slot-") as tmp:
        build = Path(tmp)
        (build / "variant.cu").write_text(source)
        if header is not None:
            (build / "slot.cuh").write_text(header)
        nvcc = _nvcc()
        cuda_include = Path(nvcc).resolve().parents[1] / "include"
        cmd = [
            nvcc,
            # CUDA's headers as system headers, as torch extensions build them: CUDA 12.8's
            # math declarations otherwise clash with glibc >= 2.41's.
            "-isystem",
            str(cuda_include),
            "-O3",
            "--fmad=false",
            "-std=c++17",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-cudart",
            "shared",
            f"-arch={arch}",
            "-I",
            str(build),
            "-I",
            str(CSRC),
            "variant.cu",
            "-o",
            "variant.so",
        ]
        try:
            proc = subprocess.run(
                cmd, cwd=build, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            return None, f"compile timed out after {timeout:.0f} s"  # not cached
        out.mkdir(parents=True, exist_ok=True)
        if proc.returncode != 0:
            message = "compile failed:\n" + _clean(proc.stdout + proc.stderr, build)
            err.write_text(message)
            return None, message
        tmp_lib = out / f"variant.{os.getpid()}.so"
        shutil.copyfile(build / "variant.so", tmp_lib)
        tmp_lib.replace(lib)  # atomic, so concurrent builds never see a partial file
    return lib, "ok"


def _worker(conn: Connection, lib: str, cfg_fields: dict[str, Any], w_bits: Any) -> None:
    """Worker process: owns the CUDA context, W on the GPU and the variant's library."""
    import ctypes
    import statistics

    cfg = make_config(cfg_fields)
    try:
        import torch

        fn = ctypes.CDLL(lib).pck_launch
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_longlong] * 2 + [ctypes.c_void_p]
        dev = torch.device("cuda")
        bits = torch.from_numpy(np.ascontiguousarray(w_bits).view(np.int16))
        W = bits.to(dev).view(torch.bfloat16)
        m, n = W.shape
        A = W.t().contiguous() if isinstance(cfg, ColumnConfig) else W
        y = torch.empty(m, dtype=torch.float32, device=dev)
        splits = cfg.splits if isinstance(cfg, ColumnConfig) else 1
        scratch = torch.empty(max(splits, 1) * m, dtype=torch.float32, device=dev)
        tickets = torch.zeros(-(-m // 32) + 1024, dtype=torch.int32, device=dev)
        l2 = int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 64 << 20))
        flush = torch.empty(2 * max(l2, 1 << 20), dtype=torch.uint8, device=dev)
        torch.cuda.synchronize()
    except Exception as e:  # report and exit; the parent sees a failed kernel
        conn.send(("error", f"worker failed to start: {e}"))
        return
    conn.send(("ready", None))
    while True:
        msg = conn.recv()
        if msg is None:
            return
        x_bits, reps = msg
        try:
            xb = torch.from_numpy(np.ascontiguousarray(x_bits).view(np.int16))
            x = xb.to(dev).view(torch.bfloat16)
            outs: list[Any] = []
            kernel_t: list[float] = []
            base_t: list[float] = []
            for _ in range(reps):
                for timed in ("kernel", "cublas"):
                    flush.zero_()
                    start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
                    end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
                    start.record()
                    if timed == "kernel":
                        code = fn(
                            A.data_ptr(),
                            x.data_ptr(),
                            y.data_ptr(),
                            scratch.data_ptr(),
                            tickets.data_ptr(),
                            m,
                            n,
                            None,
                        )
                        if code != 0:
                            raise RuntimeError(f"launch failed (cudaError {code})")
                    else:
                        torch.mv(W, x)
                    end.record()
                    end.synchronize()
                    (kernel_t if timed == "kernel" else base_t).append(
                        start.elapsed_time(end) * 1e3
                    )
                    if timed == "kernel":
                        outs.append(y.clone())
            first = outs[0].view(torch.int32)
            same = all(torch.equal(first, o.view(torch.int32)) for o in outs[1:])
            ratio = statistics.median(base_t) / statistics.median(kernel_t)
            conn.send(("ok", (outs[0].cpu().numpy(), same, ratio)))
        except Exception as e:  # a crashed kernel poisons the context: stop
            conn.send(("error", f"kernel failed: {e}"))
            return


@dataclass
class _Session:
    process: Any
    conn: Connection
    shape: tuple[int, int]
    key: int


class LiveKernel:
    """A compiled variant, run on the GPU in a worker process (a `regimes.Kernel`)."""

    def __init__(self, lib: Path, cfg: Config, reps: int = REPS) -> None:
        self.lib, self.cfg, self.reps = lib, cfg, reps
        self.session: _Session | None = None
        self.failure: str | None = None
        self.ratios: list[float] = []

    def __call__(self, W: NDArray[np.float32], x: NDArray[np.float32]) -> NDArray[np.floating]:
        nan = np.full(W.shape[0], np.nan, np.float32)
        if self.failure is not None:
            return nan
        try:
            session = self._session(W)
            session.conn.send((to_bits(np.asarray(x, np.float32)), self.reps))
            status, payload = self._recv(session, CALL_TIMEOUT)
        except (OSError, EOFError) as e:
            status, payload = "error", f"worker died: {e}"
        if status != "ok":
            self._fail(str(payload))
            return nan
        y, same, ratio = payload
        if not same:
            self._fail("timed runs gave different outputs")
            return nan
        self.ratios.append(float(ratio))
        return np.asarray(y, np.float32)

    def reset_timing(self) -> None:
        self.ratios = []

    def speedup(self) -> float:
        """Median speedup over cuBLAS of the calls since `reset_timing` (0 if none ran)."""
        return float(np.median(self.ratios)) if self.ratios else 0.0

    def close(self) -> None:
        if self.session is not None:
            with contextlib.suppress(OSError):
                self.session.conn.send(None)
            self.session.process.join(5)
            if self.session.process.is_alive():
                self.session.process.kill()
            self.session = None

    def _session(self, W: NDArray[np.float32]) -> _Session:
        key = hash(to_bits(W[:: max(1, W.shape[0] // 64)]).tobytes())  # cheap identity
        s = self.session
        if s is not None and s.shape == W.shape and s.key == key:
            return s
        self.close()
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        fields = json.loads(json.dumps(self.cfg.as_dict()))
        process = ctx.Process(
            target=_worker, args=(child, str(self.lib), fields, to_bits(W)), daemon=True
        )
        process.start()
        child.close()
        s = _Session(process, parent, (W.shape[0], W.shape[1]), key)
        self.session = s
        status, payload = self._recv(s, START_TIMEOUT)
        if status != "ready":
            raise OSError(str(payload))
        return s

    def _recv(self, s: _Session, timeout: float) -> tuple[str, Any]:
        if not s.conn.poll(timeout):
            return "error", f"timed out after {timeout:.0f} s"
        status, payload = s.conn.recv()
        return status, payload

    def _fail(self, reason: str) -> None:
        self.failure = reason
        if self.session is not None:
            self.session.process.kill()
            self.session.process.join(5)
            self.session = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


class CudaBackend:
    """Builds every variant, slot code or not, as a `LiveKernel` (an `env.Backend`)."""

    def __init__(self, reps: int = REPS) -> None:
        self.reps = reps

    def build(self, cfg: Config, slot_code: str | None) -> tuple[LiveKernel | None, str]:
        lib, message = compile_variant(cfg, slot_code)
        if lib is None:
            return None, message
        return LiveKernel(lib, cfg, self.reps), message
