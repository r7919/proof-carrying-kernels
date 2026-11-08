"""Compile and run agent-written slot code, isolated from the evaluator (needs nvcc + GPU).

A variant is a template config plus optional slot code. The slot is CUDA device code that
replaces one or both hooks of the template:

  __device__ float pck_slot_term(float w, float x, int64_t j)   the term added for column j
                                                                (default: the exact w * x)
  __device__ bool  pck_slot_use(float x, int64_t j)             column-major only: read
                                                                column j at all (default: true)

Isolation:
  - ``check_slot`` accepts only ``__device__`` function definitions (an allow-list: no
    file-scope variables, types or templates, no preprocessor in any spelling, no
    literals, inline assembly, kernels, extern or host code) of bounded length.
  - ``compile_variant`` builds one template instantiation as a shared library with nvcc in
    a subprocess with a timeout; results, errors included, are cached by source hash.
  - ``LiveKernel`` runs the library in a worker process with its own CUDA context, shared
    by all kernels of one `CudaBackend` (each task's weights are uploaded once; each
    library is loaded once). A hang (timeout), a crash or a launch error kills the worker;
    that kernel then returns NaN, which every regime rejects, and the next kernel starts
    a fresh worker.

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
NVCC_FLAGS = ("-O3", "--fmad=false", "-std=c++17", "-shared", "-Xcompiler", "-fPIC")

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

# The allow-list: after comments are removed, the slot is only __device__ function
# definitions. Anything else at file scope (a variable, whose initializer runs as host code
# when the library loads; a struct; a template) is rejected, and so are the spellings that
# get tokens past a regex: digraphs and trigraphs (%: is #), backslashes (line splices and
# \u escapes) and string or character literals.
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_SPELLINGS = re.compile(r"%:|<%|%>|<:|:>|\?\?|\\")
_LITERAL = re.compile(r"[\"']")
_QUALIFIERS = r"(?:(?:static|inline|__forceinline__|__noinline__|constexpr)\s+)*"
_HEADER = re.compile(
    rf"{_QUALIFIERS}__device__\s+{_QUALIFIERS}(?:const\s+)?(?:(?:unsigned|signed|long)\s+)*"
    r"\w+[\s*&]+\w+\s*\([^(){};]*\)"
)


def _structure_errors(code: str) -> list[str]:
    """Reasons the slot is not a sequence of __device__ function definitions."""
    stripped = _COMMENT.sub(" ", code)
    errors = []
    if _SPELLINGS.search(stripped):
        errors.append("digraphs, trigraphs and backslashes are not allowed")
    if _LITERAL.search(stripped):
        errors.append("string and character literals are not allowed")
    depth, start, headers = 0, 0, []
    for i, ch in enumerate(stripped):
        if ch == "{":
            if depth == 0:
                headers.append(stripped[start:i])
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                break
            if depth == 0:
                start = i + 1
    if depth != 0:
        return [*errors, "unbalanced braces"]
    rest = stripped[start:].strip().lstrip(";").strip()
    bad = [h for h in headers if not _HEADER.fullmatch(h.strip().lstrip(";").strip())]
    if bad or rest:
        errors.append("only __device__ function definitions are allowed at file scope")
    return errors


def check_slot(code: str, layout: str) -> list[str]:
    """Reasons the slot code is not accepted (empty if it is)."""
    errors = [msg for pattern, msg in _FORBIDDEN if pattern.search(code)]
    errors += [e for e in _structure_errors(code) if e not in errors]
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
    parts = [source, header or "", arch, deps, " ".join(NVCC_FLAGS), "isystem"]
    key = hashlib.sha256("\0".join(parts).encode()).hexdigest()  # flags too
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
            *NVCC_FLAGS,
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


def _worker(conn: Connection) -> None:
    """The shared worker process: owns the CUDA context, the tasks' weights on the GPU and
    every variant library it has been asked to run. Exits after any kernel failure (a
    crashed kernel can poison the context); the parent then starts a fresh worker."""
    import ctypes
    import statistics

    try:
        import torch

        dev = torch.device("cuda")
        l2 = int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 64 << 20))
        flush = torch.empty(2 * max(l2, 1 << 20), dtype=torch.uint8, device=dev)
        torch.cuda.synchronize()
    except Exception as e:  # report and exit; the parent sees a failed kernel
        conn.send(("error", f"worker failed to start: {e}"))
        return
    conn.send(("ready", None))
    weights: dict[int, tuple[Any, Any]] = {}  # key -> (W, W^T)
    libs: dict[str, Any] = {}
    while True:
        msg = conn.recv()
        if msg is None:
            return
        if msg[0] == "weights":
            _, key, w_bits = msg
            bits = torch.from_numpy(np.ascontiguousarray(w_bits).view(np.int16))
            W = bits.to(dev).view(torch.bfloat16)
            weights[key] = (W, None)
            conn.send(("ok", None))
            continue
        _, key, lib, cfg_fields, x_bits, reps = msg
        try:
            cfg = make_config(cfg_fields)
            if lib not in libs:
                fn = ctypes.CDLL(lib).pck_launch
                fn.restype = ctypes.c_int
                fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_longlong] * 2 + [ctypes.c_void_p]
                libs[lib] = fn
            fn = libs[lib]
            W, WT = weights[key]
            m, n = W.shape
            if isinstance(cfg, ColumnConfig) and WT is None:
                WT = W.t().contiguous()
                weights[key] = (W, WT)
            A = WT if isinstance(cfg, ColumnConfig) else W
            y = torch.empty(m, dtype=torch.float32, device=dev)
            splits = cfg.splits if isinstance(cfg, ColumnConfig) else 1
            scratch = torch.empty(max(splits, 1) * m, dtype=torch.float32, device=dev)
            tickets = torch.zeros(-(-m // 32) + 1024, dtype=torch.int32, device=dev)
            xb = torch.from_numpy(np.ascontiguousarray(x_bits).view(np.int16))
            x = xb.to(dev).view(torch.bfloat16)
            outs: list[Any] = []
            kernel_t: list[float] = []
            base_t: list[float] = []
            for _ in range(reps):
                for timed in ("kernel", "cublas"):
                    flush.zero_()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
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


def _weights_key(W: NDArray[np.float32]) -> int:
    """A cheap identity for a weight matrix (its shape and a sample of its rows)."""
    return hash((W.shape, to_bits(W[:: max(1, W.shape[0] // 64)]).tobytes()))


class SharedWorker:
    """One worker process for many kernels: started on first use, restarted after a kernel
    hangs, crashes or fails to launch (which kills it)."""

    def __init__(self) -> None:
        self.process: Any = None
        self.conn: Connection | None = None
        self.loaded: set[int] = set()

    def _start(self) -> Connection:
        if self.conn is not None and self.process is not None and self.process.is_alive():
            return self.conn
        self.stop()
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        self.process = ctx.Process(target=_worker, args=(child,), daemon=True)
        self.process.start()
        child.close()
        self.conn = parent
        status, payload = self._recv(START_TIMEOUT)
        if status != "ready":
            self.stop()
            raise OSError(str(payload))
        return parent

    def _recv(self, timeout: float) -> tuple[str, Any]:
        assert self.conn is not None
        if not self.conn.poll(timeout):
            return "error", f"timed out after {timeout:.0f} s"
        status, payload = self.conn.recv()
        return status, payload

    def run(
        self, lib: Path, cfg: Config, W: NDArray[np.float32], x: NDArray[np.float32], reps: int
    ) -> tuple[str, Any]:
        """One timed call: ("ok", (y, same, ratio)) or ("error", reason)."""
        try:
            conn = self._start()
            key = _weights_key(W)
            if key not in self.loaded:
                conn.send(("weights", key, to_bits(W)))
                status, payload = self._recv(START_TIMEOUT)
                if status != "ok":
                    raise OSError(str(payload))
                self.loaded.add(key)
            fields = json.loads(json.dumps(cfg.as_dict()))
            conn.send(("run", key, str(lib), fields, to_bits(np.asarray(x, np.float32)), reps))
            status, payload = self._recv(CALL_TIMEOUT)
        except (OSError, EOFError) as e:
            status, payload = "error", f"worker died: {e}"
        if status != "ok":
            self.stop()  # killed, so the next kernel gets a fresh context
        return status, payload

    def stop(self) -> None:
        if self.process is not None:
            if self.process.is_alive() and self.conn is not None:
                with contextlib.suppress(OSError, BrokenPipeError):
                    self.conn.send(None)
                self.process.join(2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(5)
        self.process, self.conn = None, None
        self.loaded = set()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.stop()


class LiveKernel:
    """A compiled variant, run on the GPU by a worker process (a `regimes.Kernel`)."""

    def __init__(
        self, lib: Path, cfg: Config, reps: int = REPS, worker: SharedWorker | None = None
    ) -> None:
        self.lib, self.cfg, self.reps = lib, cfg, reps
        self.worker = worker or SharedWorker()
        self._own_worker = worker is None
        self.failure: str | None = None
        self.ratios: list[float] = []

    def __call__(self, W: NDArray[np.float32], x: NDArray[np.float32]) -> NDArray[np.floating]:
        nan = np.full(W.shape[0], np.nan, np.float32)
        if self.failure is not None:
            return nan
        status, payload = self.worker.run(self.lib, self.cfg, W, x, self.reps)
        if status != "ok":
            self.failure = str(payload)
            return nan
        y, same, ratio = payload
        if not same:
            self.failure = "timed runs gave different outputs"
            self.worker.stop()  # its library may keep state: start clean
            return nan
        self.ratios.append(float(ratio))
        return np.asarray(y, np.float32)

    def reset_timing(self) -> None:
        self.ratios = []

    def speedup(self) -> float:
        """Median speedup over cuBLAS of the calls since `reset_timing` (0 if none ran)."""
        return float(np.median(self.ratios)) if self.ratios else 0.0

    def close(self) -> None:
        """Stop the worker if this kernel owns it (a backend's kernels share one)."""
        if self._own_worker:
            self.worker.stop()


class CudaBackend:
    """Builds every variant, slot code or not, as a `LiveKernel` (an `env.Backend`). All
    of a backend's kernels share one worker process and CUDA context."""

    def __init__(self, reps: int = REPS) -> None:
        self.reps = reps
        self.worker = SharedWorker()

    def build(self, cfg: Config, slot_code: str | None) -> tuple[LiveKernel | None, str]:
        lib, message = compile_variant(cfg, slot_code)
        if lib is None:
            return None, message
        return LiveKernel(lib, cfg, self.reps, self.worker), message

    def close(self) -> None:
        self.worker.stop()
