"""Rollout generation with vLLM, for GRPO (``generator = "vllm"`` in the run config).

vLLM runs as its own server process, in its own environment: it pins a torch version of
its own, so it is not a dependency of this package. Install it once, e.g.
  uv tool install vllm            (then vllm_command = "vllm", the default)
and either let the trainer start it (``vllm_url`` unset) or start it yourself and set
``vllm_url``; it must then run with LoRA enabled and runtime LoRA updates allowed:
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve Qwen/Qwen3-0.6B --enable-lora \\
      --max-lora-rank 16 --gpu-memory-utilization 0.35 --max-model-len 2048 \\
      --max-num-seqs 32 --port 8000

Each GRPO step (`VLLMGenerator.sync`) saves the trainer's LoRA adapter and hot-loads it
into the server under a new name, so every rollout is sampled from the current policy.
Prompts go in as token ids and the sampled token ids come back (with their log-
probabilities under vLLM), so training sees exactly the sampled tokens; the trainer
compares those log-probabilities with its own (the ``mismatch`` metric). Standard
library only (HTTP via urllib).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Generation:
    ids: list[int]
    logps: list[float] | None = None
    """Log-probability of each sampled token under the generator (None for HF)."""


def _post(url: str, body: dict[str, Any], timeout: float) -> Any:
    request = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode()
    return json.loads(raw) if raw.strip().startswith(("{", "[")) else raw


def token_ids(choice: dict[str, Any]) -> tuple[list[int], list[float] | None]:
    """Sampled token ids (and log-probabilities) of one completion choice: from vLLM's
    ``token_ids`` field when present, else from ``logprobs.tokens`` ("token_id:N")."""
    lp = choice.get("logprobs") or {}
    logps = lp.get("token_logprobs")
    if choice.get("token_ids") is not None:
        return [int(t) for t in choice["token_ids"]], logps
    tokens = lp.get("tokens")
    if tokens is None:
        raise ValueError("vLLM returned no token ids: needs return_tokens_as_token_ids")
    return [int(str(t).removeprefix("token_id:")) for t in tokens], logps


class VLLMGenerator:
    """Samples continuations from a vLLM server, with the policy's latest LoRA adapter."""

    def __init__(
        self,
        model: str,
        url: str | None = None,
        *,
        eos_id: int | None = None,
        lora_rank: int = 16,
        gpu_memory: float = 0.35,
        max_model_len: int = 2048,
        max_num_seqs: int = 32,
        command: str = "vllm",
        port: int = 8000,
        start_timeout: float = 600.0,
        request_timeout: float = 600.0,
    ) -> None:
        self.model, self.eos_id = model, eos_id
        self.timeout = request_timeout
        self.process: subprocess.Popen[bytes] | None = None
        self.adapter: str | None = None
        if url is None:
            url = f"http://127.0.0.1:{port}"
            sizes = (gpu_memory, max_model_len, max_num_seqs)
            self.process = self._start(command, port, lora_rank, *sizes)
        self.url = url.rstrip("/")
        self._wait(start_timeout)

    def _start(
        self, command: str, port: int, rank: int, gpu_memory: float, max_len: int, seqs: int
    ) -> subprocess.Popen[bytes]:
        # --max-num-seqs: a GRPO step runs a few dozen sequences at most; vLLM's default
        # (hundreds) captures a CUDA graph per batch size and spends the KV-cache memory.
        cmd = [
            *shlex.split(command),
            "serve",
            self.model,
            "--port",
            str(port),
            "--enable-lora",
            "--max-lora-rank",
            str(rank),
            "--max-loras",
            "2",
            "--gpu-memory-utilization",
            str(gpu_memory),
            "--max-model-len",
            str(max_len),
            "--max-num-seqs",
            str(seqs),
            "--dtype",
            "bfloat16",
        ]
        env = {**os.environ, "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
        return subprocess.Popen(cmd, env=env)

    def _wait(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while True:
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=5):
                    return
            except (urllib.error.URLError, OSError) as e:
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError("the vLLM server exited during startup") from e
                if time.time() > deadline:
                    raise TimeoutError(f"no vLLM server at {self.url}") from e
                time.sleep(2)

    def sync(self, adapter_dir: Path, step: int) -> None:
        """Load the adapter saved in ``adapter_dir`` and use it for every later request."""
        name = f"policy-{step}"
        body = {"lora_name": name, "lora_path": str(Path(adapter_dir).resolve())}
        _post(f"{self.url}/v1/load_lora_adapter", body, self.timeout)
        old, self.adapter = self.adapter, name
        if old is not None:
            _post(f"{self.url}/v1/unload_lora_adapter", {"lora_name": old}, self.timeout)

    def generate(
        self,
        prompts: Sequence[list[int]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> list[Generation]:
        """One sampled continuation per prompt, in order. Identical prompts share one
        request (``n`` samples); distinct ones are sent concurrently."""
        counts = Counter(tuple(p) for p in prompts)
        unique = list(counts)

        def request(prompt: tuple[int, ...]) -> list[Generation]:
            body = {
                "model": self.adapter or self.model,
                "prompt": list(prompt),
                "n": counts[prompt],
                "temperature": temperature,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": max_tokens,
                "logprobs": 0,
                "return_tokens_as_token_ids": True,
                "skip_special_tokens": False,
            }
            if seed is not None:
                body["seed"] = seed
            out = _post(f"{self.url}/v1/completions", body, self.timeout)
            gens = []
            for choice in sorted(out["choices"], key=lambda c: c["index"]):
                ids, logps = token_ids(choice)
                ended = choice.get("finish_reason") == "stop"
                if ended and self.eos_id is not None and (not ids or ids[-1] != self.eos_id):
                    ids = [*ids, self.eos_id]  # the stop token, so the turn ends as sampled
                    logps = None if logps is None else [*logps, 0.0]
                gens.append(Generation(ids, logps))
            return gens

        with ThreadPoolExecutor(max_workers=min(8, len(unique)) or 1) as pool:
            results = dict(zip(unique, pool.map(request, unique), strict=True))
        taken: Counter[tuple[int, ...]] = Counter()
        ordered = []
        for p in prompts:
            key = tuple(p)
            ordered.append(results[key][taken[key]])
            taken[key] += 1
        return ordered

    def close(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(30)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
