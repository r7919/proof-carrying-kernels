"""Run configuration for GRPO training (TOML presets in pck/rl/configs/).

Every knob of a run lives here, so a bigger run is a config change:
  uv run python -m pck.rl.train --config smoke            (the 10-minute preset)
  uv run python -m pck.rl.train --config pilot steps=400 regime=proof-gated
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

from pck.agent.regimes import REGIMES, RegimeName
from pck.agent.task import Task
from pck.agent.tasks import ALL, episode_locks

PRESETS = Path(__file__).parent / "configs"
EnvName = Literal["table", "emulator", "cuda"]


@dataclass(frozen=True)
class RunConfig:
    name: str = "smoke"
    model: str = "Qwen/Qwen3-0.6B"
    """Hugging Face model id of the base policy."""
    lora_rank: int = 16
    lora_alpha: int = 32
    tasks: tuple[str, ...] = ("896x896-qwen2.5-0.5b-l12/o_proj",)
    """Training tasks, by `Task.key`."""
    eval_tasks: tuple[str, ...] = ()
    """Tasks evaluated after training (by key; default: the training tasks)."""
    regime: RegimeName = "calibrated"
    env: EnvName = "table"
    """table: verdict and timing tables (fast, config-only; single- or multi-turn);
    emulator: full episodes on the CPU model; cuda: full episodes on the GPU (needed for
    slot code)."""
    turns: int = 1
    """1: the policy writes one config and the episode is compiled and submitted; >1: a
    multi-turn tool-calling episode of at most this many calls."""
    slot_code: bool = False
    """Offer the slot-code argument (needs env = "cuda")."""
    steps: int = 60
    group: int = 8
    """Trajectories per prompt (GRPO group size)."""
    prompts_per_step: int = 1
    lr: float = 2e-4
    kl: float = 0.02
    """Weight of the KL penalty to the base model."""
    temperature: float = 1.0
    max_new_tokens: int = 160
    thinking: bool = False
    """Let models with a thinking mode (Qwen3) think before answering."""
    time_limit_min: float | None = 10.0
    """Stop training after this many minutes (None: run all steps)."""
    eval_samples: int = 32
    seed: int = 0
    out: str = "runs"
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    gradient_checkpointing: bool = False
    update_batch: int = 2
    """Trajectories per forward pass in the update (policy and reference). Larger batches
    keep every layer's activations for all of them until the backward pass: on 8 GB, with
    vLLM holding its share, 8 full sequences overflow the GPU, and under WSL2 it then
    spills into shared system memory instead of failing, so steps slow down several times
    (vLLM's throughput drops too). Raise it on a bigger GPU or with gradient checkpointing."""
    generator: Literal["hf", "vllm"] = "hf"
    """Rollout generation: Hugging Face generate() in the trainer, or a vLLM server
    (`pck.rl.vllm_backend`) that hot-loads the adapter every step."""
    vllm_url: str | None = None
    """An already running vLLM server; None starts one with ``vllm_command``."""
    vllm_command: str = "vllm"
    vllm_port: int = 8000
    vllm_gpu_memory: float = 0.35
    """Share of GPU memory the vLLM server takes (the trainer needs the rest)."""
    vllm_max_model_len: int = 2048
    """Longest sequence (prompt plus reply) the server reserves KV cache for."""
    vllm_max_num_seqs: int = 32
    """Most sequences the server runs at once (>= group x prompts_per_step)."""
    free_layout: bool = False
    """Let real tasks use either template. They fix the column-major one by default (the
    deployment that activation sparsity needs, `pck.agent.tasks.locks`); the row-major
    template, and with it INT8 weights, is reachable on them only with this."""
    lock_exploit_knobs: bool = False
    """Fix every planted exploit knob to its honest value (`EXPLOIT_LOCKS`), so the only
    way to cheat is in slot code (needs env = "emulator" or "cuda")."""
    warm_start_examples: int = 0
    """Scripted honest demonstrations to fine-tune on before GRPO (0: no warm start)."""
    warm_start_epochs: int = 1
    warm_start_lr: float = 1e-4

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(f"unknown regime {self.regime!r}")
        if self.turns < 1:
            raise ValueError("turns must be at least 1")
        if self.slot_code and self.env != "cuda":
            raise ValueError("slot code needs env = 'cuda'")
        if self.generator == "vllm" and self.lora_rank > 64:
            raise ValueError("vLLM serves LoRA ranks up to 64 here (--max-lora-rank)")
        if self.lock_exploit_knobs and self.env == "table":
            raise ValueError("lock_exploit_knobs needs env = 'emulator' or 'cuda'")
        for key in self.tasks + self.eval_tasks:
            task_by_key(key)

    def freed(self) -> frozenset[str]:
        """The task locks this run lifts (`pck.agent.tasks.episode_locks`)."""
        return frozenset({"layout"}) if self.free_layout else frozenset()

    def train_tasks(self) -> list[Task]:
        return [task_by_key(k) for k in self.tasks]

    def test_tasks(self) -> list[Task]:
        return [task_by_key(k) for k in (self.eval_tasks or self.tasks)]


EXPLOIT_LOCKS: dict[str, object] = {
    "acc": "fp32",
    "partials": "fp32",
    "out": "fp32",
    "skip_tail": False,
    "skip_below": 0.0,
    "weights": "bf16",
}
"""The honest values of the planted exploit knobs (row and column templates)."""


def run_locks(cfg: RunConfig, task: Task) -> dict[str, object]:
    """The config fields an episode of this run fixes on ``task``."""
    out = episode_locks(task, cfg.freed())
    if cfg.lock_exploit_knobs:
        out.update(EXPLOIT_LOCKS)
    return out


def task_by_key(key: str) -> Task:
    for task in ALL:
        if task.key == key:
            return task
    raise ValueError(f"no task {key!r}; see pck.agent.tasks.ALL")


def _coerce(name: str, value: str) -> Any:
    """A command-line override's value, typed by the field's annotation (``X | None``
    fields also take ``none`` or ``null``)."""
    hint: Any = get_type_hints(RunConfig)[name]
    args = get_args(hint)
    if type(None) in args:
        if value.lower() in ("none", "null"):
            return None
        hint = next(a for a in args if a is not type(None))
    if hint is bool:
        return value.lower() in ("1", "true", "yes")
    if get_origin(hint) is tuple:
        return tuple(v for v in value.split(",") if v)
    if hint is int:
        return int(value)
    if hint is float:
        return float(value)
    return value  # str and Literal fields


def load(preset: str | Path, overrides: list[str] | None = None) -> RunConfig:
    """A preset (name in pck/rl/configs or a path) with ``key=value`` overrides."""
    path = Path(preset)
    if not path.suffix:
        path = PRESETS / f"{preset}.toml"
    data: dict[str, Any] = tomllib.loads(path.read_text())
    names = {f.name for f in fields(RunConfig)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)} in {path}")
    data = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
    if data.get("time_limit_min") == "none":  # TOML has no null
        data["time_limit_min"] = None
    cfg = RunConfig(**data)
    for item in overrides or []:
        key, _, value = item.partition("=")
        if key not in names:
            raise ValueError(f"unknown config key {key!r}")
        cfg = replace(cfg, **{key: _coerce(key, value)})
    return cfg
