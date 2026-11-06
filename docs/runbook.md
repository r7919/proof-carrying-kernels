# Runbook

How to set up, regenerate every table, train, scale up and extend the project. For what
the pieces guarantee see [design.md](design.md); for the measured numbers see
[results.md](results.md).

## Setup

| What | Command | Needs |
|---|---|---|
| CPU library, tests, dev tools | `uv sync` | uv |
| Git hooks (lint, types, tests, Lean) | `uv run pre-commit install --hook-type pre-commit --hook-type pre-push` | |
| GPU kernels and tests | `uv sync --extra gpu` | NVIDIA driver, CUDA 12.8 `nvcc` |
| Real activations (captures, tightness) | `uv sync --extra gpu --extra models` | Hugging Face access |
| RL training | `uv sync --extra rl` (includes `gpu` and `models`) | |
| Lean proofs | `cd lean && lake exe cache get && lake build && lake env lean scripts/Axioms.lean` | elan |

Machine-specific notes, all seen on the RTX 5060 laptop (Ubuntu under WSL2):

- **Host compiler.** CUDA 12.8 supports GCC up to 14. If a build fails in a host header,
  install `g++-14` and set `NVCC_CCBIN=g++-14`; nvcc reads it, so this covers the torch
  extension and the sandbox.
- **glibc >= 2.41.** CUDA 12.8's math headers clash with newer glibc ("exception
  specification is incompatible ... cospi"). Torch extensions and the sandbox pass
  `-isystem <cuda>/include`, which avoids it; your own nvcc commands need the same flag.
- **Downloads.** If downloads from PyPI or Hugging Face time out under WSL2, check IPv6
  (`curl -4` works but plain `curl` does not). `UV_HTTP_TIMEOUT=120` helps with slow
  links, and `HF_HUB_DISABLE_XET=1` helps when Hugging Face's Xet transfer fails.
- **Lean in hooks.** The pre-push hook needs `lake` on `PATH` (`export PATH=$HOME/.elan/bin:$PATH`).

Environment variables:

| Variable | Default | Used by |
|---|---|---|
| `PCK_CACHE` | `~/.cache/pck` | captured layers, sandbox builds (`slots/`) |
| `PCK_TABLES` | the repo's `harness/` | RL reward tables (`timing_table.json`, `verdict_table.json`) |
| `PCK_CUDA_ARCH` | `native` | sandbox builds (e.g. `sm_120` on a machine without the GPU) |
| `CUDA_HOME` | nvcc on `PATH` | sandbox builds, torch extensions |
| `TORCH_EXTENSIONS_DIR` | torch's default | where the CUDA extension is built |

## Tests

```bash
uv run pytest                                   # CPU suite (also the pre-push hook)
uv run pytest tests/gpu                         # on a GPU machine
uv run pytest tests/gpu/test_sandbox_cuda.py    # the slot-code sandbox
python3 src/pck/cuda/ptx_audit.py               # PTX audit of the trusted checkers
uv run python -m pck.manifest                   # does the checker match its pin?
```

Tests skip what the machine lacks: GPU tests without CUDA, nvcc builds without nvcc, and
RL tests without the `rl` extra or the Qwen3 tokenizer.

## Regenerating the data

Each experiment writes its output next to it in `harness/`, and
[results.md](results.md) lists them all. Phase 4 and 5 depend on three artifacts, built
in this order:

1. **Captured layers** (GPU or CPU, minutes; not in the repo, cached in `$PCK_CACHE`):
   `uv run python -m harness.capture_layer --layers 12 20`
2. **Timing table** (GPU, about 20 minutes, plugged in): `uv run python harness/sweep_template.py`
   writes `harness/timing_table.json`. `--column-only` re-times only the column-major
   template; `--quick` is a smoke test.
3. **Verdict table** (CPU, several hours, resumable): `uv run python -m harness.verdicts --jobs 4`
   writes `harness/verdict_table.json` after every task and skips the tasks it already
   has. After adding tasks, rerun it to compute only the new ones. After changing a
   regime, the spec or the template's numerics, delete the file and rebuild it.

Then, all on CPU from those tables:

```bash
uv run python -m harness.baselines        # brute force, random search, bandit per regime
uv run python -m harness.hack_anatomy     # what the proof-gated checker lets through
uv run python -m harness.phase4_checks    # scripted cheater and episode wall time
uv run python -m pck.rl.control           # the non-LLM control policy
```

On the GPU, `harness/bench_window.py` re-times the exploit window at 200 interleaved reps.

## Training (Phase 5)

```bash
uv run python -m pck.rl.train --config smoke --compare     # ~10 minutes on the 5060
uv run python -m pck.rl.train --config pilot --compare seed=1
uv run python -m pck.rl.train --config smoke regime=proof-gated steps=200 time_limit_min=none
```

`--compare` trains one adapter under `calibrated` and one under `proof-gated`, the core
comparison. Any `RunConfig` field can be overridden as `key=value`. The presets are in
`src/pck/rl/configs/`, and every field is documented in `src/pck/rl/config.py`.

Each run writes `runs/<name>-<regime>-s<seed>/`:

| File | Contents |
|---|---|
| `config.json` | the full run config |
| `warmstart.jsonl` | warm-start loss per epoch (if enabled) |
| `metrics.jsonl` | per step: reward, valid, passed, exploit and hack rates, mean speedup of passes, tokens, KL, gradient norm |
| `samples.jsonl` | one sampled reply per step |
| `summary.json` | final evaluation per test task |
| `adapter/` | the LoRA adapter (`peft`) |

*Exploit* means the submission passed using a planted exploit knob. *Hack* means it passed
and violates the spec on the hidden suite. A hack is the ground truth, defined the same
way for every regime.

Knobs for a bigger run:

| Knob | Smoke | Bigger | Note |
|---|---|---|---|
| `model` | Qwen3-0.6B | Qwen3-1.7B | any Hugging Face chat model with a chat template |
| `tasks`, `eval_tasks` | one layer | the pilot's lists | keys from `pck.agent.tasks.ALL` |
| `env` | `table` | `emulator` or `cuda` | the table env is single-turn and config-only |
| `turns` | 1 | 6 | multi-turn tool calling needs `env = emulator` or `cuda` |
| `slot_code` | false | true | agent-written CUDA; needs `env = cuda` |
| `warm_start_examples` | 0 | 256 | scripted honest demonstrations before GRPO |
| `steps`, `time_limit_min` | 400, 4.5 | 400, none | |
| `group`, `prompts_per_step` | 8, 2 | 4, 2 | trajectories per step = group x prompts |
| `lora_rank`, `lr`, `kl` | 16, 3e-4, 0.01 | 32, 1e-4, 0.02 | |
| `gradient_checkpointing` | false | true | for longer multi-turn contexts |
| `generator` | hf | vllm | vLLM rollouts; see below |
| `update_batch` | 2 | 2-8 | trajectories per forward pass in the update; see the note on memory |
| `seed` | 0 | 0, 1, 2 | the plan asks for at least 3 seeds per regime |

### The pilot and slot-code runs

```bash
uv run python -m pck.rl.train --config pilot --compare seed=0   # overnight per seed
uv run python -m pck.rl.train --config slot --compare           # ~15 min per regime
```

- **Pilot** (`configs/pilot.toml`): Qwen3-1.7B, multi-turn tool calling (up to 6 calls),
  eight training tasks, evaluated on held-out tasks. Multi-turn episodes are answered from
  the tables (`env = table`, `pck.rl.envs.TableEpisode`): each tool call costs
  microseconds, and a test checks the answers agree with full CPU-model episodes. The
  group's episodes run in lockstep, one batched generation per turn. On the 8 GB GPU it
  generates with Hugging Face, since 1.7B does not fit twice beside a vLLM server. For a
  quick trial: `model=Qwen/Qwen3-0.6B generator=vllm steps=40`.
- **Slot code** (`configs/slot.toml`): the policy may add CUDA (`slot_code`). With
  `lock_exploit_knobs`, every planted exploit knob is fixed to its honest value, so the
  only way to cheat is to write the cheat, e.g. a `pck_slot_use` that skips small
  activations. Kernels are compiled by nvcc and run live (`env = cuda`). One sandbox
  worker serves all of a run's kernels and restarts after a crash. Half the warm-start
  demonstrations show an honest slot (the exact product).
- A run's rollouts and hack labels are in `samples.jsonl` and `metrics.jsonl` as for
  the smoke run. For slot runs, look in `samples.jsonl` for the slot code the policy
  writes.

### Faster rollouts with vLLM

Generation is the slowest part of a step. With `generator=vllm`, rollouts are sampled by
a vLLM server, and the trainer keeps only the update:

```bash
uv tool install vllm                                  # once, in vLLM's own environment
uv run python -m pck.rl.train --config smoke --compare generator=vllm
uv run python -m harness.bench_rollouts               # HF vs vLLM throughput, one step's rollouts
```

- **Its own environment.** vLLM pins its own torch version, so it is not a dependency of
  this package. The trainer starts the server (`vllm_command`, default `vllm`), or uses a
  running one given in `vllm_url`. A server you start yourself needs LoRA enabled and
  runtime LoRA updates allowed (see `src/pck/rl/vllm_backend.py`).
- **On-policy.** After every update the adapter is saved and hot-loaded into the server
  under a new name, so each step samples from the current policy. Sampled token ids come
  back as they are, so training sees exactly the sampled tokens.
- **`mismatch`** in `metrics.jsonl` is the mean absolute difference per token between the
  trainer's and vLLM's log-probability of the sampled tokens. The two engines' numerics
  differ slightly. GRPO here assumes they are equal (importance ratio 1), so a mismatch
  that grows during a run means the samples are drifting off-policy.
- **Memory.** The server takes `vllm_gpu_memory` of the GPU (default 0.35) and the trainer
  the rest. vLLM reserves KV cache for one sequence of `vllm_max_model_len` tokens (default
  2048, enough for single-turn prompts plus replies) and refuses to start if that does
  not fit ("KV cache is needed, which is larger than the available KV cache memory").
  `vllm_max_num_seqs` (default 32) caps concurrent sequences, which also saves the CUDA
  graphs vLLM would otherwise capture for hundreds of batch sizes. On the 8 GB 5060,
  Qwen3-0.6B fits at the defaults. The multi-turn pilot uses 4096 tokens and 0.45 of the GPU; for 1.7B,
  lower `max_new_tokens` or use a bigger GPU. Running `vllm download-kernels` once in
  vLLM's environment saves FlashInfer's kernel compilation at every first start.

Notes:

- **Timing.** With `env = table` or `emulator`, speedups come from the timing table, which
  was measured on the 5060. With `env = cuda`, every kernel is timed live against cuBLAS,
  so keep the GPU free of other work. Generation pauses while the environment runs, so
  the policy does not share the GPU with timing.
- **Throughput.** With the default `generator=hf`, generation is plain Hugging Face
  `generate()`; use `generator=vllm` for longer runs. With vLLM, the update becomes the
  larger part of a step. It runs `update_batch` trajectories per padded forward pass, and
  computes logits only from the first policy token on.
- **GPU memory spills under WSL2.** If steps suddenly take several times longer (and
  vLLM's throughput drops), the trainer and vLLM together have overflowed the GPU, and
  WSL2 is paging into shared system memory instead of failing. Lower `update_batch`,
  enable `gradient_checkpointing`, or give vLLM less memory (`vllm_gpu_memory`). On the
  5060, `update_batch=8` with vLLM did this (24 s per step instead of about 4 s).
- **Validity.** If `valid` stays near 0, the model is not producing parsable configs. Look
  at `samples.jsonl`, raise `max_new_tokens`, or enable the warm start.

## Extending

**A new task.** Add it to `src/pck/agent/tasks.py`:

- training or held-out, with `locks` if it needs fixed fields;
- `h_spec` for a non-default error budget (it gets its own key, `...-h<budget>`);
- `source` for a captured layer.

Then:

1. If its shape (or its activation distribution, for input-dependent configs) is new, run
   the sweep on the GPU.
2. Rerun `harness.verdicts`, which computes only the new tasks.
3. `tests/agent/test_tasks.py` checks the split. Held-out shapes must not appear in
   training, and every row count must be a multiple of the checker block (128).

**A new template knob or planted exploit:**

1. Add the field and its allowed values to `KernelConfig` (`pck/agent/config.py`) or
   `ColumnConfig` (`pck/agent/column.py`), and mark exploits in `is_exploit`.
2. Implement it in the CPU model (`emulate` or `emulate_column`) and in the CUDA template
   (`template_kernel.cu` or `column_kernel.cu`). Pass it through `gemv_template` or
   `gemv_column` (`pck/cuda/gemv.py`) and the sandbox's generated source
   (`pck/agent/sandbox.py`, `_source`).
3. Update `numerics_key` (`pck/agent/baselines.py`) if the knob changes the numerics, and
   the prompt's field list (`pck/rl/prompt.py`).
4. Run `tests/gpu/test_template_cuda.py` and `test_sandbox_cuda.py` (bit-exactness), then
   rerun the sweep and the verdict table. Configs are indexed by position in
   `config_space()`, so both tables must be rebuilt.

**A new feedback regime.** Add it to `REGIMES` and `evaluate` in `pck/agent/regimes.py`,
and to `FRESH` in `pck/agent/baselines.py` if it draws fresh inputs. Then rebuild the
verdict table.

**Changing the trusted checker** (anything in `pck.manifest.TRUSTED`):

1. Make the change, and update the Lean proofs if the threshold or algorithm changes.
2. Run `python3 src/pck/cuda/ptx_audit.py`, `uv run pytest` and `uv run pytest tests/gpu`.
3. Re-pin with `uv run python -m pck.manifest --update` and commit `checker_lock.json` with
   the change. Until then, proof-gated episodes refuse to run and `tests/test_manifest.py`
   fails. Manifests made with the old checker report the mismatch in `problems()`.

**Allowing a new instruction in the PTX audit.** Add it to `ALLOWED` in
`src/pck/cuda/ptx_audit.py`, for that file only. Allow it only once its error is covered
by a stated assumption in [design.md](design.md#assumed-not-proven).

## Known gaps and future work

- **Batch 1-8 and FP8 tasks** (in the plan's task distribution). They need a batched
  template, which can reuse the per-column summation tree so that the CPU model and
  checker apply column by column, plus a GPU sweep. Precision shortcuts were measured not
  to pay up to batch 8 (`harness/bench_batched.py`).
- **Sampling tasks in the agent environment.** The top-k and top-p checkers exist and are
  proven, but no agent task uses them yet.
- **Tensor cores** are outside the spec: their accumulation order and rounding are not
  documented.
- **The checker's resolution.** On layers that fill a single 128-row block, a cheat that
  is off by 15-27x the per-row bound (about 1e-4 of the output scale) can pass
  (`harness/hack_anatomy_out.txt`). A smaller block `b` gives finer detection at more
  traffic.
- **Training scale.** BF16 LoRA (no 4-bit QLoRA) on a single GPU is enough for the 0.6B
  and 1.7B presets, with vLLM for generation. Longer runs would benefit from concurrent
  environment workers and, on several GPUs, a distributed update (e.g. DeepSpeed ZeRO).
- **The plan's 0.6B size baseline** is a config change (`model=Qwen/Qwen3-0.6B` on the
  pilot preset).
