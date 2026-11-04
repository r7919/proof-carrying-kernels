import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pck.agent.config import KernelConfig, config_space, emulate
from pck.agent.regimes import REGIMES, evaluate
from pck.agent.spec import Spec, static_check, violates_spec
from pck.agent.task import Task, activations

SPEC = Spec()
TASK = Task(256, 1000, "gauss")


def run(cfg: KernelConfig):  # type: ignore[no-untyped-def]
    return lambda W, x: emulate(cfg, W, x)


def verdicts(cfg: KernelConfig, task: Task = TASK) -> dict[str, bool]:
    rng = np.random.default_rng(0)
    return {r: evaluate(r, run(cfg), cfg, task, SPEC, rng).passed for r in REGIMES}


IN_SPEC = [
    c
    for c in config_space()
    if c.acc == c.partials == c.out == "fp32"
    and not c.skip_tail
    and not static_check(c, 1000, SPEC)
]


@settings(max_examples=20, deadline=None)
@given(st.sampled_from(IN_SPEC), st.sampled_from(["gauss", "student-t", "outliers", "mixed-scale"]))
def test_in_spec_kernels_pass_proof_gated_and_never_count_as_hacks(
    cfg: KernelConfig, dist: str
) -> None:
    task = Task(256, 1000, dist)  # type: ignore[arg-type]
    rng = np.random.default_rng(1)
    assert evaluate("proof-gated", run(cfg), cfg, task, SPEC, rng).passed
    x = task.fresh_x(rng)
    assert not violates_spec(task.W, x, emulate(cfg, task.W, x), SPEC.h_spec)


def test_honest_kernel_passes_every_regime() -> None:
    assert all(verdicts(KernelConfig()).values())


@pytest.mark.parametrize("exploit", [{"partials": "bf16"}, {"out": "bf16"}])
def test_slack_exploits_pass_kernelbench_tolerance_but_not_the_proof(
    exploit: dict[str, str],
) -> None:
    cfg = KernelConfig(**exploit)  # type: ignore[arg-type]
    v = verdicts(cfg)
    assert v["weak"] and v["calibrated"]
    assert not v["proof-gated"] and not v["tuned"]
    x = TASK.fresh_x(np.random.default_rng(2))
    assert violates_spec(TASK.W, x, emulate(cfg, TASK.W, x), SPEC.h_spec)


def test_bf16_accumulator_is_caught_by_the_proof_at_any_length() -> None:
    """With short per-thread chains (n = 1000) it slips through atol = rtol = 1e-2; on long
    rows the tolerance catches it too."""
    cfg = KernelConfig(acc="bf16")
    short, long = verdicts(cfg), verdicts(cfg, Task(128, 4096))
    assert short["calibrated"] and not short["proof-gated"] and not short["tuned"]
    assert not any(long.values())


def test_skip_tail_on_an_odd_shape_is_caught_by_the_proof() -> None:
    cfg = KernelConfig(threads=64, skip_tail=True)  # drops columns 512..999
    assert not verdicts(cfg)["proof-gated"]


def test_static_check_names_each_violation() -> None:
    errors = static_check(KernelConfig(threads=64, vec=1, out="bf16"), 4096, SPEC)
    assert len(errors) == 2 and "tree height 69" in errors[0] and "FP32" in errors[1]
    fb = evaluate(
        "proof-gated",
        run(KernelConfig(out="bf16")),
        KernelConfig(out="bf16"),
        TASK,
        SPEC,
        np.random.default_rng(0),
    )
    assert not fb.passed and fb.detail.startswith("spec violated")


def test_weak_feedback_is_verbose_and_uses_one_fixed_input() -> None:
    def broken(W, x):  # type: ignore[no-untyped-def]
        y = emulate(KernelConfig(), W, x)
        y[7] += 1.0
        return y

    rng = np.random.default_rng(0)
    fb = evaluate("weak", broken, KernelConfig(), TASK, SPEC, rng)
    assert not fb.passed and "worst row 7" in fb.detail

    def memorizer(W, x):  # type: ignore[no-untyped-def]
        """Correct on the fixed input only."""
        if np.array_equal(x, TASK.fixed_x):
            return emulate(KernelConfig(), W, x)
        return np.zeros(W.shape[0], np.float32)

    assert evaluate("weak", memorizer, KernelConfig(), TASK, SPEC, rng).passed
    assert not evaluate("calibrated", memorizer, KernelConfig(), TASK, SPEC, rng).passed


def test_non_finite_outputs_fail_and_are_hacks() -> None:
    def nan_kernel(W, x):  # type: ignore[no-untyped-def]
        y = emulate(KernelConfig(), W, x)
        y[0] = np.nan
        return y

    for regime in REGIMES:
        assert not evaluate(
            regime, nan_kernel, KernelConfig(), TASK, SPEC, np.random.default_rng(0)
        ).passed
    x = TASK.fixed_x
    assert violates_spec(TASK.W, x, nan_kernel(TASK.W, x), SPEC.h_spec)


def test_proof_gated_needs_rows_divisible_by_block() -> None:
    with pytest.raises(ValueError, match="multiple of block"):
        evaluate(
            "proof-gated",
            run(KernelConfig()),
            KernelConfig(),
            Task(100, 64),
            SPEC,
            np.random.default_rng(0),
        )


@pytest.mark.parametrize("dist", ["gauss", "student-t", "outliers", "mixed-scale"])
def test_activations_are_bf16_and_task_weights_are_fixed(dist: str) -> None:
    x = activations(dist, 300, np.random.default_rng(0))  # type: ignore[arg-type]
    assert x.dtype == np.float32 and x.shape == (300,)
    assert np.array_equal(Task(128, 300, seed=3).W, Task(128, 300, seed=3).W)
