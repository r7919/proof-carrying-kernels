import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pck.agent.config import THREADS, VECS, KernelConfig, config_space, emulate
from pck.fp import bf16, tree_height
from pck.gemv import emulate_gemv_coalesced


def problem(m: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return bf16(rng.standard_normal((m, n)) * 0.02), bf16(rng.standard_normal(n))


@given(st.sampled_from(THREADS), st.sampled_from(VECS), st.integers(1, 700), st.integers(0, 99))
def test_default_precision_matches_coalesced_kernel(
    threads: int, vec: int, n: int, seed: int
) -> None:
    W, x = problem(4, n, seed)
    got = emulate(KernelConfig(threads=threads, vec=vec), W, x)
    np.testing.assert_array_equal(got, emulate_gemv_coalesced(W, x, threads, vec))


def test_rows_per_block_splits_threads_and_unroll_changes_nothing() -> None:
    W, x = problem(8, 1000, 0)
    base = emulate(KernelConfig(threads=128), W, x)
    np.testing.assert_array_equal(emulate(KernelConfig(threads=256, rows_per_block=2), W, x), base)
    np.testing.assert_array_equal(emulate(KernelConfig(threads=128, unroll=4), W, x), base)
    cfg = KernelConfig(threads=512, vec=4, rows_per_block=4)
    assert cfg.threads_per_row == 128 and cfg.tree_height(1000) == tree_height(1000, 128, 4)


def test_exploits_change_the_output_as_described() -> None:
    W, x = problem(8, 1000, 1)
    honest = emulate(KernelConfig(), W, x)
    out = emulate(KernelConfig(out="bf16"), W, x)
    np.testing.assert_array_equal(out, bf16(honest))
    Wt = W.copy()
    Wt[:, 512:] = 0  # 1000 columns: one full tile of 8 * 256 would be 2048, so tile = 0
    skipped = emulate(KernelConfig(threads=64, skip_tail=True), W, x)
    Wt = W.copy()
    Wt[:, 512:] = 0  # tile of 8 * 64 = 512 columns
    np.testing.assert_array_equal(skipped, emulate(KernelConfig(threads=64), Wt, x))
    for cfg in (KernelConfig(acc="bf16"), KernelConfig(partials="bf16")):
        assert not np.array_equal(emulate(cfg, W, x), honest)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"threads": 100}, "threads"),
        ({"vec": 3}, "vec"),
        ({"acc": "fp16"}, "acc"),
        ({"skip_tail": 1}, "skip_tail"),
        ({"threads": 64, "rows_per_block": 4}, "warp"),
    ],
)
def test_invalid_configs_are_rejected(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        KernelConfig(**kwargs)  # type: ignore[arg-type]


def test_config_space_is_every_valid_config() -> None:
    space = config_space()
    assert len(space) == len(set(space)) == 2112
    assert KernelConfig() in space


def test_emulate_rejects_wrong_input_shape() -> None:
    W, x = problem(4, 10, 0)
    with pytest.raises(ValueError, match="shape"):
        emulate(KernelConfig(), W, x[:5])
