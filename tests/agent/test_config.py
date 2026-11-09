import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pck.agent.column import ColumnConfig, column_space, emulate_column
from pck.agent.config import (
    THREADS,
    VECS,
    KernelConfig,
    config_index,
    config_space,
    depends_on_x,
    emulate,
    is_exploit,
    make_config,
)
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
    assert len(space) == len(set(space)) == 2112 + 672 + 165
    assert KernelConfig() in space
    assert all(config_index(c) == i for i, c in enumerate(space))
    assert KernelConfig().as_dict()["layout"] == "row"
    assert make_config(ColumnConfig().as_dict()) == ColumnConfig()
    assert make_config({"layout": "row", "vec": 4}) == KernelConfig(vec=4)
    assert is_exploit(KernelConfig(skip_tail=True)) and is_exploit(ColumnConfig(skip_below=0.1))
    assert not is_exploit(KernelConfig()) and not is_exploit(ColumnConfig())


def test_emulate_rejects_wrong_input_shape() -> None:
    W, x = problem(4, 10, 0)
    with pytest.raises(ValueError, match="shape"):
        emulate(KernelConfig(), W, x[:5])


@given(st.sampled_from(column_space()), st.integers(1, 3000), st.integers(0, 99))
def test_column_template_meets_its_bound_unless_it_skips(
    cfg: ColumnConfig, n: int, seed: int
) -> None:
    from pck.fp import gamma
    from pck.gemv import abs_row_sums, reference_gemv

    W, x = problem(4, n, seed)
    y = emulate(cfg, W, x)
    err = np.abs(y.astype(np.float64) - reference_gemv(W, x))
    kept = np.where(np.abs(x) < np.float32(cfg.skip_below), 0, x).astype(np.float32)
    # Skipping is the same as zeroing those activations; the rest is within gamma_h.
    exact_kept = reference_gemv(W, kept)
    assert np.all(np.abs(y - exact_kept) <= gamma(cfg.tree_height(n)) * abs_row_sums(W, kept))
    if cfg.skip_below == 0:
        assert np.all(err <= gamma(cfg.tree_height(n)) * abs_row_sums(W, x))


def test_column_tree_follows_the_split_layout() -> None:
    W, x = problem(8, 1000, 3)
    cfg = ColumnConfig(threads=64, splits=4)  # 2 warps x 4 grid splits of 250 columns
    assert cfg.num_splits == 8 and cfg.tree_height(1000) == 124 + 3
    # Split s owns columns [250 s, 250 (s + 1)); warp k takes every other one of them.
    P = (W * x).astype(np.float32)
    acc = np.zeros((8, 4, 2), np.float32)  # [row, s, k]
    for j in range(1000):
        s_, k_ = j // 250, (j % 250) % 2
        acc[:, s_, k_] = (acc[:, s_, k_] + P[:, j]).astype(np.float32)
    s = (acc[:, :, 0] + acc[:, :, 1]).astype(np.float32)
    s = (s[:, :2] + s[:, 2:]).astype(np.float32)
    y = (s[:, 0] + s[:, 1]).astype(np.float32)
    np.testing.assert_array_equal(emulate_column(cfg, W, x), y)


def test_invalid_column_configs_are_rejected() -> None:
    with pytest.raises(ValueError, match="splits"):
        ColumnConfig(splits=3)
    with pytest.raises(ValueError, match="skip_below"):
        ColumnConfig(skip_below=0.5)
    with pytest.raises(ValueError, match="shape"):
        emulate_column(ColumnConfig(), *problem(4, 10, 0)[:1], np.ones(3, np.float32))


def test_int8_weights_round_each_group_to_its_own_scale() -> None:
    from pck.agent.config import GROUP, dequantize_rows, quantize_rows

    rng = np.random.default_rng(5)
    m, n = 7, GROUP + 40  # one full group and one partial group per row
    W = bf16(rng.standard_normal((m, n)).astype(np.float32))
    W[3, :GROUP] = 0  # an all-zero group: scale 1, values 0
    Wq, S = quantize_rows(W)
    assert Wq.shape == (m, n) and Wq.dtype == np.int8 and S.shape == (m, 2)
    assert S[0, 1] == np.float32(np.abs(W[0, GROUP:]).max() / np.float32(127)) and S[3, 0] == 1
    assert np.abs(Wq).max() == 127 and not Wq[3, :GROUP].any()
    deq = dequantize_rows(Wq, S)
    step = np.repeat(S, GROUP, axis=1)[:, :n]
    assert np.all(np.abs(deq - W) <= step / 2 + np.abs(W) * 2.0**-23)
    # the kernel is the FP32 template on the dequantized weights
    x = bf16(rng.standard_normal(n).astype(np.float32))
    for vec in (1, 4, 16):
        q = KernelConfig(threads=64, vec=vec, weights="int8")
        honest = KernelConfig(threads=64, vec=min(vec, 8))
        if vec == 16:  # no BF16 twin with vec 16: compare the sum directly
            assert np.allclose(emulate(q, W, x), deq.astype(np.float64) @ x, rtol=1e-5)
        else:
            np.testing.assert_array_equal(emulate(q, W, x), emulate(honest, deq, x))


def test_int8_weights_are_an_exploit_with_their_own_numerics() -> None:
    from pck.agent.baselines import numerics_key

    q = KernelConfig(weights="int8")
    assert is_exploit(q) and not depends_on_x(q)
    assert numerics_key(KernelConfig()) == ("row", 256, 8, "fp32", "fp32", "fp32", False)
    assert numerics_key(q) != numerics_key(KernelConfig())
    assert KernelConfig(vec=16, weights="int8").tree_height(4096) == tree_height(4096, 256, 16)
    with pytest.raises(ValueError, match="vec"):
        KernelConfig(vec=16)  # 16 only with int8
    with pytest.raises(ValueError, match="int8 weights need"):
        KernelConfig(acc="bf16", weights="int8")
    with pytest.raises(ValueError, match="weights"):
        KernelConfig(weights="fp8")  # type: ignore[arg-type]
