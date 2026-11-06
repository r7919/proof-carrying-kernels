"""The vLLM rollout client, against a fake server (no vLLM or GPU needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pck.rl.vllm_backend import VLLMGenerator, token_ids
from tests.rl.fake_vllm import FakeVLLM, serve

EOS = 9


@pytest.mark.parametrize("field", [True, False])
def test_reads_token_ids_either_way(field: bool) -> None:
    fake = FakeVLLM([5, 6, EOS], token_ids_field=field)
    choice = fake.completion({"n": 1})["choices"][0]
    assert token_ids(choice) == ([5, 6, EOS], [-0.5, -0.5, -0.5])
    with pytest.raises(ValueError, match="no token ids"):
        token_ids({"logprobs": {}})


def test_groups_identical_prompts_and_keeps_their_order() -> None:
    fake = FakeVLLM([5, 6, EOS])
    with serve(fake) as url:
        gen = VLLMGenerator("m", url, eos_id=EOS)
        out = gen.generate([[1, 2], [3], [1, 2]], temperature=1.0, max_tokens=8, seed=4)
    assert [g.ids for g in out] == [[5, 6, EOS]] * 3
    assert sorted(r["n"] for r in fake.requests) == [1, 2]  # one request per distinct prompt
    body = fake.requests[0]
    assert body["model"] == "m" and body["top_k"] == -1 and body["return_tokens_as_token_ids"]
    assert body["seed"] == 4 and body["max_tokens"] == 8


def test_appends_a_stop_token_the_server_left_out() -> None:
    fake = FakeVLLM([5, 6, EOS], with_eos=False)
    with serve(fake) as url:
        out = VLLMGenerator("m", url, eos_id=EOS).generate([[1]], temperature=1.0, max_tokens=8)
    assert out[0].ids == [5, 6, EOS] and out[0].logps == [-0.5, -0.5, 0.0]


def test_sync_swaps_adapters(tmp_path: Path) -> None:
    fake = FakeVLLM([EOS])
    with serve(fake) as url:
        gen = VLLMGenerator("m", url, eos_id=EOS)
        gen.sync(tmp_path / "a", 1)
        gen.sync(tmp_path / "b", 2)
        gen.generate([[1]], temperature=1.0, max_tokens=4)
    assert fake.loaded == ["policy-1", "policy-2"] and fake.unloaded == ["policy-1"]
    assert fake.requests[-1]["model"] == "policy-2"  # requests use the newest adapter


def test_a_missing_server_times_out() -> None:
    with pytest.raises(TimeoutError, match="no vLLM server"):
        VLLMGenerator("m", "http://127.0.0.1:9", start_timeout=0.1)
