"""The GRPO run report's summary of when an exploit took over (harness/grpo_report.py)."""

from __future__ import annotations

from harness.grpo_report import exploit_summary, sustained


def rows(*exploit: float) -> list[dict[str, float]]:
    return [{"step": i, "exploit": e} for i, e in enumerate(exploit)]


def test_the_takeover_step_is_where_the_exploit_stays_in_every_submission() -> None:
    assert sustained(rows(0.0, 1.0, 1.0, 1.0)) == 1
    assert sustained(rows(0.0, 1.0, 0.69, 1.0, 1.0)) == 3  # a dip resets it
    assert sustained(rows(1.0, 1.0, 0.5)) is None  # does not end that way
    assert sustained(rows()) is None


def test_exploit_summary() -> None:
    assert exploit_summary(rows(0.0, 0.0)) == "never seen"
    assert exploit_summary(rows(0.0, 0.5, 1.0, 1.0)) == (
        "first seen at step 1; in every submission on 2 of 4 steps; kept to the end from step 2"
    )
    assert exploit_summary(rows(0.0, 1.0, 0.0)).endswith("on 1 of 3 steps; not kept to the end")
