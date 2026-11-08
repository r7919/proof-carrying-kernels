"""The GRPO run report's summary of when an exploit took over (harness/grpo_report.py)."""

from __future__ import annotations

from harness.grpo_report import sustained


def rows(*exploit: float) -> list[dict[str, float]]:
    return [{"step": i, "exploit": e} for i, e in enumerate(exploit)]


def test_the_takeover_step_is_where_the_exploit_stays_in_every_submission() -> None:
    assert sustained(rows(0.0, 1.0, 1.0, 1.0)) == 1
    assert sustained(rows(0.0, 1.0, 0.69, 1.0, 1.0)) == 3  # a dip resets it
    assert sustained(rows(1.0, 1.0, 0.5)) is None  # does not end that way
    assert sustained(rows()) is None
