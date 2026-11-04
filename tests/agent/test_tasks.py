from pck.agent.spec import Spec
from pck.agent.tasks import ALL, HELD_OUT, TRAIN, TRAIN_LOCKED, shapes


def test_split_is_disjoint_and_every_task_fits_the_checker() -> None:
    assert not set(shapes(TRAIN)) & set(shapes(HELD_OUT))
    assert all(t.m % Spec().block == 0 for t in ALL)
    assert len({(t.m, t.n, t.dist, t.seed) for t in ALL}) == len(ALL)


def test_held_out_has_shapes_where_skip_tail_drops_work() -> None:
    assert TRAIN_LOCKED == {"skip_tail": False}
    assert any(t.n % 4096 for t in HELD_OUT)
    assert {t.dist for t in HELD_OUT} - {t.dist for t in TRAIN}
