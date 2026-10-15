from pytest import mark

from ir_axioms.tools.pivot import (
    FirstPivotSelection,
    LastPivotSelection,
    MiddlePivotSelection,
    PivotSelection,
    RandomPivotSelection,
)


@mark.parametrize(
    "pivot_selection",
    [
        RandomPivotSelection(),
        FirstPivotSelection(),
        LastPivotSelection(),
        MiddlePivotSelection(),
    ],
)
def test_implements_protocol(pivot_selection: PivotSelection) -> None:
    assert isinstance(pivot_selection, PivotSelection)


@mark.parametrize(
    "pivot_selection",
    [
        RandomPivotSelection(seed=42),
        FirstPivotSelection(),
        LastPivotSelection(),
        MiddlePivotSelection(),
    ],
)
def test_single_vertex_is_selected(pivot_selection: PivotSelection) -> None:
    assert pivot_selection.select_pivot(None, ["a"]) == "a"


def test_first_selects_first_vertex() -> None:
    assert FirstPivotSelection[None, str]().select_pivot(None, ["a", "b", "c"]) == "a"


def test_last_selects_last_vertex() -> None:
    assert LastPivotSelection[None, str]().select_pivot(None, ["a", "b", "c"]) == "c"


def test_middle_selects_middle_vertex_for_odd_length() -> None:
    assert MiddlePivotSelection[None, str]().select_pivot(None, ["a", "b", "c"]) == "b"


def test_middle_selects_lower_middle_vertex_for_even_length() -> None:
    assert (
        MiddlePivotSelection[None, str]().select_pivot(None, ["a", "b", "c", "d"])
        == "b"
    )


def test_random_selects_from_vertices() -> None:
    pivot_selection = RandomPivotSelection[None, str](seed=42)
    vertices = ["a", "b", "c", "d", "e"]

    pivots = [pivot_selection.select_pivot(None, vertices) for _ in range(100)]

    assert set(pivots) <= set(vertices)
    # With 100 draws, every vertex should be selected at least once.
    assert set(pivots) == set(vertices)


def test_random_with_same_seed_is_deterministic() -> None:
    vertices = list(range(100))
    selection1 = RandomPivotSelection[None, int](seed=1)
    selection2 = RandomPivotSelection[None, int](seed=1)

    pivots1 = [selection1.select_pivot(None, vertices) for _ in range(10)]
    pivots2 = [selection2.select_pivot(None, vertices) for _ in range(10)]

    assert pivots1 == pivots2
    # Subsequent draws should not all be the same.
    assert len(set(pivots1)) > 1
