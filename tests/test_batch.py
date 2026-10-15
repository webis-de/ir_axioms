from numpy import array
from pytest import raises

from ir_axioms.axiom import DistanceAxiom, Axiom


def test_preferences() -> None:
    axiom: Axiom[str, int] = DistanceAxiom()

    preferences = axiom.preferences("q1", [3, 1, 0])

    assert preferences.shape == (3, 3)
    assert (
        preferences
        == array(
            [
                [0, 2, 3],
                [-2, 0, 1],
                [-3, -1, 0],
            ]
        )
    ).all()


def test_preferences_empty() -> None:
    axiom: Axiom[str, int] = DistanceAxiom()

    preferences = axiom.preferences("q1", [])

    assert preferences.shape == (0, 0)


def test_batch_preferences() -> None:
    axiom: Axiom[str, int] = DistanceAxiom()

    preferences = axiom.batch_preferences(
        [
            ("q1", [3, 1]),
            ("q2", [1, 4]),
        ]
    )

    assert preferences.shape == (2, 2, 2)
    assert (
        preferences
        == array(
            [
                [
                    [0, 2],
                    [-2, 0],
                ],
                [
                    [0, -3],
                    [3, 0],
                ],
            ]
        )
    ).all()


def test_batch_preferences_matches_preferences() -> None:
    axiom: Axiom[str, int] = DistanceAxiom()

    batch = [
        ("q1", [3, 1, 0]),
        ("q2", [0, 1, 3]),
    ]
    batch_preferences = axiom.batch_preferences(batch)

    for k, (input, outputs) in enumerate(batch):
        assert (batch_preferences[k] == axiom.preferences(input, outputs)).all()


def test_batch_preferences_empty() -> None:
    axiom: Axiom[str, int] = DistanceAxiom()

    preferences = axiom.batch_preferences([])

    assert preferences.shape == (0, 0, 0)


def test_batch_preferences_different_lengths() -> None:
    axiom: Axiom[str, int] = DistanceAxiom()

    with raises(ValueError):
        axiom.batch_preferences(
            [
                ("q1", [3, 1]),
                ("q2", [3]),
            ]
        )
