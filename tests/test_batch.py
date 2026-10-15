from numpy import array
from pytest import raises

from ir_axioms.axiom import UniformAxiom, Axiom
from ir_axioms.model import Query, Document


def test_preferences() -> None:
    query = Query(id="q1")
    document1 = Document(id="d1")
    document2 = Document(id="d2")
    document3 = Document(id="d3")

    axiom: Axiom[Query, Document] = UniformAxiom(scalar=2)

    preferences = axiom.preferences(query, [document1, document2, document3])

    assert preferences.shape == (3, 3)
    assert (
        preferences
        == array(
            [
                [0, -2, -2],
                [2, 0, -2],
                [2, 2, 0],
            ]
        )
    ).all()


def test_preferences_empty() -> None:
    query = Query(id="q1")

    axiom: Axiom[Query, Document] = UniformAxiom(scalar=1)

    preferences = axiom.preferences(query, [])

    assert preferences.shape == (0, 0)


def test_batch_preferences() -> None:
    query1 = Query(id="q1")
    query2 = Query(id="q2")
    document1 = Document(id="d1")
    document2 = Document(id="d2")
    document3 = Document(id="d3")

    axiom: Axiom[Query, Document] = UniformAxiom(scalar=1)

    preferences = axiom.batch_preferences(
        [
            (query1, [document1, document2]),
            (query2, [document2, document3]),
        ]
    )

    assert preferences.shape == (2, 2, 2)
    assert (
        preferences
        == array(
            [
                [
                    [0, -1],
                    [1, 0],
                ],
                [
                    [0, -1],
                    [1, 0],
                ],
            ]
        )
    ).all()


def test_batch_preferences_matches_preferences() -> None:
    query1 = Query(id="q1")
    query2 = Query(id="q2")
    document1 = Document(id="d1")
    document2 = Document(id="d2")
    document3 = Document(id="d3")

    axiom: Axiom[Query, Document] = UniformAxiom(scalar=3)

    batch = [
        (query1, [document1, document2, document3]),
        (query2, [document3, document2, document1]),
    ]
    batch_preferences = axiom.batch_preferences(batch)

    for k, (query, documents) in enumerate(batch):
        assert (batch_preferences[k] == axiom.preferences(query, documents)).all()


def test_batch_preferences_empty() -> None:
    axiom: Axiom[Query, Document] = UniformAxiom(scalar=1)

    preferences = axiom.batch_preferences([])

    assert preferences.shape == (0, 0, 0)


def test_batch_preferences_different_lengths() -> None:
    query1 = Query(id="q1")
    query2 = Query(id="q2")
    document1 = Document(id="d1")
    document2 = Document(id="d2")

    axiom: Axiom[Query, Document] = UniformAxiom(scalar=1)

    with raises(ValueError):
        axiom.batch_preferences(
            [
                (query1, [document1, document2]),
                (query2, [document1]),
            ]
        )
