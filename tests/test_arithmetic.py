from numpy import array, errstate, isnan, zeros
from pytest import approx

from ir_axioms.axiom import (
    Axiom,
    DistanceAxiom,
    MultiplicativeInverseAxiom,
    NopAxiom,
    UniformAxiom,
    VoteAxiom,
)
from ir_axioms.model import Query, Document


def test_uniform() -> None:
    query = Query(id="q1")
    document1 = Document(id="d1")
    document2 = Document(id="d2")

    axiom: Axiom[Query, Document] = UniformAxiom(scalar=1)

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == 1


def test_negate() -> None:
    axiom1: Axiom[None, int] = DistanceAxiom()
    axiom2 = -axiom1

    assert axiom2.preference(None, 3, 1) == -2
    assert axiom2.preference(None, 1, 3) == 2
    assert axiom2.preference(None, 3, 3) == 0
    assert -axiom2 is axiom1

    assert (
        axiom2.preferences(None, [3, 1])
        == array(
            [
                [0, -2],
                [2, 0],
            ]
        )
    ).all()


def test_sum() -> None:
    axiom1: Axiom[None, int] = DistanceAxiom()

    axiom2 = axiom1 + axiom1 + axiom1

    assert axiom2.preference(None, 3, 1) == 6
    assert axiom2.preference(None, 1, 3) == -6

    assert (
        axiom2.preferences(None, [3, 1])
        == array(
            [
                [0, 6],
                [-6, 0],
            ]
        )
    ).all()


def test_sum_negation() -> None:
    axiom1: Axiom[None, int] = DistanceAxiom()

    axiom2 = axiom1 + -axiom1

    assert axiom2.preference(None, 3, 1) == 0
    assert axiom2.preference(None, 1, 3) == 0

    assert (axiom2.preferences(None, [3, 1]) == zeros((2, 2))).all()


def test_product() -> None:
    axiom1: Axiom[None, int] = DistanceAxiom()

    axiom2 = axiom1 * axiom1 * axiom1

    assert axiom2.preference(None, 3, 1) == 8
    assert axiom2.preference(None, 1, 3) == -8

    assert (
        axiom2.preferences(None, [3, 1])
        == array(
            [
                [0, 8],
                [-8, 0],
            ]
        )
    ).all()


def test_product_negation() -> None:
    axiom1: Axiom[None, int] = DistanceAxiom()

    axiom2 = axiom1 * -axiom1

    assert axiom2.preference(None, 3, 1) == -4
    assert axiom2.preference(None, 1, 3) == -4

    assert (
        axiom2.preferences(None, [3, 1])
        == array(
            [
                [0, -4],
                [-4, 0],
            ]
        )
    ).all()


def test_multiplicative_inverse() -> None:
    axiom1: Axiom[None, int] = DistanceAxiom()

    axiom2 = MultiplicativeInverseAxiom(axiom=axiom1)
    axiom3 = axiom1 / (axiom1 + axiom1)

    assert axiom2.preference(None, 3, 1) == approx(0.5)
    assert axiom2.preference(None, 1, 3) == approx(-0.5)
    assert axiom3.preference(None, 3, 1) == approx(0.5)
    assert axiom3.preference(None, 1, 3) == approx(0.5)

    # The distance is zero on the diagonal, so its inverse is undefined there.
    with errstate(divide="ignore", invalid="ignore"):
        preferences2 = axiom2.preferences(None, [3, 1])
        preferences3 = axiom3.preferences(None, [3, 1])
    assert preferences2[0, 1] == approx(0.5)
    assert preferences2[1, 0] == approx(-0.5)
    assert isnan(preferences3.diagonal()).all()
    assert preferences3[0, 1] == approx(0.5)
    assert preferences3[1, 0] == approx(0.5)


def test_and() -> None:
    axiom1: Axiom[None, int] = DistanceAxiom()
    axiom2: Axiom[None, int] = NopAxiom()

    axiom3 = axiom1 & axiom1
    axiom4 = axiom1 & -axiom1
    axiom5 = axiom1 & axiom1 & axiom2

    assert axiom3.preference(None, 3, 1) == 1
    assert axiom3.preference(None, 1, 3) == -1
    assert axiom4.preference(None, 3, 1) == 0
    assert axiom4.preference(None, 1, 3) == 0
    assert axiom5.preference(None, 3, 1) == 0
    assert axiom5.preference(None, 1, 3) == 0

    assert (
        axiom3.preferences(None, [3, 1])
        == array(
            [
                [0, 1],
                [-1, 0],
            ]
        )
    ).all()
    assert (axiom4.preferences(None, [3, 1]) == zeros((2, 2))).all()
    assert (axiom5.preferences(None, [3, 1]) == zeros((2, 2))).all()


def test_majority_vote() -> None:
    axiom1: Axiom[None, int] = DistanceAxiom()
    axiom2 = -axiom1

    axiom3 = axiom1 % axiom1 % axiom2
    axiom4 = VoteAxiom(axioms=[axiom1, axiom1, axiom2], minimum_votes=0.5)
    axiom5 = VoteAxiom(axioms=[axiom1, axiom1, axiom2], minimum_votes=0.75)

    assert axiom3.preference(None, 3, 1) == 1
    assert axiom3.preference(None, 1, 3) == -1
    assert axiom4.preference(None, 3, 1) == 1
    assert axiom4.preference(None, 1, 3) == -1
    assert axiom5.preference(None, 3, 1) == 0
    assert axiom5.preference(None, 1, 3) == 0

    assert (
        axiom3.preferences(None, [3, 1])
        == array(
            [
                [0, 1],
                [-1, 0],
            ]
        )
    ).all()
    assert (
        axiom4.preferences(None, [3, 1])
        == array(
            [
                [0, 1],
                [-1, 0],
            ]
        )
    ).all()
    assert (axiom5.preferences(None, [3, 1]) == zeros((2, 2))).all()


def test_cascade() -> None:
    axiom1: Axiom[None, int] = DistanceAxiom()
    axiom2: Axiom[None, int] = NopAxiom()

    axiom3 = axiom2 | axiom1
    axiom4 = axiom2 | -axiom1 | axiom1

    assert axiom3.preference(None, 3, 1) == 2
    assert axiom3.preference(None, 1, 3) == -2
    assert axiom4.preference(None, 3, 1) == -2
    assert axiom4.preference(None, 1, 3) == 2

    assert (
        axiom3.preferences(None, [3, 1])
        == array(
            [
                [0, 2],
                [-2, 0],
            ]
        )
    ).all()
    assert (
        axiom4.preferences(None, [3, 1])
        == array(
            [
                [0, -2],
                [2, 0],
            ]
        )
    ).all()


def test_normalize() -> None:
    axiom1: Axiom[None, int] = DistanceAxiom()
    axiom2 = +axiom1

    assert axiom1.preference(None, 3, 1) == 2
    assert axiom1.preference(None, 1, 3) == -2
    assert axiom2.preference(None, 3, 1) == 1
    assert axiom2.preference(None, 1, 3) == -1
    assert axiom2.preference(None, 3, 3) == 0

    assert (
        axiom1.preferences(None, [3, 1])
        == array(
            [
                [0, 2],
                [-2, 0],
            ]
        )
    ).all()
    assert (
        axiom2.preferences(None, [3, 1])
        == array(
            [
                [0, 1],
                [-1, 0],
            ]
        )
    ).all()
