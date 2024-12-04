from axioms.axiom import UniformAxiom, VoteAxiom
from axioms.model import Query, RankedTextDocument
from pytest import approx


def test_uniform():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "w1 w2 w3")
    document2 = RankedTextDocument("d2", 1, 2, "w1 w2 w3")

    axiom = UniformAxiom(scalar=1)

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == 1


def test_sum():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "w1 w2 w3")
    document2 = RankedTextDocument("d2", 1, 2, "w1 w2 w3")

    axiom1 = UniformAxiom(scalar=1)
    axiom2 = UniformAxiom(scalar=2)
    axiom3 = UniformAxiom(scalar=3)

    axiom = axiom1 + axiom2 + axiom3

    assert axiom.preference(query, document1, document2) == 6
    assert axiom.preference(query, document2, document1) == 6


def test_product():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "w1 w2 w3")
    document2 = RankedTextDocument("d2", 1, 2, "w1 w2 w3")

    axiom1 = UniformAxiom(scalar=1)
    axiom2 = UniformAxiom(scalar=2)
    axiom3 = UniformAxiom(scalar=3)

    axiom = axiom1 * axiom2 * axiom3

    assert axiom.preference(query, document1, document2) == 6
    assert axiom.preference(query, document2, document1) == 6


def test_multiplicative_inverse():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "w1 w2 w3")
    document2 = RankedTextDocument("d2", 1, 2, "w1 w2 w3")

    axiom = UniformAxiom(scalar=1) / UniformAxiom(scalar=2)

    assert axiom.preference(query, document1, document2) == approx(0.5)
    assert axiom.preference(query, document2, document1) == approx(0.5)


def test_and():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "w1 w2 w3")
    document2 = RankedTextDocument("d2", 1, 2, "w1 w2 w3")

    axiom1 = UniformAxiom(scalar=1)
    axiom2 = UniformAxiom(scalar=2)
    axiom3 = UniformAxiom(scalar=0)

    axiom4 = axiom1 & axiom2
    axiom5 = axiom1 & axiom2 & axiom3

    assert axiom4.preference(query, document1, document2) == 1
    assert axiom4.preference(query, document2, document1) == 1
    assert axiom5.preference(query, document1, document2) == 0
    assert axiom5.preference(query, document2, document1) == 0


def test_majority_vote():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "w1 w2 w3")
    document2 = RankedTextDocument("d2", 1, 2, "w1 w2 w3")

    axiom1 = UniformAxiom(scalar=1)
    axiom2 = UniformAxiom(scalar=2)
    axiom3 = UniformAxiom(scalar=0)

    axiom4 = axiom1 % axiom2 % axiom3
    axiom5 = VoteAxiom(axioms=[axiom1, axiom2, axiom3], minimum_votes=0.5)
    axiom6 = VoteAxiom(axioms=[axiom1, axiom2, axiom3], minimum_votes=0.75)

    assert axiom4.preference(query, document1, document2) == 1
    assert axiom4.preference(query, document2, document1) == 1
    assert axiom5.preference(query, document1, document2) == 1
    assert axiom5.preference(query, document2, document1) == 1
    assert axiom6.preference(query, document1, document2) == 0
    assert axiom6.preference(query, document2, document1) == 0


def test_cascade():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "w1 w2 w3")
    document2 = RankedTextDocument("d2", 1, 2, "w1 w2 w3")

    axiom1 = UniformAxiom(scalar=0)
    axiom2 = UniformAxiom(scalar=1)
    axiom3 = UniformAxiom(scalar=2)

    axiom4 = axiom1 | axiom2
    axiom5 = axiom1 | axiom3

    assert axiom1.preference(query, document1, document2) == 0
    assert axiom1.preference(query, document2, document1) == 0
    assert axiom4.preference(query, document1, document2) == 1
    assert axiom4.preference(query, document2, document1) == 1
    assert axiom5.preference(query, document1, document2) == 2
    assert axiom5.preference(query, document2, document1) == 2


def test_normalize():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "w1 w2 w3")
    document2 = RankedTextDocument("d2", 1, 2, "w1 w2 w3")

    axiom1 = UniformAxiom(scalar=2)
    axiom2 = +axiom1

    assert axiom1.preference(query, document1, document2) == 2
    assert axiom1.preference(query, document2, document1) == 2
    assert axiom2.preference(query, document1, document2) == 1
    assert axiom2.preference(query, document2, document1) == 1
