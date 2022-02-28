from ir_axioms.axiom import (
    NopAxiom, OriginalAxiom, AutoAxiom, to_axiom, Axiom, UniformAxiom
)
from ir_axioms.model import Query, RankedTextDocument
from tests.unit.util import MemoryIndexContext


def test_to_axiom_axiom():
    axiom = to_axiom(NopAxiom())

    assert axiom is not None
    assert isinstance(axiom, Axiom)
    assert isinstance(axiom, NopAxiom)


def test_to_axiom_string():
    axiom = to_axiom("NOP")

    assert axiom is not None
    assert isinstance(axiom, Axiom)
    assert isinstance(axiom, NopAxiom)


def test_to_axiom_uniform():
    axiom = to_axiom(42)

    assert axiom is not None
    assert isinstance(axiom, Axiom)
    assert isinstance(axiom, UniformAxiom)


def test_auto_axiom_original():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "w1 w2 w3")
    document2 = RankedTextDocument("d2", 1, 2, "w1 w2 w3")
    context = MemoryIndexContext({document1, document2})

    axiom = OriginalAxiom()
    auto_axiom = AutoAxiom("ORIG")

    assert (
            auto_axiom.preference(context, query, document1, document2) ==
            axiom.preference(context, query, document1, document2)
    )
    assert (
            auto_axiom.preference(context, query, document2, document1) ==
            axiom.preference(context, query, document2, document1)
    )
