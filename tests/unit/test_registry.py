from class_registry import SortedClassRegistry

from ir_axioms import registry
from ir_axioms.axiom import NopAxiom, OriginalAxiom, Axiom
from ir_axioms.model import Query, RankedTextDocument
from tests.unit.util import MemoryIndexContext


def test_registry():
    assert registry is not None
    assert isinstance(registry, SortedClassRegistry)
    assert len(registry) > 0


def test_registry_axiom():
    axiom = registry["NOP"]

    assert axiom is not None
    assert isinstance(axiom, Axiom)
    assert isinstance(axiom, NopAxiom)


def test_registry_original():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "w1 w2 w3")
    document2 = RankedTextDocument("d2", 1, 2, "w1 w2 w3")
    context = MemoryIndexContext({document1, document2})

    axiom = OriginalAxiom()
    registry_axiom = registry["ORIG"]

    assert (
            registry_axiom.preference(context, query, document1, document2) ==
            axiom.preference(context, query, document1, document2)
    )
    assert (
            registry_axiom.preference(context, query, document2, document1) ==
            axiom.preference(context, query, document2, document1)
    )
