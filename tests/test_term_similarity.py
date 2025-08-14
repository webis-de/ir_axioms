from ir_axioms.axiom import STMC1, STMC2
from ir_axioms.model import Query, Document


def test_stmc1() -> None:
    query = Query(id="q1", text="blue car moves")
    document1 = Document(id="d1", text="blue auto runs through the city")
    document2 = Document(id="d2", text="red airplane flies in the sky")

    axiom = STMC1()

    # Document d1 contains a more similar term.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_stmc2() -> None:
    query = Query(id="q1", text="car")
    document1 = Document(id="d1", text="car key")
    document2 = Document(id="d2", text="auto auto auto auto")

    axiom = STMC2()

    # Document d1 contains an exact match.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_stmc2_equal() -> None:
    query = Query(id="q1", text="dog breed")
    document1 = Document(id="d1", text="dog fire orange key")
    document2 = Document(id="d2", text="animal animal animal animal time key key key")
    document3 = Document(id="d3", text="dog animal time key key")

    axiom = STMC2()

    # Most similar query term 'dog' and non-query term 'animal'.
    # The document 2 non-query term frequency (0.5)
    # compared to the document 1 query term frequency (0.25)
    # is similar to the document 2 length (8)
    # compared to the document 1 length (4):
    # 2 ≈ 2
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1

    # Most similar query term 'dog' and non-query term 'animal'.
    # The document 3 non-query term frequency (0.2)
    # compared to the document 1 query term frequency (0.25)
    # is not similar to the document 3 length (5)
    # compared to the document 1 length (4):
    # 0.8 != 1.25
    assert axiom.preference(query, document1, document3) == 0
    assert axiom.preference(query, document3, document1) == 0
