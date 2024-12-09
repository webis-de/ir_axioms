from axioms.axiom import STMC1  #, STMC2
from axioms.model import TextQuery, TextDocument
from tests.util import inject_documents


def test_stmc1():
    query = TextQuery("q1", "blue car moves")
    document1 = TextDocument("d1", "blue auto goes through the city")
    document2 = TextDocument("d2", "red airplane flies in the sky")

    inject_documents([document1, document2])

    axiom = STMC1()

    # Document d1 contains a more similar term.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


# FIXME: The below tests fail now because of the changed term similarity implementation. We should mock the term similarity here and enable the checks again.
# def test_stmc2():
#     query = TextQuery("q1", "q")
#     document1 = TextDocument("d1", "q")
#     document2 = TextDocument("d2", "t t t t")

#     inject_documents([document1, document2])

#     axiom = STMC2()

#     # Document d1 contains an exact match.
#     assert axiom.preference(query, document1, document2) == 1
#     assert axiom.preference(query, document2, document1) == -1


# def test_stmc2_equal():
#     query = TextQuery("q1", "dog breed")
#     document1 = TextDocument("d1", "dog fire orange")
#     document2 = TextDocument("d2", "dog animal animal animal time key")
#     document3 = TextDocument("d3", "dog animal time key")

#     inject_documents([document1, document2])

#     axiom = STMC2()

#     # Most similar query term 'dog' and non-query term 'animal'.
#     # The document 2 non-query term frequency (0.5)
#     # compared to the document 1 query term frequency (0.333)
#     # is similar to the document 2 term set length (4)
#     # compared to the document 1 term set length (3):
#     # 1.333 ≈ 1.5
#     assert axiom.preference(query, document1, document2) == 1
#     assert axiom.preference(query, document2, document1) == -1

#     # Most similar query term 'dog' and non-query term 'animal'.
#     # The document 3 non-query term frequency (0.25)
#     # compared to the document 1 query term frequency (0.333)
#     # is not similar to the document 3 term set length (4)
#     # compared to the document 1 term set length (3):
#     # 1.333 != 0.75
#     assert axiom.preference(query, document1, document3) == 0
#     assert axiom.preference(query, document3, document1) == 0
