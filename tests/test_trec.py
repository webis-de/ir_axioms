from numpy import array, float_
from numpy.testing import assert_array_equal
from pandas import DataFrame
from trectools import TrecQrel

from ir_axioms.axiom import TrecOracleAxiom
from ir_axioms.model import Query, Document


def test_trec_oracle() -> None:
    query = Query(id="q1")
    document1 = Document(id="d1")
    document2 = Document(id="d2")

    qrels = TrecQrel()
    qrels.qrels_data = DataFrame(
        {
            "query": ["q1", "q1"],
            "q0": ["Q0", "Q0"],
            "docid": ["d1", "d2"],
            "rel": [1, 0],
        }
    )

    axiom = TrecOracleAxiom(qrels=qrels)

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_trec_oracle_preferences_equal_pairwise_preferences() -> None:
    query = Query(id="q1")
    documents = [Document(id=f"d{index}") for index in range(5)]

    qrels = TrecQrel()
    qrels.qrels_data = DataFrame(
        {
            "query": ["q1"] * len(documents),
            "q0": ["Q0"] * len(documents),
            "docid": [document.id for document in documents],
            "rel": [1, 0, 2, 1, 0],
        }
    )

    axiom = TrecOracleAxiom(qrels=qrels)

    batched = axiom.preferences(query, documents)
    pairwise = array(
        [
            axiom.preference(query, document1, document2)
            for document1 in documents
            for document2 in documents
        ],
        dtype=float_,
    ).reshape((len(documents), len(documents)))

    assert_array_equal(batched, pairwise)
