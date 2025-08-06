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
