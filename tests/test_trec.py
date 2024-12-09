from pandas import DataFrame
from trectools import TrecQrel

from axioms.axiom import TrecOracleAxiom
from axioms.model import Query, Document


def test_trec_oracle():
    query = Query("q1")
    document1 = Document("d1")
    document2 = Document("d2")

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
