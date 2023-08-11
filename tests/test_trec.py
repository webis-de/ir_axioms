from pandas import DataFrame
from trectools import TrecQrel, TrecTopics

from ir_axioms.axiom import TrecOracleAxiom
from ir_axioms.model import Query, RankedTextDocument
from tests.unit.util import MemoryIndexContext


def test_trec_oracle():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "w1 w2 w3")
    document2 = RankedTextDocument("d2", 1, 2, "w1 w2 w3")
    context = MemoryIndexContext({document1, document2})

    qrel = TrecQrel()
    qrel.qrels_data = DataFrame({
        "query": [1, 1],
        "q0": ["Q0", "Q0"],
        "docid": ["d1", "d2"],
        "rel": [1, 0],
    })
    topics = TrecTopics({
        1: "q1 q2 q3"
    })

    axiom = TrecOracleAxiom(topics, qrel)

    assert axiom.preference(context, query, document1, document2) == 1
    assert axiom.preference(context, query, document2, document1) == -1
