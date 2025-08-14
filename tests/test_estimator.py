from typing import Any

from pytest import approx
from sklearn.linear_model import LinearRegression

from ir_axioms.axiom import Axiom, ScikitLearnEstimatorAxiom
from ir_axioms.axiom.utils import strictly_greater
from ir_axioms.model import Query, Document


class _DocumentIdAxiom(Axiom[Any, Document]):
    def preference(self, input: Any, output1: Document, output2: Document) -> float:
        return strictly_greater(output1.id, output2.id)


_DOC_ID = _DocumentIdAxiom


def test_scikit_learn_estimator() -> None:
    query = Query(id="q1")
    document1 = Document(id="d1")
    document2 = Document(id="d2")

    target = _DOC_ID()
    best_axiom = target
    worst_axiom = -target
    axioms = [best_axiom, worst_axiom]

    estimator = LinearRegression()

    estimator_axiom = ScikitLearnEstimatorAxiom(
        axioms=axioms,
        estimator=estimator,
    )

    estimator_axiom.fit(
        target=target,
        inputs_outputs=[
            (query, [document1, document2]),
        ],
    )

    coefficients = estimator.coef_
    assert coefficients.shape == (len(axioms),)
    assert coefficients[0] > coefficients[1]

    assert estimator_axiom.preference(
        input=query,
        output1=document1,
        output2=document2,
    ) == approx(
        target.preference(
            input=query,
            output1=document1,
            output2=document2,
        ),
    )
    assert estimator_axiom.preferences(
        input=query,
        outputs=[document2, document1],
    ) == approx(
        target.preferences(
            input=query,
            outputs=[document2, document1],
        ),
    )
