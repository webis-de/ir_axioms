from typing import Any

from pandas import DataFrame
from pandas.testing import assert_frame_equal
from pytest import skip, fixture
from sklearn.linear_model import LinearRegression

from ir_axioms.axiom import Axiom
from ir_axioms.axiom.utils import strictly_greater
from ir_axioms.integrations import (
    KwikSortReranker,
    AxiomaticPreferences,
    AggregatedAxiomaticPreferences,
    EstimatorKwikSortReranker,
    AxiomaticExperiment,
)
from ir_axioms.model import Document
from ir_axioms.tools import MiddlePivotSelection
from ir_axioms.utils.libraries import is_pyterrier_installed


class _DocumentIdAxiom(Axiom[Any, Document]):
    def preference(self, input: Any, output1: Document, output2: Document) -> float:
        return strictly_greater(output1.id, output2.id)


_DOC_ID = _DocumentIdAxiom


def test_kwiksort_reranker_empty() -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    res = DataFrame(columns=["qid", "docno"])

    kwiksort = KwikSortReranker(
        axiom=_DOC_ID(),
        pivot_selection=MiddlePivotSelection(),
    )

    actual = kwiksort.transform(res)
    expected = DataFrame(columns=["qid", "docno"])

    assert_frame_equal(actual, expected)


def test_kwiksort_reranker() -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    res = DataFrame(
        [
            {"qid": "q1", "docno": "doc1"},
            {"qid": "q1", "docno": "doc3"},
            {"qid": "q1", "docno": "doc2"},
            {"qid": "q2", "docno": "doc5"},
            {"qid": "q2", "docno": "doc6"},
            {"qid": "q2", "docno": "doc4"},
        ]
    )

    kwiksort = KwikSortReranker(
        axiom=_DOC_ID(),
        pivot_selection=MiddlePivotSelection(),
    )

    actual = kwiksort.transform(res)
    expected = DataFrame(
        [
            {"qid": "q1", "docno": "doc3", "score": 0, "rank": 0},
            {"qid": "q1", "docno": "doc2", "score": -1, "rank": 1},
            {"qid": "q1", "docno": "doc1", "score": -2, "rank": 2},
            {"qid": "q2", "docno": "doc6", "score": 0, "rank": 0},
            {"qid": "q2", "docno": "doc5", "score": -1, "rank": 1},
            {"qid": "q2", "docno": "doc4", "score": -2, "rank": 2},
        ]
    )

    assert_frame_equal(
        actual.sort_values(by=["qid", "rank"]).reset_index(drop=True),
        expected.sort_values(by=["qid", "rank"]).reset_index(drop=True),
    )


def test_axiomatic_preferences_empty() -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    res = DataFrame(columns=["qid", "docno"])

    axiomatic_preferences = AxiomaticPreferences(
        axioms=[_DOC_ID()],
        axiom_names=["_DOC_ID"],
    )

    actual = axiomatic_preferences.transform(res)
    expected = DataFrame(columns=["qid", "docno_a", "docno_b", "_DOC_ID_preference"])

    assert_frame_equal(actual, expected)


def test_axiomatic_preferences() -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    res = DataFrame(
        [
            {"qid": "q1", "docno": "doc1"},
            {"qid": "q1", "docno": "doc3"},
            {"qid": "q1", "docno": "doc2"},
            {"qid": "q2", "docno": "doc5"},
            {"qid": "q2", "docno": "doc6"},
            {"qid": "q2", "docno": "doc4"},
        ]
    )

    axiomatic_preferences = AxiomaticPreferences(
        axioms=[_DOC_ID()],
        axiom_names=["_DOC_ID"],
    )

    actual = axiomatic_preferences.transform(res)
    expected = DataFrame(
        [
            {
                "qid": "q1",
                "docno_a": "doc1",
                "docno_b": "doc1",
                "_DOC_ID_preference": 0.0,
            },
            {
                "qid": "q1",
                "docno_a": "doc1",
                "docno_b": "doc3",
                "_DOC_ID_preference": -1.0,
            },
            {
                "qid": "q1",
                "docno_a": "doc1",
                "docno_b": "doc2",
                "_DOC_ID_preference": -1.0,
            },
            {
                "qid": "q1",
                "docno_a": "doc3",
                "docno_b": "doc1",
                "_DOC_ID_preference": 1.0,
            },
            {
                "qid": "q1",
                "docno_a": "doc3",
                "docno_b": "doc3",
                "_DOC_ID_preference": 0.0,
            },
            {
                "qid": "q1",
                "docno_a": "doc3",
                "docno_b": "doc2",
                "_DOC_ID_preference": 1.0,
            },
            {
                "qid": "q1",
                "docno_a": "doc2",
                "docno_b": "doc1",
                "_DOC_ID_preference": 1.0,
            },
            {
                "qid": "q1",
                "docno_a": "doc2",
                "docno_b": "doc3",
                "_DOC_ID_preference": -1.0,
            },
            {
                "qid": "q1",
                "docno_a": "doc2",
                "docno_b": "doc2",
                "_DOC_ID_preference": 0.0,
            },
            {
                "qid": "q2",
                "docno_a": "doc5",
                "docno_b": "doc5",
                "_DOC_ID_preference": 0.0,
            },
            {
                "qid": "q2",
                "docno_a": "doc5",
                "docno_b": "doc6",
                "_DOC_ID_preference": -1.0,
            },
            {
                "qid": "q2",
                "docno_a": "doc5",
                "docno_b": "doc4",
                "_DOC_ID_preference": 1.0,
            },
            {
                "qid": "q2",
                "docno_a": "doc6",
                "docno_b": "doc5",
                "_DOC_ID_preference": 1.0,
            },
            {
                "qid": "q2",
                "docno_a": "doc6",
                "docno_b": "doc6",
                "_DOC_ID_preference": 0.0,
            },
            {
                "qid": "q2",
                "docno_a": "doc6",
                "docno_b": "doc4",
                "_DOC_ID_preference": 1.0,
            },
            {
                "qid": "q2",
                "docno_a": "doc4",
                "docno_b": "doc5",
                "_DOC_ID_preference": -1.0,
            },
            {
                "qid": "q2",
                "docno_a": "doc4",
                "docno_b": "doc6",
                "_DOC_ID_preference": -1.0,
            },
            {
                "qid": "q2",
                "docno_a": "doc4",
                "docno_b": "doc4",
                "_DOC_ID_preference": 0.0,
            },
        ]
    )

    assert_frame_equal(
        actual.sort_values(by=["qid", "docno_a", "docno_b"]).reset_index(drop=True),
        expected.sort_values(by=["qid", "docno_a", "docno_b"]).reset_index(drop=True),
    )


def test_aggregated_axiomatic_preferences_empty() -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    res = DataFrame(columns=["qid", "docno"])

    aggregated_axiomatic_preferences = AggregatedAxiomaticPreferences(
        axioms=[_DOC_ID()], aggregations=[max, min]
    )

    actual = aggregated_axiomatic_preferences.transform(res)
    expected = DataFrame(columns=["qid", "docno", "features"])

    assert_frame_equal(actual, expected)


def test_aggregated_axiomatic_preferences() -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    res = DataFrame(
        [
            {"qid": "q1", "docno": "doc1"},
            {"qid": "q1", "docno": "doc3"},
            {"qid": "q1", "docno": "doc2"},
            {"qid": "q2", "docno": "doc5"},
            {"qid": "q2", "docno": "doc6"},
            {"qid": "q2", "docno": "doc4"},
        ]
    )

    aggregated_axiomatic_preferences = AggregatedAxiomaticPreferences(
        axioms=[_DOC_ID()],
        aggregations=[max, min],
    )

    actual = aggregated_axiomatic_preferences.transform(res)
    expected = DataFrame(
        [
            {"qid": "q1", "docno": "doc1", "features": [1, 0]},
            {"qid": "q1", "docno": "doc3", "features": [0, -1]},
            {"qid": "q1", "docno": "doc2", "features": [1, -1]},
            {"qid": "q2", "docno": "doc5", "features": [1, -1]},
            {"qid": "q2", "docno": "doc6", "features": [0, -1]},
            {"qid": "q2", "docno": "doc4", "features": [1, 0]},
        ]
    )

    assert_frame_equal(actual, expected)


def test_estimator_kwiksort_reranker() -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    res = DataFrame(
        [
            {"qid": "q1", "docno": "doc1"},
            {"qid": "q1", "docno": "doc3"},
            {"qid": "q1", "docno": "doc2"},
            {"qid": "q2", "docno": "doc5"},
            {"qid": "q2", "docno": "doc6"},
            {"qid": "q2", "docno": "doc4"},
        ]
    )
    qrels = DataFrame(columns=["qid", "docno", "label"])

    target = _DOC_ID()

    target_kwiksort = KwikSortReranker(
        axiom=target,
        pivot_selection=MiddlePivotSelection(),
    )

    best_axiom = target
    worst_axiom = -target
    axioms = [best_axiom, worst_axiom]

    estimator = LinearRegression()

    estimator_kwiksort = EstimatorKwikSortReranker(
        axioms=axioms,
        target=target,
        estimator=estimator,
        pivot_selection=MiddlePivotSelection(),
    )

    estimator_kwiksort.fit(res, qrels)

    coefficients = estimator.coef_
    assert coefficients.shape == (len(axioms),)
    assert coefficients[0] > coefficients[1]

    actual = estimator_kwiksort.transform(res)
    expected = target_kwiksort.transform(res)

    assert_frame_equal(
        actual.sort_values(by=["qid", "rank"]).reset_index(drop=True),
        expected.sort_values(by=["qid", "rank"]).reset_index(drop=True),
    )


@fixture
def axiomatic_experiment() -> AxiomaticExperiment:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")
    from pyterrier import Transformer

    return AxiomaticExperiment(
        retrieval_systems=[
            Transformer.from_df(
                DataFrame(
                    [
                        {"qid": "q1", "docno": "doc2", "score": 0.5, "rank": 0},
                        {"qid": "q1", "docno": "doc1", "score": 0.3, "rank": 1},
                        {"qid": "q2", "docno": "doc1", "score": 0.4, "rank": 0},
                        {"qid": "q2", "docno": "doc2", "score": 0.2, "rank": 1},
                    ]
                )
            )
        ],
        names=["test_system"],
        topics=DataFrame(
            [
                {"qid": "q1"},
                {"qid": "q2"},
            ]
        ),
        qrels=DataFrame(
            [
                {"qid": "q1", "docno": "doc1", "label": 1},
                {"qid": "q1", "docno": "doc2", "label": 0},
                {"qid": "q2", "docno": "doc1", "label": 1},
                {"qid": "q2", "docno": "doc2", "label": 0},
            ]
        ),
        axioms=[_DOC_ID()],
        axiom_names=["_DOC_ID"],
    )


def test_axiomatic_experiment_preferences(
    axiomatic_experiment: AxiomaticExperiment,
) -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    actual = axiomatic_experiment.preferences
    expected = DataFrame(
        [
            {
                "qid": "q1",
                "docno_a": "doc2",
                "score_a": 0.5,
                "rank_a": 0,
                "label_a": 0,
                "docno_b": "doc2",
                "score_b": 0.5,
                "rank_b": 0,
                "label_b": 0,
                "ORIG_preference": 0.0,
                "ORACLE_preference": 0.0,
                "_DOC_ID_preference": 0.0,
                "name": "test_system",
            },
            {
                "qid": "q1",
                "docno_a": "doc2",
                "score_a": 0.5,
                "rank_a": 0,
                "label_a": 0,
                "docno_b": "doc1",
                "score_b": 0.3,
                "rank_b": 1,
                "label_b": 1,
                "ORIG_preference": 1.0,
                "ORACLE_preference": -1.0,
                "_DOC_ID_preference": 1.0,
                "name": "test_system",
            },
            {
                "qid": "q1",
                "docno_a": "doc1",
                "score_a": 0.3,
                "rank_a": 1,
                "label_a": 1,
                "docno_b": "doc2",
                "score_b": 0.5,
                "rank_b": 0,
                "label_b": 0,
                "ORIG_preference": -1.0,
                "ORACLE_preference": 1.0,
                "_DOC_ID_preference": -1.0,
                "name": "test_system",
            },
            {
                "qid": "q1",
                "docno_a": "doc1",
                "score_a": 0.3,
                "rank_a": 1,
                "label_a": 1,
                "docno_b": "doc1",
                "score_b": 0.3,
                "rank_b": 1,
                "label_b": 1,
                "ORIG_preference": 0.0,
                "ORACLE_preference": 0.0,
                "_DOC_ID_preference": 0.0,
                "name": "test_system",
            },
            {
                "qid": "q2",
                "docno_a": "doc1",
                "score_a": 0.4,
                "rank_a": 0,
                "label_a": 1,
                "docno_b": "doc1",
                "score_b": 0.4,
                "rank_b": 0,
                "label_b": 1,
                "ORIG_preference": 0.0,
                "ORACLE_preference": 0.0,
                "_DOC_ID_preference": 0.0,
                "name": "test_system",
            },
            {
                "qid": "q2",
                "docno_a": "doc1",
                "score_a": 0.4,
                "rank_a": 0,
                "label_a": 1,
                "docno_b": "doc2",
                "score_b": 0.2,
                "rank_b": 1,
                "label_b": 0,
                "ORIG_preference": 1.0,
                "ORACLE_preference": 1.0,
                "_DOC_ID_preference": -1.0,
                "name": "test_system",
            },
            {
                "qid": "q2",
                "docno_a": "doc2",
                "score_a": 0.2,
                "rank_a": 1,
                "label_a": 0,
                "docno_b": "doc1",
                "score_b": 0.4,
                "rank_b": 0,
                "label_b": 1,
                "ORIG_preference": -1.0,
                "ORACLE_preference": -1.0,
                "_DOC_ID_preference": 1.0,
                "name": "test_system",
            },
            {
                "qid": "q2",
                "docno_a": "doc2",
                "score_a": 0.2,
                "rank_a": 1,
                "label_a": 0,
                "docno_b": "doc2",
                "score_b": 0.2,
                "rank_b": 1,
                "label_b": 0,
                "ORIG_preference": 0.0,
                "ORACLE_preference": 0.0,
                "_DOC_ID_preference": 0.0,
                "name": "test_system",
            },
        ]
    )

    assert_frame_equal(
        actual.sort_values(by=["qid", "docno_a", "docno_b"]).reset_index(drop=True),
        expected.sort_values(by=["qid", "docno_a", "docno_b"]).reset_index(drop=True),
    )


def test_axiomatic_experiment_preference_distribution(
    axiomatic_experiment: AxiomaticExperiment,
) -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    actual = axiomatic_experiment.preference_distribution
    expected = DataFrame(
        [
            {
                "axiom": "ORIG",
                "axiom == 0": 0,
                "axiom == ORIG": 2,
                "axiom != ORIG": 0,
            },
            {
                "axiom": "ORACLE",
                "axiom == 0": 0,
                "axiom == ORIG": 1,
                "axiom != ORIG": 1,
            },
            {
                "axiom": "_DOC_ID",
                "axiom == 0": 0,
                "axiom == ORIG": 1,
                "axiom != ORIG": 1,
            },
        ]
    )

    assert_frame_equal(
        actual.sort_values(by="axiom").reset_index(drop=True),
        expected.sort_values(by="axiom").reset_index(drop=True),
    )


def test_axiomatic_experiment_preference_consistency(
    axiomatic_experiment: AxiomaticExperiment,
) -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    actual = axiomatic_experiment.preference_consistency
    expected = DataFrame(
        [
            {
                "axiom": "ORIG",
                "ORIG_consistency": 1.0,
                "ORACLE_consistency": 0.5,
                "test_system_consistency": 1.0,
            },
            {
                "axiom": "ORACLE",
                "ORIG_consistency": 0.5,
                "ORACLE_consistency": 1.0,
                "test_system_consistency": 0.5,
            },
            {
                "axiom": "_DOC_ID",
                "ORIG_consistency": 0.5,
                "ORACLE_consistency": 0.0,
                "test_system_consistency": 0.5,
            },
        ]
    )

    assert_frame_equal(
        actual.sort_values(by="axiom").reset_index(drop=True),
        expected.sort_values(by="axiom").reset_index(drop=True),
    )


def test_axiomatic_experiment_inconsistent_pairs(
    axiomatic_experiment: AxiomaticExperiment,
) -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    actual = axiomatic_experiment.inconsistent_pairs
    expected = DataFrame(
        [
            {
                "qid": "q1",
                "docno_a": "doc1",
                "score_a": 0.3,
                "rank_a": 1,
                "label_a": 1,
                "docno_b": "doc2",
                "score_b": 0.5,
                "rank_b": 0,
                "label_b": 0,
                "ORIG_preference": -1.0,
                "ORACLE_preference": 1.0,
                "_DOC_ID_preference": -1.0,
                "name": "test_system",
            },
        ]
    )

    assert_frame_equal(
        actual.sort_values(by=["qid", "docno_a", "docno_b"]).reset_index(drop=True),
        expected.sort_values(by=["qid", "docno_a", "docno_b"]).reset_index(drop=True),
    )
