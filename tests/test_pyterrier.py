from typing import Any, TYPE_CHECKING

from approvaltests import verify
from pytest import skip, fixture

from ir_axioms.axiom import Axiom
from ir_axioms.model import Document
from ir_axioms.axiom.utils import strictly_greater
from ir_axioms.tools import MiddlePivotSelection
from ir_axioms.utils.libraries import is_pyterrier_installed


class _DocumentIdAxiom(Axiom[Any, Document]):
    def preference(self, input: Any, output1: Document, output2: Document) -> float:
        return strictly_greater(output1.id, output2.id)


_DOC_ID = _DocumentIdAxiom()


def test_kwiksort_empty_input() -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    from pandas import DataFrame
    from ir_axioms.integrations import KwikSortReranker

    df_in = DataFrame(columns=["qid", "docno"])

    kwiksort = KwikSortReranker(
        axiom=_DOC_ID,
        pivot_selection=MiddlePivotSelection(),
    )

    df_out = kwiksort.transform(df_in)
    verify(df_out.to_csv(path_or_buf=None, index=False, header=True))


def test_kwiksort_rerank() -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    from pandas import DataFrame
    from ir_axioms.integrations import KwikSortReranker

    df_in = DataFrame(
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
        axiom=_DOC_ID,
        pivot_selection=MiddlePivotSelection(),
    )

    df_out = kwiksort.transform(df_in)
    verify(df_out.to_csv(path_or_buf=None, index=False, header=True))


def test_axiomatic_preferences_empty_input() -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    from pandas import DataFrame
    from ir_axioms.integrations import AxiomaticPreferences

    df_in = DataFrame(columns=["qid", "docno"])

    axiomatic_preferences = AxiomaticPreferences(
        axioms=[_DOC_ID],
        axiom_names=["_DOC_ID"],
        text_field="text",
    )

    df_out = axiomatic_preferences.transform(df_in)
    verify(df_out.to_csv(path_or_buf=None, index=False, header=True))


def test_axiomatic_preferences_compute() -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    from pandas import DataFrame
    from ir_axioms.integrations import AxiomaticPreferences

    df_in = DataFrame(
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
        axioms=[_DOC_ID],
        axiom_names=["_DOC_ID"],
        text_field="text",
    )

    df_out = axiomatic_preferences.transform(df_in)
    verify(df_out.to_csv(path_or_buf=None, index=False, header=True))


def test_aggregated_axiomatic_preferences_empty_input() -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    from pandas import DataFrame
    from ir_axioms.integrations import AggregatedAxiomaticPreferences

    df_in = DataFrame(columns=["qid", "docno"])

    aggregated_axiomatic_preferences = AggregatedAxiomaticPreferences(
        axioms=[_DOC_ID],
        text_field="text",
    )

    df_out = aggregated_axiomatic_preferences.transform(df_in)
    verify(df_out.to_csv(path_or_buf=None, index=False, header=True))


def test_aggregated_axiomatic_preferences_compute() -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    from pandas import DataFrame
    from ir_axioms.integrations import AggregatedAxiomaticPreferences

    df_in = DataFrame(
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
        axioms=[_DOC_ID],
        text_field="text",
    )

    df_out = aggregated_axiomatic_preferences.transform(df_in)
    verify(df_out.to_csv(path_or_buf=None, index=False, header=True))


# TODO: Estimator

if TYPE_CHECKING:
    from ir_axioms.integrations import AxiomaticExperiment
else:
    AxiomaticExperiment = Any


@fixture
def axiomatic_experiment() -> AxiomaticExperiment:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    from pandas import DataFrame
    from pyterrier import Transformer

    from ir_axioms.integrations import AxiomaticExperiment

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
        axioms=[_DOC_ID],
        axiom_names=["_DOC_ID"],
        text_field="text",
    )


def test_axiomatic_experiment_preferences(
    axiomatic_experiment: AxiomaticExperiment,
) -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    verify(
        axiomatic_experiment.preferences.to_csv(
            path_or_buf=None, index=False, header=True
        )
    )


def test_axiomatic_experiment_preference_distribution(
    axiomatic_experiment: AxiomaticExperiment,
) -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    verify(
        axiomatic_experiment.preference_distribution.to_csv(
            path_or_buf=None, index=False, header=True
        )
    )


def test_axiomatic_experiment_preference_consistency(
    axiomatic_experiment: AxiomaticExperiment,
) -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    verify(
        axiomatic_experiment.preference_consistency.to_csv(
            path_or_buf=None, index=False, header=True
        )
    )


def test_axiomatic_experiment_inconsistent_pairs(
    axiomatic_experiment: AxiomaticExperiment,
) -> None:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")

    verify(
        axiomatic_experiment.inconsistent_pairs.to_csv(
            path_or_buf=None, index=False, header=True
        )
    )
