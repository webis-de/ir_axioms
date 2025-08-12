from typing import Any

from approvaltests import verify
from pytest import skip

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


# TODO: Experiment, Estimator