from math import log
from pathlib import Path
from typing import TypeVar, Any

from injector import Injector
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from pyterrier import IterDictIndexer
from pytest import fixture, approx, skip

from ir_axioms.axiom import Tfc1Axiom
from ir_axioms.precondition import NOP
from ir_axioms.dependency_injection import injector as _default_injector
from ir_axioms.integrations import inject_pyterrier, KwikSortReranker
from ir_axioms.model import Query, Document, TokenizedString
from ir_axioms.tools import (
    IndexStatistics,
    TextStatistics,
    TextContents,
    TermTokenizer,
    MiddlePivotSelection,
)
from ir_axioms.utils.lazy import lazy_inject
from ir_axioms.utils.libraries import is_pyterrier_installed


@fixture
def query() -> Query:
    return Query(id="q1", text=TokenizedString("a ##3", tokens={"a": 1, "##3": 1}))


@fixture
def document1() -> Document:
    return Document(
        id="d1",
        text=TokenizedString("a ##2 ##2", tokens={"a": 1, "##2": 2}),
    )


@fixture
def document2() -> Document:
    return Document(
        id="d2",
        text=TokenizedString("a ##3 a", tokens={"a": 2, "##3": 1}),
    )


@fixture
def index_path(
    tmp_path: Path,
    document1: Document,
    document2: Document,
) -> Any:
    assert isinstance(document1.text, TokenizedString)
    assert isinstance(document2.text, TokenizedString)

    indexer = IterDictIndexer(
        str(tmp_path.resolve()),
        meta={"docno": 20, "text": 5000},
        pretokenised=True,
    )
    indexer.index(
        [
            {
                "docno": document1.id,
                "toks": document1.text.tokens,
                "text": document1.text,
            },
            {
                "docno": document2.id,
                "toks": document2.text.tokens,
                "text": document2.text,
            },
        ]
    )  # pyright: ignore[reportOptionalCall]
    return tmp_path


_T = TypeVar("_T")


@fixture
def injector(index_path: Path) -> Injector:
    if not is_pyterrier_installed():
        skip("PyTerrier is not installed.")
    injector = Injector(parent=_default_injector)
    inject_pyterrier(
        index_location=index_path,
        text_field="text",
        injector=injector,
    )
    return injector


@fixture
def index_statistics(injector: Injector) -> IndexStatistics:
    return injector.get(IndexStatistics)


@fixture
def query_text_contents(injector: Injector) -> TextContents[Query]:
    return injector.get(TextContents[Query])


@fixture
def document_text_contents(injector: Injector) -> TextContents[Document]:
    return injector.get(TextContents[Document])


@fixture
def query_text_statistics(injector: Injector) -> TextStatistics[Query]:
    return injector.get(TextStatistics[Query])


@fixture
def document_text_statistics(injector: Injector) -> TextStatistics[Document]:
    return injector.get(TextStatistics[Document])


@fixture
def term_tokenizer(injector: Injector) -> TermTokenizer:
    return injector.get(TermTokenizer)


def test_document_count(index_statistics: IndexStatistics) -> None:
    assert index_statistics.document_count == 2


def test_document_frequency(index_statistics: IndexStatistics) -> None:
    assert index_statistics.document_frequency("computer") == 0
    assert index_statistics.document_frequency("a") == 2
    assert index_statistics.document_frequency("##2") == 1
    assert index_statistics.document_frequency("##3") == 1


def test_inverse_document_frequency(index_statistics: IndexStatistics) -> None:
    assert index_statistics.inverse_document_frequency("computer") == 0
    assert index_statistics.inverse_document_frequency("a") == approx(log(2 / 2))
    assert index_statistics.inverse_document_frequency("##2") == approx(log(2 / 1))
    assert index_statistics.inverse_document_frequency("##3") == approx(log(2 / 1))


def test_contents_query(
    query_text_contents: TextContents[Query],
    query: Query,
) -> None:
    assert query_text_contents.contents(query) == "a ##3"


def test_terms_query(
    query_text_contents: TextContents[Query],
    term_tokenizer: TermTokenizer,
    query: Query,
) -> None:
    skip("Cannot restore term positions from term frequency mapping.")
    assert term_tokenizer.terms(query_text_contents.contents(query)) == ["a", "##3"]


def test_unique_terms_query(
    query_text_contents: TextContents[Query],
    term_tokenizer: TermTokenizer,
    query: Query,
) -> None:
    assert term_tokenizer.unique_terms(query_text_contents.contents(query)) == {
        "a",
        "##3",
    }


def test_contents_document1(
    document_text_contents: TextContents[Document],
    document1: Document,
) -> None:
    assert document_text_contents.contents(document1) == "a ##2 ##2"


def test_contents_document2(
    document_text_contents: TextContents[Document],
    document2: Document,
) -> None:
    assert document_text_contents.contents(document2) == "a ##3 a"


def test_terms_document1(
    document_text_contents: TextContents[Document],
    term_tokenizer: TermTokenizer,
    document1: Document,
) -> None:
    skip("Cannot restore term positions from term frequency mapping.")
    assert term_tokenizer.terms(document_text_contents.contents(document1)) == [
        "a",
        "##2",
        "##2",
    ]


def test_terms_document2(
    document_text_contents: TextContents[Document],
    term_tokenizer: TermTokenizer,
    document2: Document,
) -> None:
    skip("Cannot restore term positions from term frequency mapping.")
    assert term_tokenizer.terms(document_text_contents.contents(document2)) == [
        "a",
        "##3",
        "##3",
    ]


def test_unique_terms_document1(
    document_text_contents: TextContents[Document],
    term_tokenizer: TermTokenizer,
    document1: Document,
) -> None:
    assert term_tokenizer.unique_terms(document_text_contents.contents(document1)) == {
        "a",
        "##2",
    }


def test_unique_terms_document2(
    document_text_contents: TextContents[Document],
    term_tokenizer: TermTokenizer,
    document2: Document,
) -> None:
    assert term_tokenizer.unique_terms(document_text_contents.contents(document2)) == {
        "a",
        "##3",
    }


def test_term_count_query(
    query_text_statistics: TextStatistics[Query],
    query: Query,
) -> None:
    assert query_text_statistics.term_count(query, "computer") == 0
    assert query_text_statistics.term_count(query, "a") == 1
    assert query_text_statistics.term_count(query, "##3") == 1


def test_term_frequency_query(
    query_text_statistics: TextStatistics[Query],
    query: Query,
) -> None:
    assert query_text_statistics.term_frequency(query, "computer") == 0
    assert query_text_statistics.term_frequency(query, "a") == approx(1 / 2)
    assert query_text_statistics.term_frequency(query, "##3") == approx(1 / 2)


def test_term_count_document1(
    document_text_statistics: TextStatistics[Document],
    document1: Document,
) -> None:
    assert document_text_statistics.term_count(document1, "computer") == 0
    assert document_text_statistics.term_count(document1, "a") == 1
    assert document_text_statistics.term_count(document1, "##2") == 2


def test_term_frequency_document1(
    document_text_statistics: TextStatistics[Document],
    document1: Document,
) -> None:
    assert document_text_statistics.term_frequency(document1, "computer") == 0
    assert document_text_statistics.term_frequency(document1, "a") == approx(1 / 3)
    assert document_text_statistics.term_frequency(document1, "##2") == approx(2 / 3)


def test_term_count_document2(
    document_text_statistics: TextStatistics[Document],
    document2: Document,
) -> None:
    assert document_text_statistics.term_count(document2, "computer") == 0
    assert document_text_statistics.term_count(document2, "a") == 2
    assert document_text_statistics.term_count(document2, "##3") == 1


def test_term_frequency_document2(
    document_text_statistics: TextStatistics[Document],
    document2: Document,
) -> None:
    assert document_text_statistics.term_frequency(document2, "computer") == 0
    assert document_text_statistics.term_frequency(document2, "a") == approx(2 / 3)
    assert document_text_statistics.term_frequency(document2, "##3") == approx(1 / 3)


def test_kwiksort_reranker(injector: Injector) -> None:
    TFC1 = lazy_inject(Tfc1Axiom, injector=injector)
    axiom = TFC1(
        precondition=NOP,
        margin_fraction=0,
    )

    res = DataFrame(
        [
            {
                "qid": "q1",
                "query": "a ##3",
                "query_toks": {"a": 1, "##3": 1},
                "docno": "d1",
                # "toks": {"a": 1, "##2": 2},
            },
            {
                "qid": "q1",
                "query": "a ##3",
                "query_toks": {"a": 1, "##3": 1},
                "docno": "d2",
                # "toks": {"a": 2, "##3": 1},
            },
        ]
    )

    kwiksort = KwikSortReranker(
        axiom=axiom,
        pivot_selection=MiddlePivotSelection(),
    )

    actual = kwiksort.transform(res)
    print(actual)
    expected = DataFrame(
        [
            {
                "qid": "q1",
                "query": "a ##3",
                "query_toks": {"a": 1, "##3": 1},
                "docno": "d2",
                # "toks": {"a": 2, "##3": 1},
                "score": 0,
                "rank": 0,
            },
            {
                "qid": "q1",
                "query": "a ##3",
                "query_toks": {"a": 1, "##3": 1},
                "docno": "d1",
                # "toks": {"a": 1, "##2": 2},
                "score": -1,
                "rank": 1,
            },
        ]
    )
    print(expected)

    assert_frame_equal(
        actual.sort_values(by=["qid", "rank"]).reset_index(drop=True),
        expected.sort_values(by=["qid", "rank"]).reset_index(drop=True),
    )
