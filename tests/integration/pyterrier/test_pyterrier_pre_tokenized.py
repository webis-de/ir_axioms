from pyterrier import started, init

if not started():
    init(tqdm="auto", version="5.9", helper_version="0.0.8")

from pathlib import Path

from pytest import fixture, approx
from typing import List

from ir_axioms.backend.pyterrier import TerrierIndexContext
from ir_axioms.backend.pyterrier.safe import IterDictIndexer
from ir_axioms.model import Query, Document
from ir_axioms.model.context import IndexContext
from ir_axioms.model.retrieval_model import Tf, TfIdf, BM25, PL2, QL


@fixture
def documents() -> List:
    return [{'docno' : 'd1', 'toks' : {'a' : 1, '##2' : 2}, 'text': 'a ##2 ##2'}, {'docno' : 'd2', 'toks' : {'a' : 2, '##3' : 1}, 'text': 'a ##3 ##3'}]

@fixture
def index_dir(tmp_path: Path, documents: List) -> Path:
    index_dir = tmp_path
    indexer = IterDictIndexer(
        str(index_dir.absolute()),
        meta={"docno": 20, 'text': 5000},
        pretokenised=True
    )
    indexer.index(documents)
    return index_dir


@fixture
def reranking_context(index_dir: Path) -> IndexContext:
    return TerrierIndexContext(index_dir)


@fixture
def context(reranking_context: IndexContext) -> IndexContext:
    return reranking_context


@fixture
def query() -> Query:
    return Query("a ##3")

@fixture
def document1() -> Document:
    return Document(id="d1")

@fixture
def document2() -> Document:
    return Document(id="d2")

def test_query(query: Query):
    assert query.title == "a ##3"


def test_cache_dir(context: IndexContext):
    assert context.cache_dir is None or isinstance(context.cache_dir, Path)


def test_document_count(context: IndexContext):
    assert context.document_count == 2


def test_document_frequency(context: IndexContext):
    assert context.document_frequency("computer") == 0
    assert context.document_frequency("a") == 2
    assert context.document_frequency("##2") == 1
    assert context.document_frequency("##3") == 1


def test_inverse_document_frequency(context: IndexContext):
    assert context.inverse_document_frequency("computer") == 0
    assert context.inverse_document_frequency("a") == approx(3.067266)
    assert context.inverse_document_frequency("##2") == approx(3.357457)
    assert context.inverse_document_frequency("##3") == 0


def test_contents_query(
        context: IndexContext,
        query: Query
):
    assert context.contents(query) == "a ##3"


def test_terms_query(context: IndexContext, query: Query):
    assert context.terms(query) == ("a", "##3")


def test_terms_document1(context: IndexContext, document1: Document):
    assert context.terms(document1) == ("a", "##2", "##2")


def test_terms_document2(context: IndexContext, document2: Document):
    assert context.terms(document2) == ("a", "##3", "##3")


def test_term_set_query(context: IndexContext, query: Query):
    assert context.term_set(query) == {"a", "##3"}


def test_term_set_document1(
        context: IndexContext,
        document1: Document
):
    assert context.term_set(document1) == {"a", "##2"}


def test_term_set_document2(
        context: IndexContext,
        document2: Document
):
    assert context.term_set(document2) == {"a", "##3"}

X="""
def test_term_frequency_query(context: IndexContext, query: Query):
    assert context.term_frequency(query, "solv") == approx(1 / 3)
    assert context.term_frequency(query, "linear") == approx(1 / 3)
    assert context.term_frequency(query, "tree") == approx(0 / 3)


def test_term_frequency_document1(
        context: IndexContext,
        document1: Document
):
    assert context.term_frequency(document1, "compact") == approx(1 / 13)
    assert context.term_frequency(document1, "capac") == approx(2 / 13)
    assert context.term_frequency(document1, "tree") == approx(0 / 13)


def test_term_frequency_document2(
        context: IndexContext,
        document2: Document
):
    assert context.term_frequency(document2, "electron") == approx(1 / 16)
    assert context.term_frequency(document2, "analogu") == approx(1 / 16)
    assert context.term_frequency(document2, "tree") == approx(0 / 16)


def test_document1_retrieval_score(
        context: IndexContext,
        query: Query,
        document1: Document
):
    d1 = document1
    assert context.retrieval_score(query, d1, Tf()) == 0
    assert context.retrieval_score(query, d1, TfIdf()) == 0
    assert context.retrieval_score(query, d1, BM25()) == 0
    assert context.retrieval_score(query, d1, PL2()) == 0
    assert context.retrieval_score(query, d1, QL()) == 0


def test_document2_retrieval_score(
        context: IndexContext,
        query: Query,
        document2: Document
):
    d2 = document2
    # FIXME: How can we verify these numbers?
    assert context.retrieval_score(query, d2, Tf()) == 3
    assert context.retrieval_score(query, d2, TfIdf()) == approx(9.823447)
    assert context.retrieval_score(query, d2, BM25()) == approx(17.657991)
    assert context.retrieval_score(query, d2, PL2()) == approx(1.133970)
    assert context.retrieval_score(query, d2, QL()) == approx(1.278038)
"""