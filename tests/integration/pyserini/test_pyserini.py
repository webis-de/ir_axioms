from json import dumps
from pathlib import Path
from subprocess import run

from ir_datasets import load, Dataset
from pytest import fixture, approx

from ir_axioms.backend.pyserini import PyseriniIndexContext
from ir_axioms.model import Query, Document
from ir_axioms.model.context import IndexContext
from ir_axioms.model.retrieval_model import TfIdf, BM25, PL2, QL


@fixture
def dataset_name() -> str:
    return "vaswani"


@fixture
def dataset(dataset_name: str) -> Dataset:
    return load(dataset_name)


@fixture
def dataset_dir(tmp_path: Path, dataset: Dataset) -> Path:
    dataset_dir = tmp_path

    dataset_file = dataset_dir / "docs.jsonl"
    with dataset_file.open("w") as file:
        for doc in dataset.docs_iter():
            doc_dict = {"id": doc.doc_id, "contents": doc.text}
            doc_line = dumps(doc_dict)
            file.write(f"{doc_line}\n")

    return dataset_dir


@fixture
def index_dir(tmp_path: Path, dataset_dir: Path) -> Path:
    index_dir = tmp_path

    index_command = [
        "python", "-m", "pyserini.index",
        "-collection", "JsonCollection",
        "-generator", "DefaultLuceneDocumentGenerator",
        "-input", str(dataset_dir.absolute()),
        "-index", str(index_dir.absolute()),
        "-threads", str(1),
        "-storePositions",
        "-storeDocvectors",
        "-storeRaw",
    ]
    run(index_command)

    return index_dir


@fixture
def reranking_context(index_dir: Path) -> IndexContext:
    return PyseriniIndexContext(index_dir)


@fixture
def context(reranking_context: IndexContext) -> IndexContext:
    return reranking_context


@fixture
def query() -> Query:
    return Query("solving linear equations")


def test_query(query: Query):
    assert query.title == "solving linear equations"


@fixture
def document1(dataset: Dataset) -> Document:
    doc = dataset.docs_iter()[0]
    return Document(doc.doc_id)


def test_document1(document1: Document):
    assert document1.id == "1"


@fixture
def document2(dataset: Dataset) -> Document:
    doc = dataset.docs_iter()[1]
    return Document(doc.doc_id)


def test_document2(document2: Document):
    assert document2.id == "2"


def test_cache_dir(context: IndexContext):
    assert context.cache_dir is None or isinstance(context.cache_dir, Path)


def test_document_count(context: IndexContext):
    assert context.document_count == 11429


def test_document_frequency(context: IndexContext):
    assert context.document_frequency("computer") == 0
    assert context.document_frequency("comput") == 532
    assert context.document_frequency("linear") == 398
    assert context.document_frequency("digital") == 0
    assert context.document_frequency("digit") == 241


def test_inverse_document_frequency(context: IndexContext):
    assert context.inverse_document_frequency("computer") == 0
    assert context.inverse_document_frequency("comput") == approx(3.067266)
    assert context.inverse_document_frequency("linear") == approx(3.357457)
    assert context.inverse_document_frequency("digital") == 0
    assert context.inverse_document_frequency("digit") == approx(3.859112)


def test_contents_query(
        context: IndexContext,
        query: Query
):
    assert context.contents(query) == "solving linear equations"


def test_contents_document1(
        context: IndexContext,
        document1: Document
):
    assert context.contents(document1) == (
        "compact memories have flexible capacities  a digital data "
        "storage\nsystem with capacity up to bits and random and or "
        "sequential access\nis described\n"
    )


def test_contents_document2(
        context: IndexContext,
        document2: Document
):
    assert context.contents(document2) == (
        "an electronic analogue computer for solving systems of linear "
        "equations\nmathematical derivation of the operating principle "
        "and stability\nconditions for a computer consisting of "
        "amplifiers\n"
    )


def test_terms_query(context: IndexContext, query: Query):
    assert context.terms(query) == ("solv", "linear", "equat")


def test_terms_document1(context: IndexContext, document1: Document):
    assert context.terms(document1) == (
        "compact", "memori", "have", "flexibl", "capac", "digit", "data",
        "storag", "system", "capac", "up", "bit", "random", "sequenti",
        "access", "describ"
    )


def test_terms_document2(context: IndexContext, document2: Document):
    assert context.terms(document2) == (
        "electron", "analogu", "comput", "solv", "system", "linear",
        "equat", "mathemat", "deriv", "oper", "principl", "stabil",
        "condit", "comput", "consist", "amplifi"
    )


def test_term_set_query(context: IndexContext, query: Query):
    assert context.term_set(query) == {"solv", "linear", "equat"}


def test_term_set_document1(
        context: IndexContext,
        document1: Document
):
    assert context.term_set(document1) == {
        "compact", "memori", "have", "flexibl", "capac", "digit", "data",
        "storag", "system", "up", "bit", "random", "sequenti", "access",
        "describ"
    }


def test_term_set_document2(
        context: IndexContext,
        document2: Document
):
    assert context.term_set(document2) == {
        "electron", "analogu", "comput", "solv", "system", "linear",
        "equat", "mathemat", "deriv", "oper", "principl", "stabil",
        "condit", "comput", "consist", "amplifi"
    }


def test_term_frequency_query(context: IndexContext, query: Query):
    assert context.term_frequency(query, "solv") == approx(1 / 3)
    assert context.term_frequency(query, "linear") == approx(1 / 3)
    assert context.term_frequency(query, "tree") == approx(0 / 3)


def test_term_frequency_document1(
        context: IndexContext,
        document1: Document
):
    assert context.term_frequency(document1, "compact") == approx(1 / 16)
    assert context.term_frequency(document1, "capac") == approx(2 / 16)
    assert context.term_frequency(document1, "tree") == approx(0 / 16)


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
    assert context.retrieval_score(query, d2, TfIdf()) == approx(3.425787)
    assert context.retrieval_score(query, d2, BM25()) == approx(5.829735)
    assert context.retrieval_score(query, d2, PL2()) == approx(2.822848)
    assert context.retrieval_score(query, d2, QL()) == approx(1.921952)
