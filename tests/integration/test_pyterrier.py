from pathlib import Path

from ir_datasets import load, Dataset
from pytest import fixture, approx

from ir_axioms.backend import PyTerrierBackendContext
from ir_axioms.model import Query, Document
from ir_axioms.model.context import RerankingContext
from ir_axioms.model.retrieval_model import Tf, TfIdf, BM25, PL2, QL

with PyTerrierBackendContext():
    from pyterrier.index import IterDictIndexer
    from ir_axioms.backend.pyterrier import IndexRerankingContext


    @fixture
    def dataset_name() -> str:
        return "vaswani"


    @fixture
    def dataset(dataset_name: str) -> Dataset:
        return load(dataset_name)


    @fixture
    def index_dir(tmp_path: Path, dataset: Dataset) -> Path:
        index_dir = tmp_path
        indexer = IterDictIndexer(str(index_dir.absolute()))
        indexer.index(
            {"docno": doc.doc_id, "text": doc.text}
            for doc in dataset.docs_iter()
        )
        return index_dir


    @fixture
    def reranking_context(index_dir: Path) -> RerankingContext:
        return IndexRerankingContext(index_dir)


    @fixture
    def context(reranking_context: RerankingContext) -> RerankingContext:
        return reranking_context


    @fixture
    def query() -> Query:
        return Query("solving linear equations")


    def test_query(query: Query):
        assert query.title == "solving linear equations"


    @fixture
    def document1(dataset: Dataset) -> Document:
        doc = dataset.docs_iter()[0]
        return Document(doc.doc_id, doc.text)


    def test_document1(document1: Document):
        assert document1.id == "1"
        assert document1.content == (
            "compact memories have flexible capacities  a digital data "
            "storage\nsystem with capacity up to bits and random and or "
            "sequential access\nis described\n"
        )


    @fixture
    def document2(dataset: Dataset) -> Document:
        doc = dataset.docs_iter()[1]
        return Document(doc.doc_id, doc.text)


    def test_document2(document2: Document):
        assert document2.id == "2"
        assert document2.content == (
            "an electronic analogue computer for solving systems of linear "
            "equations\nmathematical derivation of the operating principle "
            "and stability\nconditions for a computer consisting of "
            "amplifiers\n"
        )


    def test_cache_dir(context: RerankingContext):
        assert context.cache_dir is None or isinstance(context.cache_dir, Path)


    def test_document_count(context: RerankingContext):
        assert context.document_count == 11429


    def test_document_frequency(context: RerankingContext):
        assert context.document_frequency("computer") == 0
        assert context.document_frequency("comput") == 532
        assert context.document_frequency("linear") == 398
        assert context.document_frequency("digital") == 0
        assert context.document_frequency("digit") == 241


    def test_inverse_document_frequency(context: RerankingContext):
        assert context.inverse_document_frequency("computer") == 0
        assert context.inverse_document_frequency("comput") == approx(3.067266)
        assert context.inverse_document_frequency("linear") == approx(3.357457)
        assert context.inverse_document_frequency("digital") == 0
        assert context.inverse_document_frequency("digit") == approx(3.859112)


    def test_terms_query(context: RerankingContext, query: Query):
        assert context.terms(query) == ["solv", "linear", "equat"]


    def test_terms_document1(context: RerankingContext, document1: Document):
        assert context.terms(document1) == [
            'compact', 'memori', 'flexibl', 'capac', 'digit', 'data', 'storag',
            'system', 'capac', 'bit', 'random', 'sequenti', 'access'
        ]


    def test_terms_document2(context: RerankingContext, document2: Document):
        assert context.terms(document2) == [
            'electron', 'analogu', 'comput', 'solv', 'system', 'linear',
            'equat', 'mathemat', 'deriv', 'oper', 'principl', 'stabil',
            'condit', 'comput', 'consist', 'amplifi'
        ]


    def test_term_set_query(context: RerankingContext, query: Query):
        assert context.term_set(query) == {"solv", "linear", "equat"}


    def test_term_set_document1(
            context: RerankingContext,
            document1: Document
    ):
        assert context.term_set(document1) == {
            'compact', 'memori', 'flexibl', 'capac', 'digit', 'data', 'storag',
            'system', 'bit', 'random', 'sequenti', 'access'
        }


    def test_term_set_document2(
            context: RerankingContext,
            document2: Document
    ):
        assert context.term_set(document2) == {
            'electron', 'analogu', 'comput', 'solv', 'system', 'linear',
            'equat', 'mathemat', 'deriv', 'oper', 'principl', 'stabil',
            'condit', 'comput', 'consist', 'amplifi'
        }


    def test_term_frequency_query(context: RerankingContext, query: Query):
        assert context.term_frequency(query, "solv") == approx(1 / 3)
        assert context.term_frequency(query, "linear") == approx(1 / 3)
        assert context.term_frequency(query, "tree") == approx(0 / 3)


    def test_term_frequency_document1(
            context: RerankingContext,
            document1: Document
    ):
        assert context.term_frequency(document1, "compact") == approx(1 / 13)
        assert context.term_frequency(document1, "capac") == approx(2 / 13)
        assert context.term_frequency(document1, "tree") == approx(0 / 13)


    def test_term_frequency_document2(
            context: RerankingContext,
            document2: Document
    ):
        assert context.term_frequency(document2, "electron") == approx(1 / 16)
        assert context.term_frequency(document2, "analogu") == approx(1 / 16)
        assert context.term_frequency(document2, "tree") == approx(0 / 16)


    def test_document1_retrieval_score(
            context: RerankingContext,
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
            context: RerankingContext,
            query: Query,
            document2: Document
    ):
        d2 = document2
        assert context.retrieval_score(query, d2, Tf()) == 3
        assert context.retrieval_score(query, d2, TfIdf()) == approx(9.823447)
        assert context.retrieval_score(query, d2, BM25()) == approx(17.657991)
        assert context.retrieval_score(query, d2, PL2()) == approx(1.133970)
        assert context.retrieval_score(query, d2, QL()) == approx(1.278038)