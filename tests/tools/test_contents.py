from ir_datasets import load as irds_load
from pytest import raises

from ir_axioms.model import (
    Document,
    GenerationRequest,
    GenerationResponse,
    Query,
)
from ir_axioms.tools.contents import (
    DocumentQueryTextContents,
    GenerationInputOutputTextContents,
    IrdsDocumentTextContents,
    IrdsQueryTextContents,
    SimpleTextContents,
    TextContents,
)


class _PrefixTextContents(TextContents):
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def contents(self, input) -> str:
        return f"{self.prefix}:{input.id}"


def test_simple_returns_text() -> None:
    contents = SimpleTextContents()

    assert contents.contents(Document(id="d1", text="some text")) == "some text"
    assert contents.contents(Query(id="q1", text="query")) == "query"


def test_simple_returns_empty_text() -> None:
    assert SimpleTextContents().contents(Document(id="d1", text="")) == ""


def test_simple_raises_without_text() -> None:
    with raises(ValueError):
        SimpleTextContents().contents(Document(id="d1"))


def test_document_query_dispatches_by_type() -> None:
    contents = DocumentQueryTextContents(
        document_text_contents=_PrefixTextContents("doc"),
        query_text_contents=_PrefixTextContents("query"),
    )

    assert contents.contents(Document(id="d1")) == "doc:d1"
    assert contents.contents(Query(id="q1")) == "query:q1"


def _generation_input_output_text_contents() -> GenerationInputOutputTextContents:
    return GenerationInputOutputTextContents(
        generation_input_text_contents=_PrefixTextContents("request"),
        generation_output_text_contents=_PrefixTextContents("response"),
    )


def test_generation_request_response_dispatches_by_type() -> None:
    contents = _generation_input_output_text_contents()

    assert contents.contents(GenerationRequest(id="i1", text="")) == "request:i1"
    assert contents.contents(GenerationResponse(id="o1", text="")) == "response:o1"


def test_irds_query_looks_up_text() -> None:
    dataset = irds_load("cranfield")
    contents = IrdsQueryTextContents(dataset=dataset)

    assert (
        contents.contents(Query(id="132")) == "theoretical studies of creep buckling ."
    )
    assert (
        contents.contents(Query(id="185")) == "experimental studies on panel flutter ."
    )


def test_irds_query_prefers_given_text() -> None:
    dataset = irds_load("cranfield")
    contents = IrdsQueryTextContents(dataset=dataset)

    assert contents.contents(Query(id="132", text="override")) == "override"
    assert contents.contents(Query(id="185", text="override")) == "override"


def test_irds_query_unknown_id_raises() -> None:
    dataset = irds_load("cranfield")
    contents = IrdsQueryTextContents(dataset=dataset)

    with raises(KeyError):
        contents.contents(Query(id="unknown"))


def test_irds_document_looks_up_text() -> None:
    dataset = irds_load("cranfield")
    contents = IrdsDocumentTextContents(dataset=dataset)

    assert contents.contents(Document(id="3")) == (
        "the boundary layer in simple shear flow past a flat plate . "
        "the boundary layer in simple shear flow past a flat plate .\n"
        "the boundary-layer equations are presented for steady\n"
        "incompressible flow with no pressure gradient ."
    )
    assert contents.contents(Document(id="405")) == (
        "tables of thermal properties of gases . "
        "tables of thermal properties of gases .\n"
        "tables of thermodynamic and transport properties\n"
        "of air, argon, carbon dioxide, carbon monoxide, hydrogen,\n"
        "nitrogen, oxygen, and steam ."
    )


def test_irds_document_prefers_given_text() -> None:
    dataset = irds_load("cranfield")
    contents = IrdsDocumentTextContents(dataset=dataset)

    assert contents.contents(Document(id="3", text="override")) == "override"
    assert contents.contents(Document(id="405", text="override")) == "override"


def test_irds_document_unknown_id_raises() -> None:
    dataset = irds_load("cranfield")
    contents = IrdsDocumentTextContents(dataset=dataset)

    with raises(KeyError):
        contents.contents(Document(id="unknown"))


def test_irds_document_iter_contents_matches_contents() -> None:
    dataset = irds_load("cranfield")
    contents = IrdsDocumentTextContents(dataset=dataset)
    documents = [
        Document(id="405"),
        Document(id="3", text="override"),
        Document(id="3"),
        Document(id="405"),
    ]

    assert list(contents.iter_contents(documents)) == [
        contents.contents(document) for document in documents
    ]


def test_irds_document_iter_contents_unknown_id_raises() -> None:
    dataset = irds_load("cranfield")
    contents = IrdsDocumentTextContents(dataset=dataset)

    with raises(KeyError):
        list(contents.iter_contents([Document(id="3"), Document(id="unknown")]))
