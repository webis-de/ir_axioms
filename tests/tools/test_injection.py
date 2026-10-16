from collections.abc import Iterable
from inspect import isfunction
from typing import Any, get_origin

from injector import Binder, Injector
from pytest import fixture, mark

from ir_axioms.model import Document, GenerationRequest, GenerationResponse, Query
from ir_axioms.tools import (
    AspectExtraction,
    DocumentQueryTextContents,
    DocumentQueryTextStatistics,
    HasText,
    PivotSelection,
    SentenceSimilarity,
    SentenceTokenizer,
    TermSimilarity,
    TermTokenizer,
    TextContents,
    TextStatistics,
    ToolsModule,
)
from ir_axioms.tools.contents import GenerationInputOutputTextContents
from ir_axioms.tools.text_statistics import GenerationInputOutputTextStatistics

# All interfaces bound by the `ToolsModule`.
_INTERFACES: Iterable[type] = (
    AspectExtraction,
    TextContents[HasText],
    TextContents[Query],
    TextContents[Document],
    TextContents[GenerationRequest],
    TextContents[GenerationResponse],
    TextContents[Document | Query],
    TextContents[GenerationRequest | GenerationResponse],
    PivotSelection,
    TermSimilarity,
    SentenceSimilarity,
    TermTokenizer,
    SentenceTokenizer,
    TextStatistics[Query],
    TextStatistics[Document],
    TextStatistics[GenerationRequest],
    TextStatistics[GenerationResponse],
    TextStatistics[Document | Query],
    TextStatistics[GenerationRequest | GenerationResponse],
)

# Interfaces that combine multiple single-type bindings and must therefore
# be bound to the respective combining implementation.
_COMBINED_BINDINGS: Iterable[tuple[type, type]] = (
    (TextContents[Document | Query], DocumentQueryTextContents),
    (
        TextContents[GenerationRequest | GenerationResponse],
        GenerationInputOutputTextContents,
    ),
    (TextStatistics[Document | Query], DocumentQueryTextStatistics),
    (
        TextStatistics[GenerationRequest | GenerationResponse],
        GenerationInputOutputTextStatistics,
    ),
)


def _stub() -> None: ...


def _is_stub(member: Any) -> bool:
    return isfunction(member) and member.__code__.co_code == _stub.__code__.co_code


@fixture(scope="module")
def injector() -> Injector:
    return Injector([ToolsModule()])


def test_interfaces_are_complete(injector: Injector) -> None:
    # If this test fails, a binding was added to (or removed from) one of the
    # modules installed by `ToolsModule` (`ir_axioms/tools/*/__init__.py`).
    # Update `_INTERFACES` above accordingly. If the new binding combines
    # multiple single-type bindings (e.g., `TextContents[Union[...]]`), also
    # add it to `_COMBINED_BINDINGS`. Then update the "Tool Bindings" table in
    # `docs/axiom-development.md`.
    bound = set(injector.binder._bindings) - {Injector, Binder}
    assert bound == set(_INTERFACES), (
        "Bindings of `ToolsModule` do not match `_INTERFACES`. "
        "See the comment in this test for where to update them."
    )


@mark.parametrize("interface", _INTERFACES, ids=str)
def test_bindings_implement_interfaces(injector: Injector, interface: type) -> None:
    interface_origin = get_origin(interface) or interface
    implementation: type[Any] = type(injector.get(interface))

    assert isinstance(injector.get(interface), interface_origin)

    # TODO: Replace the below hacked stuff with clean python stdlib calls. If not possible, ask me.
    assert not getattr(implementation, "_is_protocol", False)
    unimplemented = {
        name
        for name, member in vars(interface_origin).items()
        if _is_stub(member) and getattr(implementation, name) is member
    }
    assert not unimplemented


@mark.parametrize(
    ("interface", "implementation"),
    _COMBINED_BINDINGS,
    ids=[str(interface) for interface, _ in _COMBINED_BINDINGS],
)
def test_combined_bindings(
    injector: Injector,
    interface: type,
    implementation: type,
) -> None:
    assert isinstance(injector.get(interface), implementation)


@mark.parametrize("interface", _INTERFACES, ids=str)
def test_bindings_are_singletons(injector: Injector, interface: Any) -> None:
    assert injector.get(interface) is injector.get(interface)
