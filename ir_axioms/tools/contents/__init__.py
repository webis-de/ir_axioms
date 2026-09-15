from typing import Union

from injector import Module, Binder, singleton

from ir_axioms.model import Document, Query, GenerationInput, GenerationOutput

# Re-export from sub-modules.
from ir_axioms.tools.contents.base import (
    TextContents,
)

from ir_axioms.tools.contents.combine import (
    DocumentQueryTextContents,
    GenerationInputOutputTextContents,
)

from ir_axioms.tools.contents.ir_datasets import (
    IrdsDocumentTextContents,
    IrdsQueryTextContents,
)

from ir_axioms.tools.contents.pyserini import (
    AnseriniDocumentTextContents,
)

from ir_axioms.tools.contents.pyterrier import (
    TerrierDocumentTextContents,
)

from ir_axioms.tools.contents.simple import (
    HasText,
    SimpleTextContents,
)


class ContentsModule(Module):
    def configure(self, binder: Binder) -> None:
        binder.bind(
            interface=TextContents[HasText],
            to=SimpleTextContents,  # type: ignore
            scope=singleton,
        )
        binder.bind(
            interface=TextContents[Query],
            to=SimpleTextContents,  # type: ignore
            scope=singleton,
        )
        binder.bind(
            interface=TextContents[Document],
            to=SimpleTextContents,  # type: ignore
            scope=singleton,
        )
        binder.bind(
            interface=TextContents[GenerationInput],
            to=SimpleTextContents,  # type: ignore
            scope=singleton,
        )
        binder.bind(
            interface=TextContents[GenerationOutput],
            to=SimpleTextContents,  # type: ignore
            scope=singleton,
        )
        binder.bind(
            interface=TextContents[Union[Document, Query]],
            to=DocumentQueryTextContents,
            scope=singleton,
        )
        binder.bind(
            interface=TextContents[Union[GenerationInput, GenerationOutput]],
            to=GenerationInputOutputTextContents,
            scope=singleton,
        )


__all__ = [
    "TextContents",
    "DocumentQueryTextContents",
    "GenerationInputOutputTextContents",
    "IrdsDocumentTextContents",
    "IrdsQueryTextContents",
    "AnseriniDocumentTextContents",
    "TerrierDocumentTextContents",
    "HasText",
    "SimpleTextContents",
    "ContentsModule",
]
