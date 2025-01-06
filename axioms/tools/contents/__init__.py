from typing import Union

from injector import Module, Binder, singleton

from axioms.model import Document, Query, GenerationInput, GenerationOutput

# Re-export from sub-modules.
from axioms.tools.contents.base import (  # noqa: F401
    TextContents,
)

from axioms.tools.contents.combine import (  # noqa: F401
    DocumentQueryTextContents,
    GenerationInputOutputTextContents,
)

from axioms.tools.contents.ir_datasets import (  # noqa: F401
    IrdsDocumentTextContents,
    IrdsQueryTextContents,
)

from axioms.tools.contents.pyserini import (  # noqa: F401
    AnseriniDocumentTextContents,
)

from axioms.tools.contents.pyterrier import (  # noqa: F401
    TerrierDocumentTextContents,
)

from axioms.tools.contents.simple import (  # noqa: F401
    HasText,
    SimpleTextContents,
)


class ContentsModule(Module):
    def configure(self, binder: Binder) -> None:
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
