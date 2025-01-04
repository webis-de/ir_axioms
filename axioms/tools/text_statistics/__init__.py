from typing import Union

from injector import Module, Binder, singleton

from axioms.model import Document, Query

# Re-export from sub-modules.
from axioms.tools.text_statistics.base import (  # noqa: F401
    TextStatistics,
)

from axioms.tools.text_statistics.combine import (  # noqa: F401
    DocumentQueryTextStatistics,
)

from axioms.tools.text_statistics.pyserini import (  # noqa: F401
    AnseriniTextStatistics,
)

from axioms.tools.text_statistics.pyterrier import (  # noqa: F401
    TerrierTextStatistics,
)

from axioms.tools.text_statistics.simple import (  # noqa: F401
    SimpleTextStatistics,
)


class TextStatisticsModule(Module):
    def configure(self, binder: Binder) -> None:
        from axioms.tools import TextContents, TermTokenizer

        binder.bind(
            interface=TextStatistics[Query],
            to=lambda: SimpleTextStatistics(
                text_contents=binder.injector.get(TextContents[Query]),
                term_tokenizer=binder.injector.get(TermTokenizer),
            ),
            scope=singleton,
        )
        binder.bind(
            interface=TextStatistics[Document],
            to=lambda: SimpleTextStatistics(
                text_contents=binder.injector.get(TextContents[Document]),
                term_tokenizer=binder.injector.get(TermTokenizer),
            ),
            scope=singleton,
        )
        binder.bind(
            interface=TextStatistics[Union[Query, Document]],
            to=DocumentQueryTextStatistics,
            scope=singleton,
        )
