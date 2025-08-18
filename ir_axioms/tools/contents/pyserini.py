from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyserini_installed

if is_pyserini_installed() or TYPE_CHECKING:
    from dataclasses import dataclass
    from functools import cached_property
    from pathlib import Path
    from typing import Union

    from ir_axioms.model.retrieval import Document
    from ir_axioms.tools.contents.base import TextContents
    from ir_axioms.utils.pyserini import LuceneSearcher, get_searcher

    @dataclass(frozen=True, kw_only=True)
    class AnseriniDocumentTextContents(TextContents[Document]):
        index_dir: Union[Path, str]

        @cached_property
        def _searcher(self) -> LuceneSearcher:
            return get_searcher(self.index_dir)

        def contents(self, input: Document) -> str:
            if input.text is not None:
                return input.text
            document = self._searcher.doc(input.id)
            if document is None:
                raise KeyError(f"Document '{input.id}' not found in index.")
            return document.contents()

else:
    AnseriniDocumentTextContents = NotImplemented  # type: ignore
