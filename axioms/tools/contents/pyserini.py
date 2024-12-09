from axioms.utils.libraries import is_pyserini_installed

if is_pyserini_installed():

    from dataclasses import dataclass
    from functools import cached_property
    from pathlib import Path
    from typing import Union

    from pyserini.search.lucene import LuceneSearcher

    from axioms.model.retrieval import Document
    from axioms.tools.contents.base import TextContents
    from axioms.tools.contents.simple import FallbackTextContentsMixin

    @dataclass(frozen=True, kw_only=True)
    class AnseriniDocumentTextContents(
        FallbackTextContentsMixin[Document], TextContents[Document]
    ):
        index_dir: Union[Path, str]

        @cached_property
        def _searcher(self) -> LuceneSearcher:
            if isinstance(self.index_dir, Path):
                return LuceneSearcher(str(self.index_dir.absolute()))
            elif isinstance(self.index_dir, str):
                return LuceneSearcher(self.index_dir)
            else:
                raise ValueError(f"Cannot load index from location {self.index_dir}.")

        def document_contents(self, input: Document) -> str:
            return self._searcher.doc(input.id).contents()

else:
    AnseriniDocumentTextContents = NotImplemented  # type: ignore
