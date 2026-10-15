from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed() or TYPE_CHECKING:
    from collections.abc import Mapping, Iterable
    from dataclasses import dataclass
    from functools import cached_property
    from pathlib import Path
    from typing import Any

    from pyterrier.terrier import TerrierIndex

    from ir_axioms.model import Document
    from ir_axioms.tools.text_statistics.base import TextStatistics

    @dataclass(frozen=True, kw_only=True)
    class TerrierDocumentTextStatistics(TextStatistics[Document]):
        index_location: TerrierIndex | Path | str | Any

        @cached_property
        def _index(self) -> TerrierIndex:
            return TerrierIndex.coerce(self.index_location)  # type: ignore

        @cached_property
        def _meta_index(self) -> Any:
            meta_index = self._index.meta_index()
            if meta_index is None:
                raise ValueError(
                    f"Index {self.index_location} does not have a meta index."
                )
            return meta_index

        @cached_property
        def _document_index(self) -> Any:
            document_index = self._index.document_index()
            if document_index is None:
                raise ValueError(
                    f"Index {self.index_location} does not have a document index."
                )
            return document_index

        @cached_property
        def _direct_index(self) -> Any:
            direct_index = self._index.direct_index()
            if direct_index is None:
                raise ValueError(
                    f"Index {self.index_location} does not have a direct index."
                )
            return direct_index

        @cached_property
        def _lexicon(self) -> Any:
            return self._index.lexicon()

        def term_counts(self, document: Document) -> Mapping[str, int]:
            docid: int = self._meta_index.getDocument("docno", document.id)
            document_entry: Any = self._document_index.getDocumentEntry(docid)
            postings: Iterable[Any] | None
            postings = self._direct_index.getPostings(document_entry)
            if postings is None:
                return {}
            term_counts = {
                str(
                    self._lexicon.getLexiconEntry(
                        posting.getId(),
                    ).getKey()
                ): posting.getFrequency()
                for posting in postings
            }
            return term_counts

else:
    TerrierDocumentTextStatistics = NotImplemented  # type: ignore
