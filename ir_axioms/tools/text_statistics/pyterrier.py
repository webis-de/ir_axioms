from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed() or TYPE_CHECKING:
    from dataclasses import dataclass
    from functools import cached_property
    from pathlib import Path
    from typing import Union, Any, Mapping, Iterable, Optional, TypeVar

    from ir_axioms.model import Document
    from ir_axioms.tools.text_statistics.base import TextStatistics
    from ir_axioms.utils.pyterrier import Index, IndexRef

    DocumentType = TypeVar("DocumentType", bound=Document)

    @dataclass(frozen=True, kw_only=True)
    class TerrierTextStatistics(TextStatistics[DocumentType]):
        index_location: Union[Index, IndexRef, Path, str]  # type: ignore

        @cached_property
        def _index(self) -> Any:
            from pyterrier.terrier import IndexFactory

            if isinstance(self.index_location, Index):
                return self.index_location
            elif isinstance(self.index_location, IndexRef):
                return IndexFactory.of(self.index_location)  # type: ignore
            elif isinstance(self.index_location, str):
                return IndexFactory.of(self.index_location)  # type: ignore
            elif isinstance(self.index_location, Path):
                return IndexFactory.of(str(self.index_location.absolute()))  # type: ignore
            else:
                raise ValueError(
                    f"Cannot load index from location {self.index_location}."
                )

        @cached_property
        def _meta_index(self) -> Any:
            meta_index = self._index.getMetaIndex()
            if meta_index is None:
                raise ValueError(
                    f"Index {self.index_location} does not have a meta index."
                )
            return meta_index

        @cached_property
        def _document_index(self) -> Any:
            document_index = self._index.getDocumentIndex()
            if document_index is None:
                raise ValueError(
                    f"Index {self.index_location} does not have a document index."
                )
            return document_index

        @cached_property
        def _direct_index(self) -> Any:
            direct_index = self._index.getDirectIndex()
            if direct_index is None:
                raise ValueError(
                    f"Index {self.index_location} does not have a direct index."
                )
            return direct_index

        @cached_property
        def _lexicon(self) -> Any:
            return self._index.getLexicon()

        def term_counts(self, document: DocumentType) -> Mapping[str, int]:
            docid: int = self._meta_index.getDocument("docno", document.id)
            document_entry: Any = self._document_index.getDocumentEntry(docid)
            postings: Optional[Iterable[Any]] = self._direct_index.getPostings(
                document_entry
            )
            if postings is None:
                return {}
            return {
                self._lexicon.getLexiconEntry(
                    posting.getId(),
                ).getKey(): posting.getFrequency()
                for posting in postings
            }

else:
    TerrierTextStatistics = NotImplemented  # type: ignore
