from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed() or TYPE_CHECKING:
    from dataclasses import dataclass
    from functools import lru_cache, cached_property
    from pathlib import Path
    from typing import Union, Sequence, Any

    from typing_extensions import TypeAlias  # type: ignore

    from ir_axioms.model.retrieval import Document
    from ir_axioms.tools.contents.base import TextContents
    from ir_axioms.utils.pyterrier import Index, IndexRef

    _Index: TypeAlias = Index  # type: ignore
    _IndexRef: TypeAlias = IndexRef  # type: ignore

    @dataclass(frozen=True, kw_only=True)
    class TerrierDocumentTextContents(TextContents[Document]):
        index_location: Union[_Index, _IndexRef, Path, str]
        text_field: str = "text"

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
                    f"Index {self.index_location} does not have a metaindex."
                )
            return meta_index

        @cached_property
        def _meta_index_keys(self) -> Sequence[str]:
            return [str(key) for key in self._meta_index.getKeys()]

        @lru_cache(None)
        def _document_contents(self, document_id: str) -> str:
            if self.text_field not in self._meta_index_keys:
                raise ValueError(
                    f"Index {self.index_location} did not have "
                    f'requested metaindex key "{self.text_field}". '
                    f"Keys present in metaindex "
                    f"are {self._meta_index_keys}."
                )

            doc_id = int(self._meta_index.getDocument("docno", document_id))
            contents = str(
                self._meta_index.getItem(
                    self.text_field,
                    doc_id,
                )
            )
            return contents

        def contents(self, input: Document) -> str:
            if input.text is not None:
                return input.text
            return self._document_contents(input.id)

else:
    TerrierDocumentTextContents = NotImplemented  # type: ignore
