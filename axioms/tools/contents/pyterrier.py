from axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed():

    from dataclasses import dataclass
    from functools import lru_cache, cached_property
    from pathlib import Path
    from typing import Union, Sequence, Any

    from pyterrier.java import (
        required as pt_java_required,
        autoclass as pt_java_autoclass,
    )
    from pyterrier.terrier import IndexFactory

    from axioms.model.retrieval import Document
    from axioms.tools.contents.base import TextContents
    from axioms.tools.contents.simple import FallbackTextContentsMixin

    @pt_java_required
    def autoclass(*args, **kwargs) -> Any:
        return pt_java_autoclass(*args, **kwargs)

    Index = autoclass("org.terrier.structures.Index")
    IndexRef = autoclass("org.terrier.querying.IndexRef")
    MetaIndex = autoclass("org.terrier.structures.MetaIndex")

    @pt_java_required
    @dataclass(frozen=True, kw_only=True)
    class TerrierDocumentTextContents(
        FallbackTextContentsMixin[Document], TextContents[Document]
    ):
        index_location: Union[Index, IndexRef, Path, str]  # type: ignore
        text_field: str = "text"

        @cached_property
        def _index(self) -> Any:
            if isinstance(self.index_location, Index):
                return self.index_location
            elif isinstance(self.index_location, IndexRef):
                return IndexFactory.of(self.index_location)
            elif isinstance(self.index_location, str):
                return IndexFactory.of(self.index_location)
            elif isinstance(self.index_location, Path):
                return IndexFactory.of(str(self.index_location.absolute()))
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

        def contents(self, inputocument) -> str:
            return self._document_contents(input)

else:
    TerrierDocumentTextContents = NotImplemented  # type: ignore
