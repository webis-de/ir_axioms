from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed() or TYPE_CHECKING:
    from dataclasses import dataclass
    from functools import lru_cache, cached_property
    from pathlib import Path
    from typing import Union, Any

    from ir_axioms.tools.index_statistics import IndexStatistics
    from ir_axioms.utils.pyterrier import Index, IndexRef

    @dataclass(frozen=True, kw_only=True)
    class TerrierIndexStatistics(IndexStatistics):
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
        def _lexicon(self) -> Any:
            return self._index.getLexicon()

        @cached_property
        def _collection_statistics(self) -> Any:
            return self._index.getCollectionStatistics()

        @cached_property
        def document_count(self) -> int:  # type: ignore
            return int(self._collection_statistics.numberOfDocuments)

        @lru_cache(None)
        def document_frequency(self, term: str) -> int:  # type: ignore
            entry = self._lexicon.getLexiconEntry(term)
            if entry is None or entry.getNumberOfEntries() == 0:
                del entry
                return 0
            else:
                document_frequency = int(entry.getDocumentFrequency())
                del entry
                return document_frequency

else:
    TerrierIndexStatistics = NotImplemented  # type: ignore
