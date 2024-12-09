from axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed():

    from dataclasses import dataclass
    from functools import lru_cache, cached_property
    from pathlib import Path
    from typing import Union, Any

    from pyterrier.java import (
        required as pt_java_required,
        autoclass as pt_java_autoclass,
    )
    from pyterrier.terrier import IndexFactory

    from axioms.tools.index_statistics import IndexStatistics

    @pt_java_required
    def autoclass(*args, **kwargs) -> Any:
        return pt_java_autoclass(*args, **kwargs)

    Index = autoclass("org.terrier.structures.Index")
    IndexRef = autoclass("org.terrier.querying.IndexRef")

    @dataclass(frozen=True, kw_only=True)
    class TerrierIndexStatistics(IndexStatistics):
        index_location: Union[Index, IndexRef, Path, str]  # type: ignore

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
        def _lexicon(self) -> Any:
            return self._index.getLexicon()

        @cached_property
        def _collection_statistics(self) -> Any:
            return self._index.getCollectionStatistics()

        @cached_property
        def document_count(self) -> int:
            return int(self._collection_statistics.numberOfDocuments)

        @lru_cache(None)
        def document_frequency(self, term: str) -> int:
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
