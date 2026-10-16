from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed() or TYPE_CHECKING:
    from dataclasses import dataclass
    from functools import cached_property
    from pathlib import Path
    from typing import Any

    from pyterrier.terrier import TerrierIndex

    from ir_axioms.tools.index_statistics import IndexStatistics

    @dataclass(frozen=True, kw_only=True)
    class TerrierIndexStatistics(IndexStatistics):
        index_location: TerrierIndex | Path | str | Any

        @cached_property
        def _index(self) -> TerrierIndex:
            return TerrierIndex.coerce(self.index_location)  # type: ignore

        @cached_property
        def _lexicon(self) -> Any:
            return self._index.lexicon()

        @cached_property
        def _collection_statistics(self) -> Any:
            return self._index.collection_statistics()

        @cached_property
        def document_count(self) -> int:  # type: ignore
            return int(self._collection_statistics.numberOfDocuments)

        @cached_property
        def _document_frequencies(self) -> dict[str, int]:
            return {}

        def document_frequency(self, term: str) -> int:
            if term not in self._document_frequencies:
                entry = self._lexicon.getLexiconEntry(term)
                if entry is None or entry.getNumberOfEntries() == 0:
                    document_frequency = 0
                else:
                    document_frequency = int(entry.getDocumentFrequency())
                self._document_frequencies[term] = document_frequency
            return self._document_frequencies[term]

else:
    TerrierIndexStatistics = NotImplemented  # type: ignore
