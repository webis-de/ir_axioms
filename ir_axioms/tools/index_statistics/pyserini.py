from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyserini_installed

if is_pyserini_installed() or TYPE_CHECKING:
    from dataclasses import dataclass, field
    from functools import lru_cache, cached_property
    from pathlib import Path
    from typing import Union

    from ir_axioms.tools.index_statistics.base import IndexStatistics
    from ir_axioms.utils.pyserini import (
        Analyzer,
        default_analyzer,
        LuceneIndexReader,
        get_index_reader,
    )

    @dataclass(frozen=True, kw_only=True)
    class AnseriniIndexStatistics(IndexStatistics):
        index_dir: Union[Path, str]
        analyzer: Analyzer = field(default_factory=default_analyzer)

        @cached_property
        def _index_reader(self) -> LuceneIndexReader:
            return get_index_reader(self.index_dir)

        @cached_property
        def document_count(self) -> int:  # type: ignore
            return self._index_reader.stats()["documents"]

        @lru_cache(None)
        def document_frequency(self, term: str) -> int:  # type: ignore
            document_frequency, _ = self._index_reader.get_term_counts(
                term=term,
                analyzer=self.analyzer.analyzer,
            )
            return document_frequency

else:
    AnseriniIndexStatistics = NotImplemented
