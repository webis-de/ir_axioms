from axioms.utils.libraries import is_pyserini_installed

if is_pyserini_installed():

    from dataclasses import dataclass, field
    from functools import lru_cache, cached_property
    from pathlib import Path
    from typing import Union

    from pyserini.analysis import Analyzer, get_lucene_analyzer
    from pyserini.index import LuceneIndexReader

    from axioms.tools.index_statistics.base import IndexStatistics

    @dataclass(frozen=True, kw_only=True)
    class AnseriniIndexStatistics(IndexStatistics):
        index_dir: Union[Path, str]
        analyzer: Analyzer = field(
            default_factory=lambda: Analyzer(get_lucene_analyzer())
        )

        @cached_property
        def _index_reader(self) -> LuceneIndexReader:
            if isinstance(self.index_dir, Path):
                return LuceneIndexReader(str(self.index_dir.absolute()))
            elif isinstance(self.index_dir, str):
                return LuceneIndexReader(self.index_dir)

        @cached_property
        def document_count(self) -> int:
            return self._index_reader.stats()["documents"]

        @lru_cache(None)
        def document_frequency(self, term: str) -> int:
            document_frequency, _ = self._index_reader.get_term_counts(
                term=term,
                analyzer=self.analyzer.analyzer,
            )
            return document_frequency

else:
    AnseriniIndexStatistics = NotImplemented  # type: ignore
