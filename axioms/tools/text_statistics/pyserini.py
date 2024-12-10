from axioms.utils.libraries import is_pyserini_installed

if is_pyserini_installed():

    from dataclasses import dataclass, field
    from functools import cached_property
    from pathlib import Path
    from typing import Union, Mapping, TypeVar

    from pyserini.analysis import Analyzer, get_lucene_analyzer
    from pyserini.index import LuceneIndexReader

    from axioms.model import Document
    from axioms.tools.text_statistics.base import TextStatistics

    DocumentType = TypeVar("DocumentType", bound=Document)

    @dataclass(frozen=True, kw_only=True)
    class AnseriniTextStatistics(TextStatistics[DocumentType]):
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

        def term_counts(self, document: DocumentType) -> Mapping[str, int]:
            term_positions = self._index_reader.get_term_positions(document.id)
            return {term: len(positions) for term, positions in term_positions}

        def term_frequencies(self, document: DocumentType) -> Mapping[str, float]:
            term_frequencies = self._index_reader.get_document_vector(document.id)
            if term_frequencies is None:
                return {}
            return term_frequencies

else:
    AnseriniTextStatistics = NotImplemented  # type: ignore
