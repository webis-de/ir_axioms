from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyserini_installed

if is_pyserini_installed() or TYPE_CHECKING:
    from dataclasses import dataclass, field
    from functools import cached_property
    from pathlib import Path
    from typing import Union, Mapping, TypeVar

    from ir_axioms.model import Document
    from ir_axioms.tools.text_statistics.base import TextStatistics
    from ir_axioms.utils.pyserini import (
        Analyzer,
        default_analyzer,
        LuceneIndexReader,
        get_index_reader,
    )

    DocumentType = TypeVar("DocumentType", bound=Document)

    @dataclass(frozen=True, kw_only=True)
    class AnseriniTextStatistics(TextStatistics[DocumentType]):
        index_dir: Union[Path, str]
        analyzer: Analyzer = field(default_factory=default_analyzer)

        @cached_property
        def _index_reader(self) -> LuceneIndexReader:
            return get_index_reader(self.index_dir)

        def term_counts(self, document: DocumentType) -> Mapping[str, int]:
            term_positions = self._index_reader.get_term_positions(document.id)
            if term_positions is None:
                raise KeyError(f"Document '{document.id}' not found in index.")
            return {term: len(positions) for term, positions in term_positions}

        def term_frequencies(self, document: DocumentType) -> Mapping[str, float]:
            term_frequencies = self._index_reader.get_document_vector(document.id)
            if term_frequencies is None:
                return {}
            return term_frequencies

else:
    AnseriniTextStatistics = NotImplemented
