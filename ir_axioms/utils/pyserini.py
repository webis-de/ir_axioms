from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyserini_installed

if is_pyserini_installed() or TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, TypeAlias, Union

    if TYPE_CHECKING:
        from pyserini.analysis import Analyzer
        from pyserini.index import LuceneIndexReader
        from pyserini.search.lucene import LuceneSearcher
    else:
        # Do not load the types in runtype, as importing will immediately start the JRE.
        Analyzer: TypeAlias = Any
        LuceneIndexReader: TypeAlias = Any
        LuceneSearcher: TypeAlias = Any

    def default_analyzer() -> Analyzer:
        from pyserini.analysis import Analyzer, get_lucene_analyzer

        return Analyzer(get_lucene_analyzer())

    def get_index_reader(index_dir: Union[Path, str]) -> LuceneIndexReader:
        from pyserini.index import LuceneIndexReader

        if isinstance(index_dir, Path):
            return LuceneIndexReader(str(index_dir.absolute()))
        else:
            return LuceneIndexReader(index_dir)

    def get_searcher(index_dir: Union[Path, str]) -> LuceneSearcher:
        from pyserini.search.lucene import LuceneSearcher

        if isinstance(index_dir, Path):
            return LuceneSearcher(str(index_dir.absolute()))
        else:
            return LuceneSearcher(index_dir)
else:
    default_analyzer = NotImplemented
    get_index_reader = NotImplemented
    get_searcher = NotImplemented
