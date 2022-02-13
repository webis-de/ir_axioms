# Check if Pyserini is installed.
try:
    import pyserini  # noqa: F401
except ImportError as error:
    raise ImportError(
        "The Pyserini backend requires that 'pyserini' is installed."
    ) from error

from pyserini.index import IndexReader as _IndexReader
from pyserini.search import SimpleSearcher as _SimpleSearcher
from jnius import autoclass as _autoclass

# Re-export modules.
IndexReader = _IndexReader
SimpleSearcher = _SimpleSearcher
autoclass = _autoclass
