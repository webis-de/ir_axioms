# Check if Pyserini is installed.
try:
    import pyserini  # noqa: F401
except ImportError as error:
    raise ImportError(
        "The Pyserini backend requires that 'pyserini' is installed."
    ) from error

# Re-export modules.
from pyserini.index import IndexReader  # noqa: F401
from pyserini.search import SimpleSearcher  # noqa: F401
from jnius import autoclass  # noqa: F401
