# Check if PyTerrier is installed.
try:
    import pyterrier  # noqa: F401
except ImportError as error:
    raise ImportError(
        "The PyTerrier backend requires that 'python-terrier' is installed."
    ) from error

# Ensure that the Terrier JVM has started.
from pyterrier import started

assert started()

# Re-export modules.
from pyterrier import IndexRef, IndexFactory  # noqa: F401
from pyterrier.index import Tokeniser, IterDictIndexer  # noqa: F401
from pyterrier.transformer import TransformerBase  # noqa: F401
from jnius import autoclass  # noqa: F401
