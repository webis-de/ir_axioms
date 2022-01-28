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

from jnius import autoclass as _autoclass  # noqa: F402
from pyterrier.index import (
    IndexRef as _IndexRef, IterDictIndexer as _IterDictIndexer,
    Tokeniser as _Tokeniser, IndexFactory as _IndexFactory,
    ApplicationSetup as _ApplicationSetup, StringReader as _StringReader
)  # noqa: F402
from pyterrier.transformer import Transformer as _Transformer  # noqa: F402

# Re-export modules.
IndexRef = _IndexRef
IterDictIndexer = _IterDictIndexer
Tokeniser = _Tokeniser
IndexFactory = _IndexFactory
ApplicationSetup = _ApplicationSetup
StringReader = _StringReader
Transformer = _Transformer
autoclass = _autoclass
