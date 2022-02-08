# Check if PyTerrier is installed.
try:
    import pyterrier  # noqa: F401
    # Ensure that the Terrier JVM has started.
    from pyterrier import started

    assert started()
except ImportError as error:
    raise ImportError(
        "The PyTerrier backend requires that 'python-terrier' is installed."
    ) from error

from jnius import autoclass as _autoclass
from pyterrier.apply import generic as _generic
from pyterrier.index import IterDictIndexer as _IterDictIndexer
from pyterrier.transformer import (
    Transformer as _Transformer, TransformerBase as _TransformerBase,
    IdentityTransformer as _IdentityTransformer
)

# Re-export modules.
IterDictIndexer = _IterDictIndexer
Transformer = _Transformer
TransformerBase = _TransformerBase
IdentityTransformer = _IdentityTransformer
autoclass = _autoclass
generic = _generic
