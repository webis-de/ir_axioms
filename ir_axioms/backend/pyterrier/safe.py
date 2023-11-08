# Check if PyTerrier is installed.
try:
    import pyterrier

    assert pyterrier
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
from pyterrier.datasets import IRDSDataset as _IRDSDataset
from pyterrier.transformer import (
    Transformer as _Transformer, TransformerBase as _TransformerBase,
    IdentityTransformer as _IdentityTransformer,
    EstimatorBase as _EstimatorBase
)

# Re-export modules.
IterDictIndexer = _IterDictIndexer
IRDSDataset = _IRDSDataset
Transformer = _Transformer
TransformerBase = _TransformerBase
IdentityTransformer = _IdentityTransformer
EstimatorBase = _EstimatorBase
autoclass = _autoclass
generic = _generic
