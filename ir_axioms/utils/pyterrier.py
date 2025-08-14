from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed() or TYPE_CHECKING:
    from typing import Any, TypeVar

    from pyterrier.java import autoclass as pt_java_autoclass

    if TYPE_CHECKING:
        # Fix the wrong typing of PyTerrier's required decorator.
        T = TypeVar("T")

        def pt_java_required(fn: T) -> T:
            return fn
    else:
        from pyterrier.java import required as pt_java_required

    @pt_java_required
    def autoclass(*args, **kwargs) -> Any:
        return pt_java_autoclass(*args, **kwargs)

    Index = autoclass("org.terrier.structures.Index")
    IndexRef = autoclass("org.terrier.querying.IndexRef")
    Tokeniser = autoclass("org.terrier.indexing.tokenisation.Tokeniser")
    EnglishTokeniser = autoclass("org.terrier.indexing.tokenisation.EnglishTokeniser")
    TermPipelineAccessor = autoclass("org.terrier.terms.TermPipelineAccessor")
    BaseTermPipelineAccessor = autoclass("org.terrier.terms.BaseTermPipelineAccessor")
    ApplicationSetup = autoclass("org.terrier.utility.ApplicationSetup")

else:
    autoclass = NotImplemented
    Index = NotImplemented
    IndexRef = NotImplemented
    Tokeniser = NotImplemented
    EnglishTokeniser = NotImplemented
    TermPipelineAccessor = NotImplemented
    BaseTermPipelineAccessor = NotImplemented
    ApplicationSetup = NotImplemented
