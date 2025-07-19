from injector import Module, Binder, singleton

# Re-export from sub-modules.

from ir_axioms.tools.aspects.base import (  # noqa: F401
    AspectExtraction,
)

from ir_axioms.tools.aspects.keybert import (  # noqa: F401
    KeyBertAspectExtraction,
)

from ir_axioms.tools.aspects.spacy import (  # noqa: F401
    SpacyNounChunksAspectExtraction,
    SpacyEntitiesAspectExtraction,
)


from ir_axioms.tools.aspects.textacy import (  # noqa: F401
    YakeAspectExtraction,
)


class AspectsModule(Module):
    def configure(self, binder: Binder) -> None:
        binder.bind(
            interface=AspectExtraction,
            to=SpacyNounChunksAspectExtraction,
            scope=singleton,
        )
