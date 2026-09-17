from injector import Module, Binder, singleton

# Re-export from sub-modules.

from ir_axioms.tools.aspects.base import (
    AspectExtraction,
)

from ir_axioms.tools.aspects.keybert import (
    KeyBertAspectExtraction,
)

from ir_axioms.tools.aspects.spacy import (
    SpacyNounChunksAspectExtraction,
    SpacyEntitiesAspectExtraction,
)


from ir_axioms.tools.aspects.textacy import (
    YakeAspectExtraction,
)


class AspectsModule(Module):
    def configure(self, binder: Binder) -> None:
        binder.bind(
            interface=AspectExtraction,
            to=SpacyNounChunksAspectExtraction,
            scope=singleton,
        )


__all__ = [
    "AspectExtraction",
    "KeyBertAspectExtraction",
    "YakeAspectExtraction",
    "SpacyNounChunksAspectExtraction",
    "SpacyEntitiesAspectExtraction",
    "AspectsModule",
]
