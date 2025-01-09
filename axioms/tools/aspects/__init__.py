from injector import Module, Binder, singleton

# Re-export from sub-modules.

from axioms.tools.aspects.base import (  # noqa: F401
    AspectExtraction,
)

from axioms.tools.aspects.spacy import (  # noqa: F401
    SpacyNounChunksAspectExtraction,
)


class AspectsModule(Module):
    def configure(self, binder: Binder) -> None:
        binder.bind(
            interface=AspectExtraction,
            to=SpacyNounChunksAspectExtraction,
            scope=singleton,
        )
