from injector import Module, Binder, singleton

# Re-export from sub-modules.

from axioms.tools.tokenizer.base import (  # noqa: F401
    TermTokenizer,
)

from axioms.tools.tokenizer.nltk import (  # noqa: F401
    NltkTermTokenizer,
)

from axioms.tools.tokenizer.pyserini import (  # noqa: F401
    AnseriniTermTokenizer,
)

from axioms.tools.tokenizer.pyterrier import (  # noqa: F401
    TerrierTermTokenizer,
)


class TokenizerModule(Module):
    def configure(self, binder: Binder) -> None:
        binder.bind(
            TermTokenizer,
            to=NltkTermTokenizer,
            scope=singleton,
        )
