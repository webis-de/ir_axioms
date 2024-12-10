from injector import Module, Binder, singleton

# Re-export from sub-modules.

from axioms.tools.tokenizer.base import (  # noqa: F401
    TermTokenizer,
    SentenceTokenizer,
)

from axioms.tools.tokenizer.nltk import (  # noqa: F401
    NltkTermTokenizer,
    NltkSentenceTokenizer,
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
            interface=TermTokenizer,
            to=NltkTermTokenizer,
            scope=singleton,
        )
        binder.bind(
            interface=SentenceTokenizer,
            to=NltkSentenceTokenizer,
            scope=singleton,
        )
