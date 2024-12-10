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


from axioms.tools.tokenizer.spacy import (  # noqa: F401
    SpacyTermTokenizer,
    SpacySentenceTokenizer,
)


class TokenizerModule(Module):
    def configure(self, binder: Binder) -> None:
        binder.bind(
            interface=TermTokenizer,
            to=SpacyTermTokenizer,
            scope=singleton,
        )
        binder.bind(
            interface=SentenceTokenizer,
            to=SpacySentenceTokenizer,
            scope=singleton,
        )
