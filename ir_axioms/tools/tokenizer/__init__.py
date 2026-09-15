from injector import Module, Binder, singleton

# Re-export from sub-modules.

from ir_axioms.tools.tokenizer.base import (
    TermTokenizer,
    SentenceTokenizer,
)

from ir_axioms.tools.tokenizer.nltk import (
    NltkTermTokenizer,
    NltkSentenceTokenizer,
)

from ir_axioms.tools.tokenizer.pyserini import (
    AnseriniTermTokenizer,
)

from ir_axioms.tools.tokenizer.pyterrier import (
    TerrierTermTokenizer,
)


from ir_axioms.tools.tokenizer.spacy import (
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


__all__ = [
    "TermTokenizer",
    "SentenceTokenizer",
    "NltkTermTokenizer",
    "NltkSentenceTokenizer",
    "AnseriniTermTokenizer",
    "TerrierTermTokenizer",
    "SpacyTermTokenizer",
    "SpacySentenceTokenizer",
    "TokenizerModule",
]
