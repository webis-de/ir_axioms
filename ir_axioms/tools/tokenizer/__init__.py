from injector import Module, Binder, singleton

from ir_axioms.utils.libraries import is_blingfire_installed

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

from ir_axioms.tools.tokenizer.blingfire import (
    BlingfireTermTokenizer,
    BlingfireSentenceTokenizer,
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
        if is_blingfire_installed():
            binder.bind(
                interface=TermTokenizer,
                to=BlingfireTermTokenizer,
                scope=singleton,
            )
            binder.bind(
                interface=SentenceTokenizer,
                to=BlingfireSentenceTokenizer,
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
    "BlingfireTermTokenizer",
    "BlingfireSentenceTokenizer",
    "TokenizerModule",
]
