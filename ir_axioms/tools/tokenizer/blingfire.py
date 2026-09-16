from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_blingfire_installed

if is_blingfire_installed() or TYPE_CHECKING:
    from dataclasses import dataclass
    from typing import Sequence

    from blingfire import text_to_words, text_to_sentences

    from ir_axioms.tools.tokenizer.base import TermTokenizer, SentenceTokenizer

    @dataclass(frozen=True, kw_only=True)
    class BlingfireTermTokenizer(TermTokenizer):
        remove_punctuation: bool = True

        def terms(self, text: str) -> Sequence[str]:
            return [
                word
                for word in text_to_words(text).split()
                if not self.remove_punctuation or any(char.isalnum() for char in word)
            ]

    @dataclass(frozen=True, kw_only=True)
    class BlingfireSentenceTokenizer(SentenceTokenizer):
        language_name: str = "en_core_web_sm"

        def sentences(self, text: str) -> Sequence[str]:
            return text_to_sentences(text).splitlines(keepends=False)

else:
    BlingfireTermTokenizer = NotImplemented  # type: ignore
    BlingfireSentenceTokenizer = NotImplemented  # type: ignore
