from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

from nltk import word_tokenize, WordNetLemmatizer, sent_tokenize

from ir_axioms.utils.nltk import download_nltk_dependencies
from ir_axioms.tools.tokenizer.base import TermTokenizer, SentenceTokenizer


@dataclass(frozen=True, kw_only=True)
class NltkTermTokenizer(TermTokenizer):
    lemmatize: bool = False
    lowercase: bool = False

    def __post_init__(self) -> None:
        download_nltk_dependencies("punkt")
        download_nltk_dependencies("punkt_tab")

    @cached_property
    def _lemmatizer(self) -> WordNetLemmatizer:
        download_nltk_dependencies("wordnet", "omw-1.4")
        return WordNetLemmatizer()

    def terms(self, text: str) -> Sequence[str]:
        terms = word_tokenize(text)
        if self.lemmatize:
            terms = [self._lemmatizer.lemmatize(term) for term in terms]
        if self.lowercase:
            terms = [term.lower() for term in terms]
        return terms


class NltkSentenceTokenizer(SentenceTokenizer):
    def __init__(self) -> None:
        download_nltk_dependencies("punkt")
        download_nltk_dependencies("punkt_tab")

    def sentences(self, text: str) -> Sequence[str]:
        return sent_tokenize(text)
