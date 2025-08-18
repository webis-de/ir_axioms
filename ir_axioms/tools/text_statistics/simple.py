from collections import Counter
from dataclasses import dataclass
from typing import Mapping, TypeVar

from injector import inject

from ir_axioms.model.utils import TokenizedString
from ir_axioms.tools.text_statistics.base import TextStatistics
from ir_axioms.tools.contents.base import TextContents
from ir_axioms.tools.tokenizer.base import TermTokenizer


T = TypeVar("T", contravariant=True)


@inject
@dataclass(frozen=True, kw_only=True)
class SimpleTextStatistics(TextStatistics[T]):
    text_contents: TextContents[T]
    term_tokenizer: TermTokenizer

    def term_counts(self, document: T) -> Mapping[str, int]:
        text = self.text_contents.contents(input=document)
        if isinstance(text, TokenizedString):
            return text.tokens
        terms = self.term_tokenizer.terms_unordered(text)
        return Counter(terms)
