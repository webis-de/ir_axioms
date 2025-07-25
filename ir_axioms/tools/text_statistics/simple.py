from collections import Counter
from dataclasses import dataclass
from typing import Mapping, TypeVar

from injector import inject

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
        terms = self.term_tokenizer.terms(
            text=self.text_contents.contents(input=document),
        )
        return Counter(terms)
