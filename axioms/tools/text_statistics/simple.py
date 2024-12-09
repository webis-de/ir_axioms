from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Protocol, TypeVar, runtime_checkable

from injector import inject

from axioms.tools.text_statistics.base import TextStatistics
from axioms.tools.contents.base import TextContents
from axioms.tools.tokenizer.base import TermTokenizer


@runtime_checkable
class HasText(Protocol):
    text: str


T = TypeVar("T", contravariant=True)


@inject
@dataclass(frozen=True, kw_only=True)
class SimpleTextStatistics(TextStatistics[T]):
    text_contents: TextContents[T]
    term_tokenizer: TermTokenizer

    def term_frequencies(self, document: T) -> Mapping[str, int]:
        terms = self.term_tokenizer.terms(
            text=self.text_contents.contents(input=document),
        )
        return Counter(terms)
