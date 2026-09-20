from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Mapping, TypeVar

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
    # Cache of already-tokenized texts, keyed by the raw text content.
    # Excluded from equality/hashing (`compare=False`) so it doesn't affect
    # this (frozen) dataclass's identity, and mutated in place despite the
    # class being frozen (only attribute *reassignment* is blocked).
    _term_counts_cache: Dict[str, Mapping[str, int]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def term_counts(self, document: T) -> Mapping[str, int]:
        text = self.text_contents.contents(input=document)
        if isinstance(text, TokenizedString):
            return text.tokens
        cached = self._term_counts_cache.get(text)
        if cached is not None:
            return cached
        terms = self.term_tokenizer.terms_unordered(text)
        counts = Counter(terms)
        self._term_counts_cache[text] = counts
        return counts
