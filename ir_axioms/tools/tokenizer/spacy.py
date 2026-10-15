from dataclasses import dataclass
from functools import cached_property
from collections.abc import Collection, Iterable, Iterator, Sequence, Set

from spacy import load as spacy_load
from spacy.language import Language
from spacy.tokens import Doc, Token

from ir_axioms.tools.tokenizer.base import TermTokenizer, SentenceTokenizer


@dataclass(frozen=True, kw_only=True)
class SpacyTermTokenizer(TermTokenizer):
    language_name: str = "en_core_web_sm"
    lemmatize: bool = True
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True

    @cached_property
    def _language(self) -> Language:
        return spacy_load(
            name=self.language_name,
            # exclude=["parser", "ner"] + (["lemmatizer"] if not self.lemmatize else []),
        )

    def terms(self, text: str) -> Sequence[str]:
        return self._terms(self._language(text))

    def iter_terms(self, texts: Iterable[str]) -> Iterator[Sequence[str]]:
        for document in self._language.pipe(texts):
            yield self._terms(document)

    def iter_terms_unordered(self, texts: Iterable[str]) -> Iterator[Collection[str]]:
        return self.iter_terms(texts)

    def iter_unique_terms(self, texts: Iterable[str]) -> Iterator[Set[str]]:
        for terms in self.iter_terms(texts):
            yield set(terms)

    def _terms(self, document: Doc) -> Sequence[str]:
        tokens: Iterable[Token] = (token for token in document)
        if self.remove_punctuation:
            tokens = (token for token in tokens if not token.is_punct)
        if self.remove_stopwords:
            tokens = (token for token in tokens if not token.is_stop)

        terms: Iterable[str]
        if self.lemmatize:
            terms = (token.lemma_ for token in tokens)
        else:
            terms = (token.text for token in tokens)
        if self.lowercase:
            terms = (term.lower() for term in terms)

        return list(terms)


@dataclass(frozen=True, kw_only=True)
class SpacySentenceTokenizer(SentenceTokenizer):
    language_name: str = "en_core_web_sm"

    @cached_property
    def _language(self) -> Language:
        return spacy_load(
            name=self.language_name,
            # exclude=["parser", "ner", "lemmatizer"],
            # enable=["sentencizer"],
        )

    def sentences(self, text: str) -> Sequence[str]:
        document = self._language(text)
        return [sentence.text for sentence in document.sents]

    def iter_sentences(self, texts: Iterable[str]) -> Iterator[Sequence[str]]:
        for document in self._language.pipe(texts):
            yield [sentence.text for sentence in document.sents]
