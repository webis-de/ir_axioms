from dataclasses import dataclass, field
from functools import lru_cache
from math import nan
from pathlib import Path
from statistics import mean
from typing import Any, Final, Iterable, Dict, Optional, Union

from injector import inject, NoInject
from nltk import WordNetLemmatizer, sent_tokenize, word_tokenize
from targer_api import ArgumentSentences, ArgumentLabel, ArgumentTag, analyze_text
from targer_api.constants import DEFAULT_TARGER_MODELS, DEFAULT_TARGER_API_URL

from axioms.axiom.base import Axiom
from axioms.axiom.precondition import PreconditionMixin
from axioms.dependency_injection import injector
from axioms.precondition.base import Precondition
from axioms.precondition.length import LEN
from axioms.axiom.utils import strictly_greater, strictly_less
from axioms.model import Query, Document
from axioms.tools import TextContents, TermTokenizer
from axioms.utils.nltk import download_nltk_dependencies
from axioms.utils.lazy import lazy_inject


@lru_cache(None)
def _normalize(word: str):
    _word_net_lemmatizer = WordNetLemmatizer()
    return _word_net_lemmatizer.lemmatize(word).lower()


def _count_argumentative_units(sentences: ArgumentSentences) -> int:
    return _count_claims(sentences) + _count_premises(sentences)


def _count_premises(sentences: ArgumentSentences) -> int:
    return sum(
        1
        for sentence in sentences
        for tag in sentence
        if tag.label == ArgumentLabel.P_B and tag.probability > 0.5
    )


def _count_claims(sentences: ArgumentSentences) -> int:
    last_tag_was_claim: bool = False
    count: int = 0
    for sentence in sentences:
        for tag in sentence:
            if not last_tag_was_claim and _is_claim(tag) and tag.probability > 0.5:
                last_tag_was_claim = True
                count += 1
            elif last_tag_was_claim and _is_claim(tag) and tag.probability > 0.5:
                pass
            elif last_tag_was_claim and not _is_claim(tag):
                last_tag_was_claim = False
    return count


def _is_claim(tag: ArgumentTag) -> bool:
    return (
        tag.label == ArgumentLabel.C_B
        or tag.label == ArgumentLabel.C_I
        or tag.label == ArgumentLabel.MC_B
        or tag.label == ArgumentLabel.MC_I
    )


def _is_premise(tag: ArgumentTag) -> bool:
    return (
        tag.label == ArgumentLabel.P_B
        or tag.label == ArgumentLabel.P_I
        or tag.label == ArgumentLabel.MP_B
        or tag.label == ArgumentLabel.MP_I
    )


def _is_claim_or_premise(tag: ArgumentTag) -> bool:
    return _is_claim(tag) or _is_premise(tag)


def _count_query_terms(
    text_contents: Union[TextContents[Query], TextContents[Union[Query, Document]]],
    term_tokenizer: TermTokenizer,
    sentences: ArgumentSentences,
    query: Query,
    normalize: bool = True,
) -> int:
    term_count = 0
    for term in term_tokenizer.terms(text_contents.contents(query)):
        normalized_term = _normalize(term) if normalize else term
        for sentence in sentences:
            for tag in sentence:
                token = tag.token
                normalized_token = _normalize(token) if normalize else token
                if (
                    normalized_term == normalized_token
                    and _is_claim_or_premise(tag)
                    and tag.probability > 0.5
                ):
                    term_count += 1
    return term_count


def _query_term_position_in_argument(
    text_contents: Union[TextContents[Query], TextContents[Union[Query, Document]]],
    term_tokenizer: TermTokenizer,
    sentences: ArgumentSentences,
    query: Query,
    penalty: int,
    normalize: bool = True,
) -> float:
    term_argument_position = []
    tags = [tag for sentence in sentences for tag in sentence]
    for term in term_tokenizer.terms(text_contents.contents(query)):
        normalized_term = _normalize(term) if normalize else term
        found: bool = False
        for i, tag in enumerate(tags):
            position = i + 1
            token = tag.token
            normalized_token = _normalize(token) if normalize else token
            if (
                normalized_term == normalized_token
                and tag.label != ArgumentLabel.O
                and tag.probability > 0.5
            ):
                term_argument_position.append(position)
                found = True
                break
        if not found:
            term_argument_position.append(penalty)
    if len(term_argument_position) == 0:
        return penalty
    return mean(term_argument_position)


# TODO: Replace with sentence tokenizer tool.
def _sentence_length(
    text_contents: TextContents[Document],
    document: Document,
) -> float:
    download_nltk_dependencies("punkt")
    download_nltk_dependencies("punkt_tab")
    # TODO: Replace with interchangable sentence tokenizer tool.
    sentences = sent_tokenize(text_contents.contents(document))
    if len(sentences) == 0:
        return nan
    return mean(
        len(word_tokenize(sentence, preserve_line=True)) for sentence in sentences
    )


# TODO: Migrate TARGER usage to `ArgumentExtractor` tool that can be injected.
@dataclass(frozen=True, kw_only=True)
class _TargerMixin:
    models: NoInject[Iterable[str]] = DEFAULT_TARGER_MODELS
    api_url: NoInject[str] = DEFAULT_TARGER_API_URL
    cache_dir: NoInject[Optional[Path]] = None

    @lru_cache(None)
    def _analyze_text(
        self,
        contents: str,
    ) -> Dict[str, ArgumentSentences]:
        return analyze_text(
            contents,
            model_or_models=self.models,
            api_url=self.api_url,
            cache_dir=self.cache_dir,
        )

    def analyze_text(
        self,
        text_contents: Union[
            TextContents[Document], TextContents[Union[Query, Document]]
        ],
        document: Document,
    ) -> Dict[str, ArgumentSentences]:
        return self._analyze_text(text_contents.contents(document))


@inject
@dataclass(frozen=True, kw_only=True)
class ArgumentativeUnitsCountAxiom(
    PreconditionMixin[Any, Document], _TargerMixin, Axiom[Any, Document]
):
    """
    Favor documents with more argumentative units.
    """

    text_contents: TextContents[Document]
    precondition: NoInject[Precondition[Any, Document]] = field(default_factory=LEN)

    def preference(
        self,
        input: Any,
        output1: Document,
        output2: Document,
    ):
        arguments1 = self.analyze_text(self.text_contents, output1)
        arguments2 = self.analyze_text(self.text_contents, output2)

        count1 = sum(
            _count_argumentative_units(sentences) for _, sentences in arguments1.items()
        )
        count2 = sum(
            _count_argumentative_units(sentences) for _, sentences in arguments2.items()
        )

        return strictly_greater(count1, count2)


ArgUC: Final = lazy_inject(ArgumentativeUnitsCountAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class QueryTermOccurrenceInArgumentativeUnitsAxiom(
    PreconditionMixin[Query, Document], _TargerMixin, Axiom[Query, Document]
):
    """
    Favor documents with more query terms in argumentative units.
    """

    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer
    normalize: bool = True
    """
    Normalize query terms and tokens from argumentative units
    using the WordNet lemmatizer.
    """
    precondition: NoInject[Precondition[Any, Document]] = field(default_factory=LEN)

    def __post_init__(self):
        download_nltk_dependencies("wordnet", "omw-1.4")

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ):
        arguments1 = self.analyze_text(
            text_contents=self.text_contents,
            document=output1,
        )
        arguments2 = self.analyze_text(
            text_contents=self.text_contents,
            document=output2,
        )

        count1 = sum(
            _count_query_terms(
                text_contents=self.text_contents,
                term_tokenizer=self.term_tokenizer,
                sentences=sentences,
                query=input,
                normalize=self.normalize,
            )
            for _, sentences in arguments1.items()
        )
        count2 = sum(
            _count_query_terms(
                text_contents=self.text_contents,
                term_tokenizer=self.term_tokenizer,
                sentences=sentences,
                query=input,
                normalize=self.normalize,
            )
            for _, sentences in arguments2.items()
        )

        return strictly_greater(count1, count2)


QTArg: Final = lazy_inject(QueryTermOccurrenceInArgumentativeUnitsAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class QueryTermPositionInArgumentativeUnitsAxiom(
    PreconditionMixin[Query, Document], _TargerMixin, Axiom[Query, Document]
):
    """
    Favor documents where the first occurrence of a query term
    in an argumentative unit is closer to the beginning of the document.

    This axiom is based on the general observation that
    query terms occur “earlier” in relevant documents.

    References:
        Troy, A.D., Zhang, G.: Enhancing Relevance Scoring with Chronological
            Term Rank. In: Proceedings of SIGIR 2007. pp. 599–606. ACM.
        Mitra, B., Diaz, F., Craswell, N.: Learning to Match Using Local and
            Distributed Representations of Text for Web Search. In: Proceedings
            of WWW 2017. pp. 1291–1299. ACM.
    """

    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer
    normalize: bool = True
    """
    Normalize query terms and tokens from argumentative units
    using the WordNet lemmatizer.
    """
    penalty: Optional[int] = 100000
    """
    Penalty for the average query term position,
    if a query term is not found in any argumentative unit for a document.
    Set to None to use the maximum length of the two compared documents.
    """
    precondition: NoInject[Precondition[Any, Document]] = field(default_factory=LEN)

    def __post_init__(self):
        download_nltk_dependencies("wordnet", "omw-1.4")

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ):
        arguments1 = self.analyze_text(
            text_contents=self.text_contents,
            document=output1,
        )
        arguments2 = self.analyze_text(
            text_contents=self.text_contents,
            document=output1,
        )

        if len(arguments1) == 0 or len(arguments2) == 0:
            return 0

        penalty = self.penalty
        if penalty is None:
            penalty = (
                max(
                    len(
                        self.term_tokenizer.terms(
                            self.text_contents.contents(output1),
                        ),
                    ),
                    len(
                        self.term_tokenizer.terms(
                            self.text_contents.contents(output1),
                        ),
                    ),
                )
                + 1
            )

        position1 = mean(
            _query_term_position_in_argument(
                text_contents=self.text_contents,
                term_tokenizer=self.term_tokenizer,
                sentences=sentences,
                query=input,
                penalty=penalty,
                normalize=self.normalize,
            )
            for _, sentences in arguments1.items()
        )
        position2 = mean(
            _query_term_position_in_argument(
                text_contents=self.text_contents,
                term_tokenizer=self.term_tokenizer,
                sentences=sentences,
                query=input,
                penalty=penalty,
                normalize=self.normalize,
            )
            for _, sentences in arguments2.items()
        )

        return strictly_less(position1, position2)


QTPArg: Final = lazy_inject(QueryTermPositionInArgumentativeUnitsAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class AverageSentenceLengthAxiom(
    PreconditionMixin[Query, Document], Axiom[Any, Document]
):
    """
    Favor documents with an average sentence length between
    a minimum (default: 12) and a maximum (default: 20) number of words.

    This axiom is based on the general observation
    for text readability / good writing style [8, 10].

    References:
        Markel, M.: Technical Communication. 9th ed. Bedford/St Martin’s (2010)
        Newell, C.: Editing Tip: Sentence Length (2014)
    """

    text_contents: TextContents[Document]
    min_sentence_length: int = 12
    max_sentence_length: int = 20
    precondition: NoInject[Precondition[Any, Document]] = field(default_factory=LEN)

    def preference(
        self,
        input: Any,
        output1: Document,
        output2: Document,
    ):
        sentence_length1 = _sentence_length(self.text_contents, output1)
        sentence_length2 = _sentence_length(self.text_contents, output2)

        min_length = self.min_sentence_length
        max_length = self.max_sentence_length

        length_in_range1 = min_length <= sentence_length1 <= max_length
        length_in_range2 = min_length <= sentence_length2 <= max_length

        if length_in_range1 and not length_in_range2:
            return 1
        elif length_in_range2 and not length_in_range1:
            return -1
        else:
            return 0


aSLDoc: Final = lazy_inject(AverageSentenceLengthAxiom, injector)
aSL: Final = aSLDoc
