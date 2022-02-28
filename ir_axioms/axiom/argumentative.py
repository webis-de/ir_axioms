from dataclasses import dataclass, field
from functools import lru_cache
from math import nan
from pathlib import Path
from statistics import mean
from typing import Set, Dict, Optional

from nltk import WordNetLemmatizer, sent_tokenize, word_tokenize
from targer_api import (
    ArgumentSentences, ArgumentLabel, ArgumentTag, analyze_text
)
from targer_api.constants import DEFAULT_TARGER_MODELS, DEFAULT_TARGER_API_URL

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.preconditions import LEN_Mixin
from ir_axioms.model import Query, RankedDocument, IndexContext
from ir_axioms.utils.nltk import download_nltk_dependencies


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
            if (
                    not last_tag_was_claim and
                    _is_claim(tag) and
                    tag.probability > 0.5
            ):
                last_tag_was_claim = True
                count += 1
            elif (
                    last_tag_was_claim and
                    _is_claim(tag) and
                    tag.probability > 0.5
            ):
                pass
            elif last_tag_was_claim and not _is_claim(tag):
                last_tag_was_claim = False
    return count


def _is_claim(tag: ArgumentTag) -> bool:
    return (
            tag.label == ArgumentLabel.C_B or
            tag.label == ArgumentLabel.C_I or
            tag.label == ArgumentLabel.MC_B or
            tag.label == ArgumentLabel.MC_I
    )


def _is_premise(tag: ArgumentTag) -> bool:
    return (
            tag.label == ArgumentLabel.P_B or
            tag.label == ArgumentLabel.P_I or
            tag.label == ArgumentLabel.MP_B or
            tag.label == ArgumentLabel.MP_I
    )


def _is_claim_or_premise(tag: ArgumentTag) -> bool:
    return _is_claim(tag) or _is_premise(tag)


def _count_query_terms(
        context: IndexContext,
        sentences: ArgumentSentences,
        query: Query,
        normalize: bool = True,
) -> int:
    term_count = 0
    for term in context.terms(query):
        normalized_term = _normalize(term) if normalize else term
        for sentence in sentences:
            for tag in sentence:
                token = tag.token
                normalized_token = _normalize(token) if normalize else token
                if (
                        normalized_term == normalized_token and
                        _is_claim_or_premise(tag) and
                        tag.probability > 0.5
                ):
                    term_count += 1
    return term_count


def _query_term_position_in_argument(
        context: IndexContext,
        sentences: ArgumentSentences,
        query: Query,
        penalty: int,
        normalize: bool = True,
) -> float:
    term_argument_position = []
    tags = [tag for sentence in sentences for tag in sentence]
    for term in context.terms(query):
        normalized_term = _normalize(term) if normalize else term
        found: bool = False
        for i, tag in enumerate(tags):
            position = i + 1
            token = tag.token
            normalized_token = _normalize(token) if normalize else token
            if (
                    normalized_term == normalized_token and
                    tag.label != ArgumentLabel.O and
                    tag.probability > 0.5
            ):
                term_argument_position.append(position)
                found = True
                break
        if not found:
            term_argument_position.append(penalty)
    if len(term_argument_position) == 0:
        return penalty
    return mean(term_argument_position)


@lru_cache(None)
def _sentence_length(
        context: IndexContext,
        document: RankedDocument,
) -> float:
    download_nltk_dependencies("punkt")
    sentences = sent_tokenize(context.contents(document))
    if len(sentences) == 0:
        return nan
    return mean(
        len(word_tokenize(sentence, preserve_line=True))
        for sentence in sentences
    )


@dataclass(frozen=True)
class _TargerMixin:
    models: Set[str] = field(default_factory=lambda: DEFAULT_TARGER_MODELS)
    api_url: str = DEFAULT_TARGER_API_URL

    @lru_cache(None)
    def _analyze_text(
            self,
            contents: str,
            cache_dir: Optional[Path],
    ) -> Dict[str, ArgumentSentences]:
        return analyze_text(
            contents,
            model_or_models=self.models,
            api_url=self.api_url,
            cache_dir=(
                cache_dir / "targer"
                if cache_dir is not None
                else None
            )
        )

    def analyze_text(
            self,
            context: IndexContext,
            document: RankedDocument,
    ) -> Dict[str, ArgumentSentences]:
        return self._analyze_text(
            context.contents(document),
            context.cache_dir
        )


@dataclass(frozen=True)
class ArgumentativeUnitsCountAxiom(_TargerMixin, Axiom):
    """
    Favor documents with more argumentative units.
    """

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        arguments1 = self.analyze_text(context, document1)
        arguments2 = self.analyze_text(context, document2)

        count1 = sum(
            _count_argumentative_units(sentences)
            for _, sentences in arguments1.items()
        )
        count2 = sum(
            _count_argumentative_units(sentences)
            for _, sentences in arguments2.items()
        )

        if count1 > count2:
            return 1
        elif count1 < count2:
            return -1
        else:
            return 0


@dataclass(frozen=True)
class ArgUC(LEN_Mixin, ArgumentativeUnitsCountAxiom):
    name = "ArgUC"


@dataclass(frozen=True)
class QueryTermOccurrenceInArgumentativeUnitsAxiom(_TargerMixin, Axiom):
    """
    Favor documents with more query terms in argumentative units.
    """

    normalize: bool = True
    """
    Normalize query terms and tokens from argumentative units
    using the WordNet lemmatizer.
    """

    # noinspection PyMethodMayBeStatic
    def __post_init__(self):
        download_nltk_dependencies("wordnet", "omw-1.4")

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        arguments1 = self.analyze_text(context, document1)
        arguments2 = self.analyze_text(context, document2)

        count1 = sum(
            _count_query_terms(context, sentences, query, self.normalize)
            for _, sentences in arguments1.items()
        )
        count2 = sum(
            _count_query_terms(context, sentences, query, self.normalize)
            for _, sentences in arguments2.items()
        )

        if count1 > count2:
            return 1
        elif count1 < count2:
            return -1
        else:
            return 0


@dataclass(frozen=True)
class QTArg(LEN_Mixin, QueryTermOccurrenceInArgumentativeUnitsAxiom):
    name = "QTArg"


@dataclass(frozen=True)
class QueryTermPositionInArgumentativeUnitsAxiom(_TargerMixin, Axiom):
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

    normalize: bool = True
    """
    Normalize query terms and tokens from argumentative units
    using the WordNet lemmatizer.
    """
    penalty: Optional[int] = 10000000
    """
    Penalty for the average query term position,
    if a query term is not found in any argumentative unit for a document.
    Set to None to use the maximum length of the two compared documents.
    """

    # noinspection PyMethodMayBeStatic
    def __post_init__(self):
        download_nltk_dependencies("wordnet", "omw-1.4")

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        arguments1 = self.analyze_text(context, document1)
        arguments2 = self.analyze_text(context, document2)

        if len(arguments1) == 0 or len(arguments2) == 0:
            return 0

        penalty = self.penalty
        if penalty is None:
            penalty = max(
                len(context.terms(document1)),
                len(context.terms(document2)),
            ) + 1

        position1 = mean(
            _query_term_position_in_argument(
                context,
                sentences,
                query,
                penalty,
                self.normalize
            )
            for _, sentences in arguments1.items()
        )
        position2 = mean(
            _query_term_position_in_argument(
                context,
                sentences,
                query,
                penalty,
                self.normalize
            )
            for _, sentences in arguments2.items()
        )

        if position1 < position2:
            return 1
        elif position1 > position2:
            return -1
        else:
            return 0


@dataclass(frozen=True)
class QTPArg(LEN_Mixin, QueryTermPositionInArgumentativeUnitsAxiom):
    name = "QTPArg"


@dataclass(frozen=True)
class AverageSentenceLengthAxiom(Axiom):
    """
    Favor documents with an average sentence length between
    a minimum (default: 12) and a maximum (default: 20) number of words.

    This axiom is based on the general observation
    for text readability / good writing style [8, 10].

    References:
        Markel, M.: Technical Communication. 9th ed. Bedford/St Martin’s (2010)
        Newell, C.: Editing Tip: Sentence Length (2014)
    """

    min_sentence_length: int = 12
    max_sentence_length: int = 20

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        sentence_length1 = _sentence_length(context, document1)
        sentence_length2 = _sentence_length(context, document2)

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


@dataclass(frozen=True)
class aSL(LEN_Mixin, AverageSentenceLengthAxiom):
    name = "aSL"
