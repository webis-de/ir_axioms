from dataclasses import dataclass
from statistics import mean
from typing import List, Set, Dict, Optional

from nltk import WordNetLemmatizer
from targer.api import fetch_arguments
from targer.constants import DEFAULT_TARGER_API_URL, DEFAULT_TARGER_MODELS
from targer.model import (
    TargerArgumentSentences, TargerArgumentLabel, TargerArgumentTag
)

from ir_axioms.axiom import Axiom
from ir_axioms.axiom.utils import approximately_same_length
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext
from ir_axioms.utils.nltk import download_nltk_dependencies


def _normalize(word: str):
    download_nltk_dependencies("wordnet")
    _word_net_lemmatizer = WordNetLemmatizer()
    return _word_net_lemmatizer.lemmatize(word).lower()


def _count_argumentative_units(sentences: TargerArgumentSentences) -> int:
    return _count_claims(sentences) + _count_premises(sentences)


def _count_premises(sentences: TargerArgumentSentences) -> int:
    count: int = 0
    for sentence in sentences:
        for tag in sentence:
            if tag.label == TargerArgumentLabel.P_B and tag.probability > 0.5:
                count += 1
    return count


def _count_claims(sentences: TargerArgumentSentences) -> int:
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


def _is_claim(tag: TargerArgumentTag) -> bool:
    return (
            tag.label == TargerArgumentLabel.C_B or
            tag.label == TargerArgumentLabel.C_I or
            tag.label == TargerArgumentLabel.MC_B or
            tag.label == TargerArgumentLabel.MC_I
    )


def _is_premise(tag: TargerArgumentTag) -> bool:
    return (
            tag.label == TargerArgumentLabel.P_B or
            tag.label == TargerArgumentLabel.P_I or
            tag.label == TargerArgumentLabel.MP_B or
            tag.label == TargerArgumentLabel.MP_I
    )


def _is_claim_or_premise(tag: TargerArgumentTag) -> bool:
    return _is_claim(tag) or _is_premise(tag)


def _count_query_terms(
        context: RerankingContext,
        sentences: TargerArgumentSentences,
        query: Query,
        normalize: bool = True,
):
    term_count = 0
    for term in context.terms(query.title):
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
        context: RerankingContext,
        sentences: TargerArgumentSentences,
        query: Query,
        penalty: int,
        normalize: bool = True,
):
    term_arg_pos: List[int] = []
    tags = [tag for sentence in sentences for tag in sentence]
    for term in context.terms(query.title):
        normalized_term = _normalize(term) if normalize else term
        found: bool = False
        for i, tag in enumerate(tags):
            position = i + 1
            token = tag.token
            normalized_token = _normalize(token) if normalize else token
            if (
                    normalized_term == normalized_token and
                    tag.label != TargerArgumentLabel.O and
                    tag.probability > 0.5
            ):
                term_arg_pos.append(position)
                found = True
                break
        if not found:
            term_arg_pos.append(penalty)
    return mean(term_arg_pos)


@dataclass
class _TargerAxiomMixin:
    models: Set[str] = DEFAULT_TARGER_MODELS
    api_url: str = DEFAULT_TARGER_API_URL

    def fetch_arguments(
            self,
            context: RerankingContext,
            document: RankedDocument,
    ) -> Dict[str, TargerArgumentSentences]:
        return fetch_arguments(
            document.content,
            models=self.models,
            api_url=self.api_url,
            cache_dir=context.cache_dir / "targer"
        )


class ArgumentativeUnitsCountAxiom(Axiom, _TargerAxiomMixin):
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        arguments1 = self.fetch_arguments(context, document1)
        arguments2 = self.fetch_arguments(context, document2)

        if not approximately_same_length(context, document1, document2):
            return 0

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


@dataclass
class QueryTermOccurrenceInArgumentativeUnitsAxiom(Axiom, _TargerAxiomMixin):
    normalize: bool = True
    """
    Normalize query terms and tokens from argumentative units
    using the WordNet lemmatizer.
    """

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        arguments1 = self.fetch_arguments(context, document1)
        arguments2 = self.fetch_arguments(context, document2)

        if not approximately_same_length(context, document1, document2):
            return 0

        count1 = sum(
            _count_query_terms(context, sentences, query)
            for _, sentences in arguments1.items()
        )
        count2 = sum(
            _count_query_terms(context, sentences, query)
            for _, sentences in arguments2.items()
        )

        if count1 > count2:
            return 1
        elif count1 < count2:
            return -1
        else:
            return 0


@dataclass
class QueryTermPositionInArgumentativeUnitsAxiom(Axiom, _TargerAxiomMixin):
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

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        arguments1 = self.fetch_arguments(context, document1)
        arguments2 = self.fetch_arguments(context, document2)

        if not approximately_same_length(context, document1, document2):
            return 0

        penalty = self.penalty
        if penalty is None:
            penalty = max(
                len(context.terms(document1.content)),
                len(context.terms(document2.content)),
            ) + 1

        position1 = mean(list(
            _query_term_position_in_argument(
                context,
                sentences,
                query,
                penalty
            )
            for _, sentences in arguments1.items()
        ))
        position2 = mean(list(
            _query_term_position_in_argument(
                context,
                sentences,
                query,
                penalty
            )
            for _, sentences in arguments2.items()
        ))

        if position1 < position2:
            return 1
        elif position1 > position2:
            return -1
        else:
            return 0


# Aliases for shorter names:
ArgUC = ArgumentativeUnitsCountAxiom
QTArg = QueryTermOccurrenceInArgumentativeUnitsAxiom
QTPArg = QueryTermPositionInArgumentativeUnitsAxiom
