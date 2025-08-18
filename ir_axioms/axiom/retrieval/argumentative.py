from dataclasses import dataclass, field
from functools import lru_cache
from math import nan  # pyright: ignore[reportShadowedImports]
from pathlib import Path
from statistics import mean
from typing import Any, Final, Iterable, Dict, Optional, Sequence, Union

from injector import inject, NoInject
from numpy import array, float_
from targer_api import ArgumentSentences, ArgumentLabel, ArgumentTag, analyze_text
from targer_api.constants import DEFAULT_TARGER_MODELS, DEFAULT_TARGER_API_URL
from tqdm.auto import tqdm

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.precondition import PreconditionMixin
from ir_axioms.precondition.base import Precondition
from ir_axioms.precondition.length import LEN
from ir_axioms.axiom.utils import strictly_greater, strictly_less
from ir_axioms.model import PreferenceMatrix, Query, Document, Preference
from ir_axioms.tools import TextContents, TermTokenizer, SentenceTokenizer
from ir_axioms.utils.lazy import lazy_inject


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
) -> int:
    term_count = 0
    for term in term_tokenizer.terms_unordered(text_contents.contents(query)):
        for sentence in sentences:
            for tag in sentence:
                token = tag.token
                if (
                    term == token
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
) -> float:
    term_argument_position = []
    tags = [tag for sentence in sentences for tag in sentence]
    for term in term_tokenizer.terms_unordered(text_contents.contents(query)):
        found: bool = False
        for i, tag in enumerate(tags):
            position = i + 1
            token = tag.token
            if term == token and tag.label != ArgumentLabel.O and tag.probability > 0.5:
                term_argument_position.append(position)
                found = True
                break
        if not found:
            term_argument_position.append(penalty)
    if len(term_argument_position) == 0:
        return penalty
    return mean(term_argument_position)


def _average_sentence_length(
    text_contents: TextContents[Document],
    term_tokenizer: TermTokenizer,
    sentence_tokenizer: SentenceTokenizer,
    document: Document,
) -> float:
    sentences = sentence_tokenizer.sentences(text_contents.contents(document))
    if len(sentences) == 0:
        return nan
    return mean(len(term_tokenizer.terms_unordered(sentence)) for sentence in sentences)


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
            set(self.models),
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


ArgUC: Final = lazy_inject(ArgumentativeUnitsCountAxiom)


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
    precondition: NoInject[Precondition[Any, Document]] = field(default_factory=LEN)

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
            )
            for _, sentences in arguments1.items()
        )
        count2 = sum(
            _count_query_terms(
                text_contents=self.text_contents,
                term_tokenizer=self.term_tokenizer,
                sentences=sentences,
                query=input,
            )
            for _, sentences in arguments2.items()
        )

        return strictly_greater(count1, count2)


QTArg: Final = lazy_inject(QueryTermOccurrenceInArgumentativeUnitsAxiom)


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
    penalty: Optional[int] = 100000
    """
    Penalty for the average query term position,
    if a query term is not found in any argumentative unit for a document.
    Set to None to use the maximum length of the two compared documents.
    """
    precondition: NoInject[Precondition[Any, Document]] = field(default_factory=LEN)

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
                        self.term_tokenizer.terms_unordered(
                            self.text_contents.contents(output1),
                        ),
                    ),
                    len(
                        self.term_tokenizer.terms_unordered(
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
            )
            for _, sentences in arguments2.items()
        )

        return strictly_less(position1, position2)


QTPArg: Final = lazy_inject(QueryTermPositionInArgumentativeUnitsAxiom)


@inject
@dataclass(frozen=True, kw_only=True)
class AverageSentenceLengthAxiom(
    PreconditionMixin[Any, Document], Axiom[Any, Document]
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
    term_tokenizer: TermTokenizer
    sentence_tokenizer: SentenceTokenizer
    min_sentence_length: int = 12
    max_sentence_length: int = 20
    precondition: NoInject[Precondition[Any, Document]] = field(default_factory=LEN)

    def preference(
        self,
        input: Any,
        output1: Document,
        output2: Document,
    ) -> Preference:
        sentence_length1 = _average_sentence_length(
            text_contents=self.text_contents,
            term_tokenizer=self.term_tokenizer,
            sentence_tokenizer=self.sentence_tokenizer,
            document=output1,
        )
        sentence_length2 = _average_sentence_length(
            text_contents=self.text_contents,
            term_tokenizer=self.term_tokenizer,
            sentence_tokenizer=self.sentence_tokenizer,
            document=output2,
        )

        min_length = self.min_sentence_length
        max_length = self.max_sentence_length

        length_in_range1 = min_length <= sentence_length1 <= max_length
        length_in_range2 = min_length <= sentence_length2 <= max_length

        return strictly_greater(length_in_range1, length_in_range2)

    def preferences(
        self,
        input: Any,
        outputs: Sequence[Document],
    ) -> PreferenceMatrix:
        lengths = (
            _average_sentence_length(
                text_contents=self.text_contents,
                term_tokenizer=self.term_tokenizer,
                sentence_tokenizer=self.sentence_tokenizer,
                document=document,
            )
            for document in outputs
        )
        lengths_in_range = [
            self.min_sentence_length <= length <= self.max_sentence_length
            for length in tqdm(
                lengths,
                total=len(outputs),
                desc="Sentence lengths",
                unit="document",
            )
        ]
        return array(
            [
                strictly_greater(length_in_range1, length_in_range2)
                for length_in_range1 in lengths_in_range
                for length_in_range2 in lengths_in_range
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


aSLDoc: Final = lazy_inject(AverageSentenceLengthAxiom)
aSL: Final = aSLDoc
