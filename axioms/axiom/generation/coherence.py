# Coherence: statement arrangement to form a narrative without contradictions
# Logical coherence:
# - [ ] well-structuredness (e.g., answer first, explanation later; implementation: maybe match "because" or dependency tree)
# Stylistic coherence:
# - [x] uniform style of speech

from dataclasses import dataclass
from itertools import chain
from functools import cached_property
from math import isclose, nan
from typing import Final, Sequence, Any, TYPE_CHECKING

from fastcoref import spacy_component
from injector import inject, NoInject
from more_itertools import pairwise
from numpy import array, float_
from spacy import load as spacy_load
from spacy.language import Language
from tqdm.auto import tqdm

from axioms.axiom.base import Axiom
from axioms.axiom.utils import strictly_less
from axioms.dependency_injection import injector
from axioms.model.base import Preference, PreferenceMatrix
from axioms.model.generation import GenerationOutput
from axioms.tools import (
    TextContents,
    TermTokenizer,
    SentenceTokenizer,
)
from axioms.utils.lazy import lazy_inject


@inject
@dataclass(frozen=True, kw_only=True)
class WordLengthDeviationCoherenceAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text with lower standard deviation of average word lengths across sentences.
    """

    text_contents: TextContents[GenerationOutput]
    sentence_tokenizer: SentenceTokenizer
    term_tokenizer: TermTokenizer

    margin_fraction: NoInject[float] = 0.0

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        average_word_lengths1 = array(
            [
                array(
                    [len(word) for word in self.term_tokenizer.terms(sentence)]
                ).mean()
                for sentence in self.sentence_tokenizer.sentences(
                    self.text_contents.contents(output1)
                )
            ]
        )
        average_word_lengths2 = array(
            [
                array(
                    [len(word) for word in self.term_tokenizer.terms(sentence)]
                ).mean()
                for sentence in self.sentence_tokenizer.sentences(
                    self.text_contents.contents(output2)
                )
            ]
        )

        average_word_lengths_stdev1 = average_word_lengths1.std()
        average_word_lengths_stdev2 = average_word_lengths2.std()

        if isclose(
            average_word_lengths_stdev1,
            average_word_lengths_stdev2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_less(average_word_lengths_stdev1, average_word_lengths_stdev2)

    def preferences(
        self,
        input: Any,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        average_word_lengths = (
            array(
                [
                    array(
                        [len(word) for word in self.term_tokenizer.terms(sentence)]
                    ).mean()
                    for sentence in self.sentence_tokenizer.sentences(
                        self.text_contents.contents(output)
                    )
                ]
            )
            for output in tqdm(
                outputs,
                desc="Calculate average word lengths",
                unit="output",
            )
        )

        average_word_lengths_stdevs = [
            average_word_lengths.std() for average_word_lengths in average_word_lengths
        ]

        return array(
            [
                (
                    strictly_less(
                        average_word_lengths_stdev1, average_word_lengths_stdev2
                    )
                    if not isclose(
                        average_word_lengths_stdev1,
                        average_word_lengths_stdev2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for average_word_lengths_stdev1 in average_word_lengths_stdevs
                for average_word_lengths_stdev2 in average_word_lengths_stdevs
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


COH1: Final = lazy_inject(WordLengthDeviationCoherenceAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class SentenceLengthDeviationCoherenceAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text with lower standard deviation of sentence lengths.
    """

    text_contents: TextContents[GenerationOutput]
    sentence_tokenizer: SentenceTokenizer
    term_tokenizer: TermTokenizer

    margin_fraction: NoInject[float] = 0.0

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        sentence_lengths1 = array(
            [
                sum(1 for _ in self.term_tokenizer.terms(sentence))
                for sentence in self.sentence_tokenizer.sentences(
                    self.text_contents.contents(output1)
                )
            ]
        )
        sentence_lengths2 = array(
            [
                sum(1 for _ in self.term_tokenizer.terms(sentence))
                for sentence in self.sentence_tokenizer.sentences(
                    self.text_contents.contents(output2)
                )
            ]
        )

        sentence_lengths_stdev1 = sentence_lengths1.std()
        sentence_lengths_stdev2 = sentence_lengths2.std()

        if isclose(
            sentence_lengths_stdev1,
            sentence_lengths_stdev2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_less(sentence_lengths_stdev1, sentence_lengths_stdev2)

    def preferences(
        self,
        input: Any,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        sentence_lengths = (
            array(
                [
                    sum(1 for _ in self.term_tokenizer.terms(sentence))
                    for sentence in self.sentence_tokenizer.sentences(
                        self.text_contents.contents(output)
                    )
                ]
            )
            for output in tqdm(
                outputs,
                desc="Calculate sentence lengths",
                unit="output",
            )
        )

        sentence_lengths_stdevs = [
            sentence_lengths.std() for sentence_lengths in sentence_lengths
        ]

        return array(
            [
                (
                    strictly_less(sentence_lengths_stdev1, sentence_lengths_stdev2)
                    if not isclose(
                        sentence_lengths_stdev1,
                        sentence_lengths_stdev2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for sentence_lengths_stdev1 in sentence_lengths_stdevs
                for sentence_lengths_stdev2 in sentence_lengths_stdevs
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


COH2: Final = lazy_inject(SentenceLengthDeviationCoherenceAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class TenseSwitchingCoherenceAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text with fewer switches of tenses.
    """

    text_contents: TextContents[GenerationOutput]

    language_name: NoInject[str] = "en_core_web_sm"
    margin_fraction: NoInject[float] = 0.0

    # TODO: Migrate to tool injected by DI.
    @cached_property
    def _language(self) -> Language:
        return spacy_load(
            name=self.language_name,
            disable=["parser"],
        )

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        tenses1 = list(
            chain.from_iterable(
                token.morph.get("Tense", [])
                for token in self._language(self.text_contents.contents(output1))
                if token.pos_ == "VERB" and token.morph.get("Tense", []) != []
            )
        )
        tenses2 = list(
            chain.from_iterable(
                token.morph.get("Tense", [])
                for token in self._language(self.text_contents.contents(output2))
                if token.pos_ == "VERB" and token.morph.get("Tense", []) != []
            )
        )

        num_tense_switches1 = sum(
            1 for tense, next_tense in pairwise(tenses1) if tense != next_tense
        )
        num_tense_switches2 = sum(
            1 for tense, next_tense in pairwise(tenses2) if tense != next_tense
        )

        if isclose(
            num_tense_switches1,
            num_tense_switches2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_less(num_tense_switches1, num_tense_switches2)

    def preferences(
        self,
        input: Any,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        tenses = (
            list(
                chain.from_iterable(
                    token.morph.get("Tense", [])
                    for token in self._language(self.text_contents.contents(output))
                    if token.pos_ == "VERB" and token.morph.get("Tense", []) != []
                )
            )
            for output in tqdm(
                outputs,
                desc="Determine verb tenses",
                unit="output",
            )
        )

        num_tense_switches = [
            sum(1 for tense, next_tense in pairwise(tense) if tense != next_tense)
            for tense in tenses
        ]

        return array(
            [
                (
                    strictly_less(num_tense_switches1, num_tense_switches2)
                    if not isclose(
                        num_tense_switches1,
                        num_tense_switches2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for num_tense_switches1 in num_tense_switches
                for num_tense_switches2 in num_tense_switches
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


COH3: Final = lazy_inject(TenseSwitchingCoherenceAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class CoreferenceClusterClosenessCoherenceAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text with coreference clusters that are closer together.
    """

    text_contents: TextContents[GenerationOutput]

    language_name: NoInject[str] = "en_core_web_sm"
    margin_fraction: float = 0.1

    # TODO: Migrate to tool for dependency injection.
    @cached_property
    def _language(self) -> Language:
        if TYPE_CHECKING:
            spacy_component  # To ignore the unused import warning.
        language = spacy_load(
            name=self.language_name,
            exclude=["parser", "lemmatizer", "ner", "textcat"],
        )
        language.add_pipe("fastcoref")
        return language

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        coreference_clusters1: Sequence[Sequence[tuple[int, int]]] = self._language(
            self.text_contents.contents(output1)
        )._.coref_clusters
        coreference_clusters2: Sequence[Sequence[tuple[int, int]]] = self._language(
            self.text_contents.contents(output2)
        )._.coref_clusters

        coreference_distances1 = array(
            [
                (
                    cluster[-1][1] - cluster[0][0]
                    if len(coreference_clusters1) > 0
                    else nan
                )
                for cluster in coreference_clusters1
            ]
        )
        coreference_distances2 = array(
            [
                (
                    cluster[-1][1] - cluster[0][0]
                    if len(coreference_clusters2) > 0
                    else nan
                )
                for cluster in coreference_clusters2
            ]
        )

        avg_coreference_distance1 = coreference_distances1.mean()
        avg_coreference_distance2 = coreference_distances2.mean()

        if isclose(
            avg_coreference_distance1,
            avg_coreference_distance2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_less(avg_coreference_distance1, avg_coreference_distance2)

    def preferences(
        self,
        input: Any,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        docs = self._language.pipe(
            self.text_contents.contents(output)
            for output in tqdm(
                outputs,
                desc="Analyze coreference clusters",
                unit="output",
            )
        )

        coreference_clusters = (doc._.coref_clusters for doc in docs)

        coreference_distances = [
            array(
                [
                    (
                        cluster[-1][1] - cluster[0][0]
                        if len(coreference_clusters) > 0
                        else nan
                    )
                ]
                for cluster in coreference_clusters
            )
            for coreference_clusters in coreference_clusters
        ]

        avg_coreference_distances = [
            coreference_distances.mean()
            for coreference_distances in coreference_distances
        ]

        return array(
            [
                (
                    strictly_less(avg_coreference_distance1, avg_coreference_distance2)
                    if not isclose(
                        avg_coreference_distance1,
                        avg_coreference_distance2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for avg_coreference_distance1 in avg_coreference_distances
                for avg_coreference_distance2 in avg_coreference_distances
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


COH4: Final = lazy_inject(CoreferenceClusterClosenessCoherenceAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class ShortAnswerCoherenceAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text that gives the answer earlier (i.e., the first phrase before the first punctuation or conjunction).
    """

    text_contents: TextContents[GenerationOutput]

    language_name: NoInject[str] = "en_core_web_sm"
    margin_fraction: float = 0.1

    # TODO: Migrate to tool for dependency injection.
    @cached_property
    def _language(self) -> Language:
        return spacy_load(
            name=self.language_name,
        )

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        doc1 = self._language(self.text_contents.contents(output1))
        doc2 = self._language(self.text_contents.contents(output2))

        first_punct1 = next(
            (
                token.i
                for token in doc1
                if token.is_punct or token.pos_ in ("CCONJ", "SCONJ")
            ),
            len(doc1),
        )
        first_punct2 = next(
            (
                token.i
                for token in doc2
                if token.is_punct or token.text in {"and", "or"}
            ),
            len(doc2),
        )

        if isclose(
            first_punct1,
            first_punct2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_less(first_punct1, first_punct2)

    def preferences(
        self,
        input: Any,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        docs = self._language.pipe(
            self.text_contents.contents(output)
            for output in tqdm(
                outputs,
                desc="Analyze text",
                unit="output",
            )
        )

        first_puncts = [
            next(
                (
                    token.i
                    for token in doc
                    if token.is_punct or token.pos_ in ("CCONJ", "SCONJ")
                ),
                len(doc),
            )
            for doc in docs
        ]

        return array(
            [
                (
                    strictly_less(first_punct1, first_punct2)
                    if not isclose(
                        first_punct1,
                        first_punct2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for first_punct1 in first_puncts
                for first_punct2 in first_puncts
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


COH5: Final = lazy_inject(ShortAnswerCoherenceAxiom, injector)
