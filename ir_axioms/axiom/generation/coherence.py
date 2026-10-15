"""
Coherence axioms for retrieval augmented text.

For a definition of this utility dimension, see: https://doi.org/10.1145/3626772.3657849

- Logical coherence: Is the response well-structured?
- Stylistic coherence: Does the response have a uniform style of speech?
"""

from dataclasses import dataclass
from functools import cached_property
from math import isclose, nan
from statistics import mean
from typing import Final, Sequence, Any

from injector import inject, NoInject
from numpy import array, float_
from spacy import load as spacy_load
from spacy.language import Language
from spacy.tokens import Doc
from tqdm.auto import tqdm
from textacy.extract.triples import subject_verb_object_triples

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import strictly_less
from ir_axioms.model.base import Preference, PreferenceMatrix
from ir_axioms.model.generation import GenerationResponse
from ir_axioms.tools import (
    TextContents,
    TermTokenizer,
    SentenceTokenizer,
)
from ir_axioms.utils.lazy import lazy_inject


@inject
@dataclass(frozen=True, kw_only=True)
class WordLengthDeviationCoherenceAxiom(Axiom[Any, GenerationResponse]):
    """
    Prefer text with lower standard deviation of average word lengths across sentences.
    """

    text_contents: TextContents[GenerationResponse]
    sentence_tokenizer: SentenceTokenizer
    term_tokenizer: TermTokenizer

    margin_fraction: NoInject[float] = 0.0

    def preference(
        self,
        input: Any,
        output1: GenerationResponse,
        output2: GenerationResponse,
    ) -> Preference:
        average_word_lengths1 = array(
            [
                array(
                    [
                        len(word)
                        for word in self.term_tokenizer.terms_unordered(sentence)
                    ]
                ).mean()
                for sentence in self.sentence_tokenizer.sentences(
                    self.text_contents.contents(output1)
                )
            ]
        )
        average_word_lengths2 = array(
            [
                array(
                    [
                        len(word)
                        for word in self.term_tokenizer.terms_unordered(sentence)
                    ]
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
        outputs: Sequence[GenerationResponse],
    ) -> PreferenceMatrix:
        average_word_lengths = (
            array(
                [
                    array(
                        [
                            len(word)
                            for word in self.term_tokenizer.terms_unordered(sentence)
                        ]
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
                disable=not self.verbose,
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


COH1: Final = lazy_inject(WordLengthDeviationCoherenceAxiom)


def _average_max_subject_verb_distance(document: Doc) -> float:
    max_subject_verb_distances = [
        max(abs(verb.i - subject.i) for subject in subject_toks for verb in verb_toks)
        for subject_toks, verb_toks, _ in subject_verb_object_triples(document)
    ]
    if len(max_subject_verb_distances) == 0:
        # Undefined without subject-verb-object triples.
        return nan
    return mean(max_subject_verb_distances)


@inject
@dataclass(frozen=True, kw_only=True)
class SubjectVerbClosenessCoherenceAxiom(Axiom[Any, GenerationResponse]):
    """
    Prefer text with subjects and verbs that are closer together.
    """

    text_contents: TextContents[GenerationResponse]

    language_name: NoInject[str] = "en_core_web_sm"
    margin_fraction: NoInject[float] = 0.1

    @cached_property
    def _language(self) -> Language:
        return spacy_load(
            name=self.language_name,
        )

    def preference(
        self,
        input: Any,
        output1: GenerationResponse,
        output2: GenerationResponse,
    ) -> Preference:
        avg_max_sv_distance1 = _average_max_subject_verb_distance(
            self._language(self.text_contents.contents(output1))
        )
        avg_max_sv_distance2 = _average_max_subject_verb_distance(
            self._language(self.text_contents.contents(output2))
        )

        if isclose(
            avg_max_sv_distance1,
            avg_max_sv_distance2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_less(avg_max_sv_distance1, avg_max_sv_distance2)

    def preferences(
        self,
        input: Any,
        outputs: Sequence[GenerationResponse],
    ) -> PreferenceMatrix:
        docs = self._language.pipe(
            self.text_contents.contents(output) for output in outputs
        )

        avg_max_sv_distances = [
            _average_max_subject_verb_distance(doc)
            for doc in tqdm(
                docs,
                total=len(outputs),
                desc="Calculate S-V distances",
                unit="output",
                disable=not self.verbose,
            )
        ]

        return array(
            [
                (
                    strictly_less(avg_max_sv_distance1, avg_max_sv_distance2)
                    if not isclose(
                        avg_max_sv_distance1,
                        avg_max_sv_distance2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for avg_max_sv_distance1 in avg_max_sv_distances
                for avg_max_sv_distance2 in avg_max_sv_distances
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


COH2: Final = lazy_inject(SubjectVerbClosenessCoherenceAxiom)


# TODO: Topic position and stess position by looking up if noun chunks appear in the beginning or end of the sentence.
