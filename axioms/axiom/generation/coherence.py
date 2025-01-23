"""
Coherence axioms for retrieval augmented text.

For a definition of this utility dimension, see: https://doi.org/10.1145/3626772.3657849

- Logical coherence: Is the response well-structured?
- Stylistic coherence: Does the response have a uniform style of speech?
"""

from dataclasses import dataclass
from functools import cached_property
from math import isclose
from typing import Final, Sequence, Any

from injector import inject, NoInject
from numpy import array, float_
from spacy import load as spacy_load
from spacy.language import Language
from tqdm.auto import tqdm
from textacy.extract.triples import subject_verb_object_triples

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
class SubjectVerbClosenessCoherenceAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text with subjects and verbs that are closer together.
    """

    text_contents: TextContents[GenerationOutput]

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
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        doc1 = self._language(self.text_contents.contents(output1))
        doc2 = self._language(self.text_contents.contents(output2))

        svo_triples1 = subject_verb_object_triples(doc1)
        svo_triples2 = subject_verb_object_triples(doc2)

        max_sv_distances1 = array(
            [
                max(
                    abs(verb.i - subject.i)
                    for subject in subject_toks
                    for verb in verb_toks
                )
                for subject_toks, verb_toks, _ in svo_triples1
            ]
        )
        max_sv_distances2 = array(
            [
                max(
                    abs(verb.i - subject.i)
                    for subject in subject_toks
                    for verb in verb_toks
                )
                for subject_toks, verb_toks, _ in svo_triples2
            ]
        )

        avg_max_sv_distance1 = max_sv_distances1.mean()
        avg_max_sv_distance2 = max_sv_distances2.mean()

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
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        docs = self._language.pipe(
            self.text_contents.contents(output) for output in outputs
        )

        svo_triples = (subject_verb_object_triples(doc) for doc in docs)

        max_sv_distances = (
            array(
                [
                    max(
                        abs(verb.i - subject.i)
                        for subject in subject_toks
                        for verb in verb_toks
                    )
                    for subject_toks, verb_toks, _ in svo_triples
                ]
            )
            for svo_triples in svo_triples
        )

        avg_max_sv_distances = [
            max_sv_distances.mean()
            for max_sv_distances in tqdm(
                max_sv_distances,
                total=len(outputs),
                desc="Calculate S-V distances",
                unit="output",
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


COH6: Final = lazy_inject(SubjectVerbClosenessCoherenceAxiom, injector)


# TODO: Topic position and stess position by looking up if noun chunks appear in the beginning or end of the sentence.
