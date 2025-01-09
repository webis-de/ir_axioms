# Clarity: generated text is comprehensible and clearly communicates the key information/sources
# Language clarity:
# - [x] concise
# - [ ] fluent (-> compare "size"/"depth" of dependency parse tree)
# - [ ] comprehensible (-> check language; n-gram statistics with Netspeak)
# - [x] lexically correct
# - [x] grammatically correct
# - [ ] user-accessible (-> detect garbled text)
# - [ ] avoid topic switching
# Content clarity:
# - [x] communicate most salient information (-> avoid redundancy)
# - [x] avoid jargon (-> check vocabulary commonness)
# - [ ] attribute sources (-> count citations/links)

from dataclasses import dataclass
from functools import cached_property
from math import isclose
from typing import Final, Sequence, Any, Mapping, Union

from injector import inject, NoInject
from language_tool_python import LanguageTool
from numpy import array, float_
from tqdm.auto import tqdm
from scipy.special import rel_entr
from spacy import load as spacy_load
from spacy.language import Language
from spellchecker import SpellChecker
from wordfreq import word_frequency

from axioms.axiom.base import Axiom
from axioms.axiom.utils import strictly_less
from axioms.dependency_injection import injector
from axioms.model import PreferenceMatrix
from axioms.model.base import Preference
from axioms.model.generation import GenerationOutput, GenerationInput
from axioms.tools import (
    TextContents,
    SentenceSimilarity,
    SentenceTokenizer,
    TextStatistics,
)
from axioms.utils.lazy import lazy_inject


@inject
@dataclass(frozen=True, kw_only=True)
class GrammarErrorsClarityAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text with fewer grammar errors.
    """

    text_contents: TextContents[GenerationOutput]

    language: NoInject[str] = "en-US"
    margin_fraction: float = 0.0

    @cached_property
    def _language_tool(self) -> LanguageTool:
        # TODO: Make the grammar checker a configurable dependency.
        return LanguageTool(language=self.language)

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        matches1 = self._language_tool.check(self.text_contents.contents(output1))
        matches2 = self._language_tool.check(self.text_contents.contents(output2))
        num_matches1 = len(matches1)
        num_matches2 = len(matches2)
        if isclose(
            num_matches1,
            num_matches2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_less(num_matches1, num_matches2)

    def preferences(
        self,
        input: Any,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        matches = (
            self._language_tool.check(self.text_contents.contents(output))
            for output in tqdm(
                outputs,
                desc="Checking grammar",
                unit="output",
            )
        )
        num_matches = [len(matches) for matches in matches]
        return array(
            [
                (
                    strictly_less(num_matches1, num_matches2)
                    if not isclose(
                        num_matches1,
                        num_matches2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for num_matches1 in num_matches
                for num_matches2 in num_matches
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


CLAR1: Final = lazy_inject(GrammarErrorsClarityAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class GrammarErrorTypesClarityAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text with fewer types of grammar errors.
    """

    text_contents: TextContents[GenerationOutput]

    language: NoInject[str] = "en-US"
    margin_fraction: float = 0.0

    @cached_property
    def _language_tool(self) -> LanguageTool:
        return LanguageTool(language=self.language)

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        matches1 = self._language_tool.check(self.text_contents.contents(output1))
        matches2 = self._language_tool.check(self.text_contents.contents(output2))
        match_types1 = set(match.ruleId for match in matches1)
        match_types2 = set(match.ruleId for match in matches2)
        num_matches1 = len(match_types1)
        num_matches2 = len(match_types2)
        if isclose(
            num_matches1,
            num_matches2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_less(num_matches1, num_matches2)

    def preferences(
        self,
        input: Any,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        matches = (
            self._language_tool.check(self.text_contents.contents(output))
            for output in tqdm(
                outputs,
                desc="Checking grammar",
                unit="output",
            )
        )
        match_types = (set(match.ruleId for match in matches) for matches in matches)
        num_matches = [len(match_types) for match_types in match_types]
        return array(
            [
                (
                    strictly_less(num_matches1, num_matches2)
                    if not isclose(
                        num_matches1,
                        num_matches2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for num_matches1 in num_matches
                for num_matches2 in num_matches
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


CLAR2: Final = lazy_inject(GrammarErrorTypesClarityAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class GrammarErrorProportionClarityAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text with a lower proportion of charecters covered by grammar errors.
    """

    text_contents: TextContents[GenerationOutput]

    language: NoInject[str] = "en-US"
    margin_fraction: float = 0.1

    @cached_property
    def _language_tool(self) -> LanguageTool:
        return LanguageTool(language=self.language)

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        contents1 = self.text_contents.contents(output1)
        contents2 = self.text_contents.contents(output2)
        matches1 = self._language_tool.check(contents1)
        matches2 = self._language_tool.check(contents2)
        characters_matched1 = {
            i
            for match in matches1
            for i in range(match.offset, match.offset + match.errorLength)
        }
        characters_matched2 = {
            i
            for match in matches2
            for i in range(match.offset, match.offset + match.errorLength)
        }
        num_characters_matched1 = len(characters_matched1)
        num_characters_matched2 = len(characters_matched2)
        error_coverage1 = (
            num_characters_matched1 / len(contents1) if len(contents1) > 0 else 0
        )
        error_coverage2 = (
            num_characters_matched2 / len(contents2) if len(contents2) > 0 else 0
        )
        if isclose(
            error_coverage1,
            error_coverage2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_less(error_coverage1, error_coverage2)

    def preferences(
        self,
        input: Any,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        contents = [self.text_contents.contents(output) for output in outputs]
        matches = (
            self._language_tool.check(content)
            for content in tqdm(
                contents,
                desc="Checking grammar",
                unit="output",
            )
        )
        characters_matched = (
            {
                i
                for match in matches
                for i in range(match.offset, match.offset + match.errorLength)
            }
            for matches in matches
        )
        num_characters_matched = (
            len(characters_matched) for characters_matched in characters_matched
        )
        error_coverages = [
            num_characters_matched / len(content) if len(content) > 0 else 0
            for num_characters_matched, content in zip(num_characters_matched, contents)
        ]
        return array(
            [
                (
                    strictly_less(error_coverage1, error_coverage2)
                    if not isclose(
                        error_coverage1,
                        error_coverage2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for error_coverage1 in error_coverages
                for error_coverage2 in error_coverages
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


CLAR3: Final = lazy_inject(GrammarErrorProportionClarityAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class SentenceRedundancyClarityAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text with less redundant sentences as measured by average similarity of all sentences.
    """

    text_contents: TextContents[GenerationOutput]
    sentence_tokenizer: SentenceTokenizer
    sentence_similarity: SentenceSimilarity

    margin_fraction: float = 0.1

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        similarities1 = self.sentence_similarity.similarities(
            self.sentence_tokenizer.sentences(self.text_contents.contents(output1))
        )
        similarities2 = self.sentence_similarity.similarities(
            self.sentence_tokenizer.sentences(self.text_contents.contents(output2))
        )
        # TODO: Make aggregation configurable.
        aggregate_similarity1 = similarities1.mean()
        aggregate_similarity2 = similarities2.mean()
        if isclose(
            aggregate_similarity1,
            aggregate_similarity2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_less(aggregate_similarity1, aggregate_similarity2)

    def preferences(
        self,
        input: Any,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        similarities = (
            self.sentence_similarity.similarities(
                self.sentence_tokenizer.sentences(self.text_contents.contents(output))
            )
            for output in tqdm(
                outputs,
                desc="Computing sentence similarities",
                unit="text",
            )
        )
        aggregate_similarities = [similarities.mean() for similarities in similarities]
        return array(
            [
                (
                    strictly_less(aggregate_similarity1, aggregate_similarity2)
                    if not isclose(
                        aggregate_similarity1,
                        aggregate_similarity2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for aggregate_similarity1 in aggregate_similarities
                for aggregate_similarity2 in aggregate_similarities
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


CLAR4: Final = lazy_inject(SentenceRedundancyClarityAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class MisspellingsClarityAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer outputs that contain fewer misspellings.
    """

    text_contents: TextContents[GenerationOutput]

    language_name: NoInject[str] = "en_core_web_sm"
    margin_fraction: float = 0.1

    # TODO: Migrate spell checker to tool for dependency injection.
    @cached_property
    def _language(self) -> Language:
        return spacy_load(name=self.language_name)

    @cached_property
    def _spell_checker(self) -> SpellChecker:
        return SpellChecker(
            language=self._language.lang,
        )

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        return self.preferences(input, [output1, output2])[0, 1]

    def preferences(
        self,
        input: Any,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        words = (
            {
                token.text
                for token in self._language(self.text_contents.contents(output))
            }
            for output in outputs
        )
        misspellings = [
            (
                len(self._spell_checker.unknown(words)) / len(words)
                if len(words) > 0
                else 0
            )
            for words in words
        ]
        return array(
            [
                (
                    0
                    if isclose(
                        misspellings1,
                        misspellings2,
                        rel_tol=self.margin_fraction,
                    )
                    else strictly_less(misspellings1, misspellings2)
                )
                for misspellings1 in misspellings
                for misspellings2 in misspellings
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


CLAR5: Final = lazy_inject(MisspellingsClarityAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class WordCommonnessClarityAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text with more common vocabulary according to some reference corpus, e.g., the Wikipedia.
    """

    text_statistics: TextStatistics[GenerationOutput]

    language: NoInject[str] = "en"
    margin_fraction: float = 0.1

    def _commonness(self, term_frequencies: Mapping[str, float]) -> float:
        unique_terms = sorted(term_frequencies.keys())
        expected_frequencies = array(
            [
                word_frequency(
                    word=term,
                    lang=self.language,
                    wordlist="small",
                )
                for term in unique_terms
            ]
        )
        expected_frequencies /= expected_frequencies.sum()
        observed_frequencies = array([term_frequencies[term] for term in unique_terms])
        return rel_entr(observed_frequencies, expected_frequencies).sum()

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        commonnness1 = self._commonness(self.text_statistics.term_frequencies(output1))
        commonnness2 = self._commonness(self.text_statistics.term_frequencies(output2))
        if isclose(
            commonnness1,
            commonnness2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_less(commonnness1, commonnness2)

    def preferences(
        self,
        input: Any,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        commonnesses = [
            self._commonness(self.text_statistics.term_frequencies(output))
            for output in tqdm(
                outputs,
                desc="Computing word commonnesses",
                unit="text",
            )
        ]
        return array(
            [
                (
                    strictly_less(commonness1, commonness2)
                    if not isclose(
                        commonness1,
                        commonness2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for commonness1 in commonnesses
                for commonness2 in commonnesses
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


CLAR6: Final = lazy_inject(WordCommonnessClarityAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class NormalizedWordCommonnessClarityAxiom(Axiom[GenerationInput, GenerationOutput]):
    """
    Prefer text with more common vocabulary according to some reference corpus but cancel out expected divergence from the query's word frequencies.
    """

    text_statistics: TextStatistics[Union[GenerationInput, GenerationOutput]]

    language: NoInject[str] = "en"
    margin_fraction: float = 0.0

    def _commonness(
        self,
        output_term_frequencies: Mapping[str, float],
        input_term_frequencies: Mapping[str, float],
    ) -> float:
        unique_terms = sorted(output_term_frequencies.keys())
        expected_frequencies = array(
            [
                word_frequency(
                    word=term,
                    lang=self.language,
                    wordlist="small",
                )
                for term in unique_terms
            ]
        )
        expected_frequencies /= expected_frequencies.sum()

        observed_output_frequencies = array(
            [output_term_frequencies[term] for term in unique_terms]
        )
        observed_input_frequencies = array(
            [input_term_frequencies.get(term, 0) for term in unique_terms]
        )
        observed_input_frequencies /= observed_input_frequencies.sum()

        output_divergence = rel_entr(observed_output_frequencies, expected_frequencies)
        input_divergence = rel_entr(observed_input_frequencies, expected_frequencies)

        normalized_output_divergence = output_divergence / input_divergence
        return normalized_output_divergence.sum()

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        input_term_frequencies = self.text_statistics.term_frequencies(input)
        commonnness1 = self._commonness(
            self.text_statistics.term_frequencies(output1),
            input_term_frequencies,
        )
        commonnness2 = self._commonness(
            self.text_statistics.term_frequencies(output2),
            input_term_frequencies,
        )
        if isclose(
            commonnness1,
            commonnness2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_less(commonnness1, commonnness2)

    def preferences(
        self,
        input: GenerationInput,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        input_term_frequencies = self.text_statistics.term_frequencies(input)
        commonnesses = [
            self._commonness(
                self.text_statistics.term_frequencies(output),
                input_term_frequencies,
            )
            for output in tqdm(
                outputs,
                desc="Computing word commonnesses",
                unit="text",
            )
        ]
        return array(
            [
                (
                    strictly_less(commonness1, commonness2)
                    if not isclose(
                        commonness1,
                        commonness2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for commonness1 in commonnesses
                for commonness2 in commonnesses
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


CLAR7: Final = lazy_inject(NormalizedWordCommonnessClarityAxiom, injector)
