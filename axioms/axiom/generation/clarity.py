# Clarity: generated text is comprehensible and clearly communicates the key information/sources

from dataclasses import dataclass
from functools import cached_property
from math import isclose, nan
from typing import Final, Sequence, Any, Literal, Iterable
from typing_extensions import TypeAlias  # type: ignore

from injector import inject, NoInject
from language_tool_python import LanguageTool
from numpy import array, float_
from tqdm.auto import tqdm
from spacy import load as spacy_load
from spacy.language import Language
from textacy.text_stats import flesch_reading_ease

from axioms.axiom.base import Axiom
from axioms.axiom.utils import strictly_less
from axioms.dependency_injection import injector
from axioms.model import PreferenceMatrix
from axioms.model.base import Preference
from axioms.model.generation import GenerationOutput
from axioms.tools import TextContents
from axioms.utils.lazy import lazy_inject


_LanguageToolCategory: TypeAlias = Literal[
    "CASING",
    "COLLOQUIALISMS",
    "COMPOUNDING",
    "CONFUSED_WORDS",
    "FALSE_FRIENDS",
    "GENDER_NEUTRALITY",
    "GRAMMAR",
    "MISC",
    "PLAIN_ENGLISH",
    "PUNCTUATION",
    "REDUNDANCY",
    "REGIONALISMS",
    "REPETITIONS",
    "REPETITIONS_STYLE",
    "SEMANTICS",
    "STYLE",
    "TYPOGRAPHY",
    "TYPOS",
    "WIKIPEDIA",
]


@inject
@dataclass(frozen=True, kw_only=True)
class _LanguageToolErrorCountsClarityAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text with fewer errors found by language tool.
    """

    text_contents: TextContents[GenerationOutput]

    language: NoInject[str] = "en-US"
    enabled_categories: NoInject[Iterable[_LanguageToolCategory]] = frozenset()
    margin_fraction: NoInject[float] = 0.0

    @cached_property
    def _language_tool(self) -> LanguageTool:
        # TODO: Make the grammar checker a configurable dependency.
        language_tool = LanguageTool(language=self.language)
        language_tool.enabled_categories = set(self.enabled_categories)
        language_tool.enabled_rules_only = True
        return language_tool

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
                desc="Check grammar",
                unit="output",
            )
        )
        num_matches = [len(matches) for matches in matches]
        print(num_matches)
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


@inject
@dataclass(frozen=True, kw_only=True)
class _LanguageToolErrorProportionClarityAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text with a lower proportion of characters covered by errors from LanguageTool.
    """

    text_contents: TextContents[GenerationOutput]

    language: NoInject[str] = "en-US"
    enabled_categories: NoInject[Iterable[_LanguageToolCategory]] = frozenset()
    margin_fraction: NoInject[float] = 0.1

    @cached_property
    def _language_tool(self) -> LanguageTool:
        # TODO: Make the grammar checker a configurable dependency.
        language_tool = LanguageTool(language=self.language)
        language_tool.enabled_categories = set(self.enabled_categories)
        language_tool.enabled_rules_only = True
        return language_tool

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
                desc="Check grammar",
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


@inject
@dataclass(frozen=True, kw_only=True)
class LanguageToolGrammarErrorCountsClarityAxiom(_LanguageToolErrorCountsClarityAxiom):
    """
    Prefer text with fewer grammar errors.
    """

    enabled_categories: Final[NoInject[Iterable[_LanguageToolCategory]]] = frozenset(
        {"GRAMMAR"}
    )


CLAR1: Final = lazy_inject(LanguageToolGrammarErrorCountsClarityAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class LanguageToolGrammarErrorProportionClarityAxiom(
    _LanguageToolErrorProportionClarityAxiom
):
    """
    Prefer text with a lower proportion of characters covered by grammar errors.
    """

    enabled_categories: Final[NoInject[Iterable[_LanguageToolCategory]]] = frozenset(
        {"GRAMMAR"}
    )


CLAR2: Final = lazy_inject(LanguageToolGrammarErrorProportionClarityAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class LanguageToolSpellingErrorCountsClarityAxiom(_LanguageToolErrorCountsClarityAxiom):
    """
    Prefer text with fewer spelling errors.
    """

    enabled_categories: Final[NoInject[Iterable[_LanguageToolCategory]]] = frozenset(
        {
            "CASING",
            "TYPOS",
        }
    )


CLAR3: Final = lazy_inject(LanguageToolSpellingErrorCountsClarityAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class LanguageToolSpellingErrorProportionClarityAxiom(
    _LanguageToolErrorProportionClarityAxiom
):
    """
    Prefer text with a lower proportion of characters covered by spelling errors.
    """

    enabled_categories: Final[NoInject[Iterable[_LanguageToolCategory]]] = frozenset(
        {
            "CASING",
            "TYPOS",
        }
    )


CLAR4: Final = lazy_inject(LanguageToolSpellingErrorProportionClarityAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class LanguageToolRepetitionsErrorCountsClarityAxiom(
    _LanguageToolErrorCountsClarityAxiom
):
    """
    Prefer text with fewer repetitions errors.
    """

    enabled_categories: Final[NoInject[Iterable[_LanguageToolCategory]]] = frozenset(
        {"REPETITIONS"}
    )


CLAR5: Final = lazy_inject(LanguageToolRepetitionsErrorCountsClarityAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class LanguageToolRepetitionsErrorProportionClarityAxiom(
    _LanguageToolErrorProportionClarityAxiom
):
    """
    Prefer text with a lower proportion of characters covered by repetitions errors.
    """

    enabled_categories: Final[NoInject[Iterable[_LanguageToolCategory]]] = frozenset(
        {"REPETITIONS"}
    )


CLAR6: Final = lazy_inject(LanguageToolRepetitionsErrorProportionClarityAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class FleschReadingEaseClarityAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text with higher readability as measured by the Flesch reading ease score.
    """

    text_contents: TextContents[GenerationOutput]

    language_name: NoInject[str] = "en_core_web_sm"
    margin_fraction: NoInject[float] = 0.0

    @cached_property
    def _language(self) -> Language:
        return spacy_load(name=self.language_name)

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        document1 = self._language(self.text_contents.contents(output1))
        document2 = self._language(self.text_contents.contents(output2))

        flesch_reading_ease1 = flesch_reading_ease(document1)
        flesch_reading_ease2 = flesch_reading_ease(document2)

        if isclose(
            flesch_reading_ease1,
            flesch_reading_ease2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_less(flesch_reading_ease1, flesch_reading_ease2)

    def preferences(
        self,
        input: Any,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        documents = (
            self._language(self.text_contents.contents(output)) for output in outputs
        )
        flesch_reading_eases = [
            flesch_reading_ease(document) if len(document) > 0 else nan
            for document in tqdm(
                documents,
                total=len(outputs),
                desc="Flesch reading eases",
                unit="output",
            )
        ]
        return array(
            [
                (
                    strictly_less(flesch_reading_ease1, flesch_reading_ease2)
                    if not isclose(
                        flesch_reading_ease1,
                        flesch_reading_ease2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for flesch_reading_ease1 in flesch_reading_eases
                for flesch_reading_ease2 in flesch_reading_eases
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


CLAR7: Final = lazy_inject(FleschReadingEaseClarityAxiom, injector)
