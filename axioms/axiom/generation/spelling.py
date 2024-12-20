from dataclasses import dataclass
from functools import cached_property
from typing import Final, Sequence, Any

from injector import inject
from numpy import array, float_
from spacy import load as spacy_load
from spacy.language import Language
from spellchecker import SpellChecker

from axioms.axiom.base import Axiom
from axioms.axiom.utils import approximately_equal, strictly_less
from axioms.dependency_injection import injector
from axioms.model.base import Preference, PreferenceMatrix
from axioms.model.generation import GenerationOutput
from axioms.tools import TextContents
from axioms.utils.lazy import lazy_inject


@inject
@dataclass(frozen=True, kw_only=True)
class GenSpellAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer outputs that contain less misspellings.
    """

    text_contents: TextContents[GenerationOutput]

    language_name: str = "en_core_web_sm"
    margin_fraction: float = 0.1

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
            len(self._spell_checker.unknown(words)) / len(words) for words in words
        ]
        return array(
            [
                (
                    0
                    if approximately_equal(
                        misspellings1,
                        misspellings2,
                        margin_fraction=self.margin_fraction,
                    )
                    else strictly_less(misspellings1, misspellings2)
                )
                for misspellings1 in misspellings
                for misspellings2 in misspellings
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


GEN_SPELL: Final = lazy_inject(GenSpellAxiom, injector)
