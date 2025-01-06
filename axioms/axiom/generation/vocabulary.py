# TODO: Preconditions: approx. semantic equivalence of both texts
# TODO: Add n-gram generalizations (NetSpeak).

from dataclasses import dataclass
from math import isclose
from typing import Final, Mapping, Sequence, Any, Union

from numpy import array, float_
from scipy.special import rel_entr
from tqdm import tqdm
from wordfreq import word_frequency

from axioms.axiom.base import Axiom
from axioms.axiom.utils import strictly_less
from axioms.dependency_injection import injector
from axioms.model import GenerationInput, PreferenceMatrix
from axioms.model.base import Preference
from axioms.model.generation import GenerationOutput
from axioms.tools import TextStatistics
from axioms.utils.lazy import lazy_inject


@dataclass(frozen=True, kw_only=True)
class GenerativeWordCommonnessAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text with more common vocabulary according to some reference corpus, e.g., the Wikipedia.
    """

    text_statistics: TextStatistics[GenerationOutput]

    language: str = "en"
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


GEN_W_COMM: Final = lazy_inject(GenerativeWordCommonnessAxiom, injector)


@dataclass(frozen=True, kw_only=True)
class GenerativeNormalizedWordCommonnessAxiom(Axiom[GenerationInput, GenerationOutput]):
    """
    Prefer text with more common vocabulary according to some reference corpus, but factor in the query's word frequencies.
    """

    text_statistics: TextStatistics[Union[GenerationInput, GenerationOutput]]

    language: str = "en"
    margin_fraction: float = 0.1

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
            [input_term_frequencies[term] for term in unique_terms]
        )

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


GEN_N_W_COMM: Final = lazy_inject(GenerativeNormalizedWordCommonnessAxiom, injector)
