# Correctness: factual verifiability and topical alignment
# Factual correctness:
# - [ ] Verifyaiity/faithfulness
#   - Implementation: Extract sentences containing aspects from input, then find contradictions of the retrieved context as evidence.
# Topical correctness:
# - [ ] Topical relevance/alignment (-> use retrieval axioms?)
# TODO: Propose axioms for correctness.


from dataclasses import dataclass
from math import isclose
from re import compile as re_compile
from typing import Final, Sequence, Any

from injector import inject, NoInject
from numpy import array, float_
from tqdm.auto import tqdm

from axioms.axiom.base import Axiom
from axioms.axiom.utils import strictly_less
from axioms.dependency_injection import injector
from axioms.model.base import Preference, PreferenceMatrix
from axioms.model.generation import GenerationOutput
from axioms.tools import (
    TextContents,
    SentenceTokenizer,
)
from axioms.utils.lazy import lazy_inject

_PATTERN_URL: Final = re_compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)

_PATTERN_CITATION: Final = re_compile(r"\[[0-9]+\]")


@inject
@dataclass(frozen=True, kw_only=True)
class UrlSentenceCorrectnessAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text with more sentences containing HTTP URLs (for reference).
    """

    text_contents: TextContents[GenerationOutput]
    sentence_tokenizer: SentenceTokenizer

    margin_fraction: NoInject[float] = 0.1

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        contains_url1 = array(
            [
                _PATTERN_URL.search(sentence) is not None
                for sentence in self.sentence_tokenizer.sentences(
                    self.text_contents.contents(output1)
                )
            ]
        )
        contains_url2 = array(
            [
                _PATTERN_URL.search(sentence) is not None
                for sentence in self.sentence_tokenizer.sentences(
                    self.text_contents.contents(output2)
                )
            ]
        )

        url_coverage1 = contains_url1.mean()
        url_coverage2 = contains_url2.mean()

        if isclose(
            url_coverage1,
            url_coverage2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_less(url_coverage1, url_coverage2)

    def preferences(
        self,
        input: Any,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        contains_url = (
            array(
                [
                    _PATTERN_URL.search(sentence) is not None
                    for sentence in self.sentence_tokenizer.sentences(
                        self.text_contents.contents(output)
                    )
                ]
            )
            for output in tqdm(
                outputs,
                desc="Find URLs",
                unit="output",
            )
        )
        url_coverages = [contains_url.mean() for contains_url in contains_url]
        return array(
            [
                (
                    strictly_less(url_coverage1, url_coverage2)
                    if not isclose(
                        url_coverage1,
                        url_coverage2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for url_coverage1 in url_coverages
                for url_coverage2 in url_coverages
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


CORR1: Final = lazy_inject(UrlSentenceCorrectnessAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class CitationSentenceCorrectnessAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text with more sentences containing citations.
    """

    text_contents: TextContents[GenerationOutput]
    sentence_tokenizer: SentenceTokenizer

    margin_fraction: NoInject[float] = 0.1

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        contains_citation1 = array(
            [
                _PATTERN_CITATION.search(sentence) is not None
                for sentence in self.sentence_tokenizer.sentences(
                    self.text_contents.contents(output1)
                )
            ]
        )
        contains_citation2 = array(
            [
                _PATTERN_CITATION.search(sentence) is not None
                for sentence in self.sentence_tokenizer.sentences(
                    self.text_contents.contents(output2)
                )
            ]
        )

        citation_coverage1 = contains_citation1.mean()
        citation_coverage2 = contains_citation2.mean()

        if isclose(
            citation_coverage1,
            citation_coverage2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_less(citation_coverage1, citation_coverage2)

    def preferences(
        self,
        input: Any,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        contains_citation = (
            array(
                [
                    _PATTERN_CITATION.search(sentence) is not None
                    for sentence in self.sentence_tokenizer.sentences(
                        self.text_contents.contents(output)
                    )
                ]
            )
            for output in tqdm(
                outputs,
                desc="Find citations",
                unit="output",
            )
        )
        citation_coverages = [
            contains_citation.mean() for contains_citation in contains_citation
        ]
        return array(
            [
                (
                    strictly_less(citation_coverage1, citation_coverage2)
                    if not isclose(
                        citation_coverage1,
                        citation_coverage2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for citation_coverage1 in citation_coverages
                for citation_coverage2 in citation_coverages
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


CORR2: Final = lazy_inject(CitationSentenceCorrectnessAxiom, injector)
