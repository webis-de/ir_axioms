# Coverage: how well the information need is addressed
# Broad coverage: response covers diverse information
# - [ ] Cover diverse aspects of the query
# Deep coverage: response provides in-depth and informative content
# - [x] Cover aspects mentioned in query.

from dataclasses import dataclass
from math import isclose, nan
from typing import Final, Union, Sequence, Any, AbstractSet

from injector import inject
from numpy import array, float_, zeros
from tqdm.auto import tqdm

from axioms.axiom.base import Axiom
from axioms.axiom.utils import strictly_greater, strictly_less
from axioms.dependency_injection import injector
from axioms.model.base import Preference, PreferenceMatrix
from axioms.model.generation import GenerationInput, GenerationOutput
from axioms.tools import TextContents, AspectExtraction, SentenceSimilarity
from axioms.utils.lazy import lazy_inject


def _coverage(
    a: AbstractSet[str],
    b: AbstractSet[str],
) -> float:
    divisor = min(len(a), len(b))
    if divisor == 0:
        return nan
    return len(a & b) / divisor


def _jaccard(
    a: AbstractSet[str],
    b: AbstractSet[str],
) -> float:
    divisor = len(a | b)
    if divisor == 0:
        return nan
    return len(a & b) / divisor


@inject
@dataclass(frozen=True, kw_only=True)
class AspectOverlapCoverageAxiom(Axiom[GenerationInput, GenerationOutput]):
    """
    Prefer text with larger overlap of extracted aspects to the aspects extracted from the input text.
    """

    text_contents: TextContents[Union[GenerationInput, GenerationOutput]]
    aspect_extraction: AspectExtraction

    margin_fraction: float = 0.1

    def preference(
        self,
        input: GenerationInput,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        input_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(input)
        )
        if len(input_unique_aspects) == 0:
            return 0
        output1_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(output1)
        )
        output2_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(output2)
        )

        coverage1 = _coverage(input_unique_aspects, output1_unique_aspects)
        coverage2 = _coverage(input_unique_aspects, output2_unique_aspects)

        if isclose(
            coverage1,
            coverage2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_greater(coverage1, coverage2)

    def preferences(
        self,
        input: GenerationInput,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        input_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(input)
        )
        if len(input_unique_aspects) == 0:
            return zeros((len(outputs), len(outputs)))
        output_unique_aspects = (
            self.aspect_extraction.unique_aspects(self.text_contents.contents(output))
            for output in tqdm(
                outputs,
                desc="Extract aspects",
            )
        )

        coverage = [
            _coverage(input_unique_aspects, aspects)
            for aspects in output_unique_aspects
        ]

        return array(
            [
                (
                    strictly_greater(coverage1, coverage2)
                    if not isclose(
                        coverage1,
                        coverage2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for coverage1 in coverage
                for coverage2 in coverage
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


COV1: Final = lazy_inject(AspectOverlapCoverageAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class AspectJaccardCoverageAxiom(Axiom[GenerationInput, GenerationOutput]):
    """
    Prefer text with larger Jaccard index of extracted aspects to the aspects extracted from the input text.
    """

    text_contents: TextContents[Union[GenerationInput, GenerationOutput]]
    aspect_extraction: AspectExtraction

    margin_fraction: float = 0.1

    def preference(
        self,
        input: GenerationInput,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        input_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(input)
        )
        if len(input_unique_aspects) == 0:
            return 0
        output1_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(output1)
        )
        output2_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(output2)
        )

        jaccard1 = _jaccard(input_unique_aspects, output1_unique_aspects)
        jaccard2 = _jaccard(input_unique_aspects, output2_unique_aspects)

        if isclose(
            jaccard1,
            jaccard2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_greater(jaccard1, jaccard2)

    def preferences(
        self,
        input: GenerationInput,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        input_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(input)
        )
        if len(input_unique_aspects) == 0:
            return zeros((len(outputs), len(outputs)))
        output_unique_aspects = (
            self.aspect_extraction.unique_aspects(self.text_contents.contents(output))
            for output in tqdm(
                outputs,
                desc="Extract aspects",
            )
        )

        jaccard = [
            _jaccard(input_unique_aspects, aspects) for aspects in output_unique_aspects
        ]

        return array(
            [
                (
                    strictly_greater(jaccard1, jaccard2)
                    if not isclose(
                        jaccard1,
                        jaccard2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for jaccard1 in jaccard
                for jaccard2 in jaccard
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


COV2: Final = lazy_inject(AspectOverlapCoverageAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class AspectSimilarityCoverageAxiom(Axiom[GenerationInput, GenerationOutput]):
    """
    Prefer text with extracted aspects more similar to the aspects from in the input text.

    This axiom is an adaption of STMC1, but uses aspects instead of terms and sentence similarity.
    """

    text_contents: TextContents[Union[GenerationInput, GenerationOutput]]
    aspect_extraction: AspectExtraction
    sentence_similarity: SentenceSimilarity

    margin_fraction: float = 0.5

    def preference(
        self,
        input: GenerationInput,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        input_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(input)
        )

        document1_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(output1),
        )
        document2_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(output2),
        )

        similarity1 = self.sentence_similarity.average_similarity(
            document1_unique_aspects, input_unique_aspects
        )
        similarity2 = self.sentence_similarity.average_similarity(
            document2_unique_aspects, input_unique_aspects
        )
        if isclose(
            similarity1,
            similarity2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_greater(similarity1, similarity2)

    def preferences(
        self,
        input: GenerationInput,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        input_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(input)
        )

        document_unique_aspects = (
            self.aspect_extraction.unique_aspects(self.text_contents.contents(output))
            for output in tqdm(
                outputs,
                desc="Extract aspects",
            )
        )

        similarity = [
            self.sentence_similarity.average_similarity(aspects, input_unique_aspects)
            for aspects in document_unique_aspects
        ]

        return array(
            [
                (
                    strictly_greater(similarity1, similarity2)
                    if not isclose(
                        similarity1,
                        similarity2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for similarity1 in similarity
                for similarity2 in similarity
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


COV3: Final = lazy_inject(AspectSimilarityCoverageAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class AspectCountCoverageAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text with more distinct extracted aspects.
    """

    text_contents: TextContents[GenerationOutput]
    aspect_extraction: AspectExtraction

    margin_fraction: float = 0.0

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        document1_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(output1),
        )
        document2_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(output2),
        )

        count1 = len(document1_unique_aspects)
        count2 = len(document2_unique_aspects)
        if isclose(
            count1,
            count2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_greater(count1, count2)

    def preferences(
        self,
        input: Any,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        document_unique_aspects = (
            self.aspect_extraction.unique_aspects(self.text_contents.contents(output))
            for output in tqdm(
                outputs,
                desc="Extract aspects",
            )
        )

        counts = [len(aspects) for aspects in document_unique_aspects]

        return array(
            [
                (
                    strictly_greater(count1, count2)
                    if not isclose(
                        count1,
                        count2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for count1 in counts
                for count2 in counts
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


COV4: Final = lazy_inject(AspectCountCoverageAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class AspectRedundancyCoverageAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text that has less redundant extracted aspects according to sentence similarity between the aspects.
    """

    text_contents: TextContents[GenerationOutput]
    aspect_extraction: AspectExtraction
    sentence_similarity: SentenceSimilarity

    margin_fraction: float = 0.2

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        document1_aspects = self.aspect_extraction.aspects(
            self.text_contents.contents(output1),
        )
        document2_aspects = self.aspect_extraction.aspects(
            self.text_contents.contents(output2),
        )

        similarities1 = self.sentence_similarity.similarities(document1_aspects)
        similarities2 = self.sentence_similarity.similarities(document2_aspects)

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
        document_aspects = (
            self.aspect_extraction.aspects(self.text_contents.contents(output))
            for output in tqdm(
                outputs,
                desc="Extract aspects",
            )
        )
        similarities = (
            self.sentence_similarity.similarities(aspects)
            for aspects in document_aspects
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


COV5: Final = lazy_inject(AspectRedundancyCoverageAxiom, injector)
