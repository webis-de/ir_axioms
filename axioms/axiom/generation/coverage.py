# Coverage: how well the information need is addressed
# Broad coverage: response covers diverse information
# - [ ] Cover diverse aspects of the query
# Deep coverage: response provides in-depth and informative content
# - [x] Cover aspects mentioned in query.

from dataclasses import dataclass
from math import isclose, nan
from typing import Final, Union, Sequence, Any, AbstractSet

from injector import inject, NoInject
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

    margin_fraction: NoInject[float] = 0.1

    def preference(
        self,
        input: GenerationInput,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        input_aspects = self.aspect_extraction.aspects(
            self.text_contents.contents(input)
        )
        if len(input_aspects) == 0:
            return 0
        aspects1 = self.aspect_extraction.aspects(
            self.text_contents.contents(output1)
        )
        aspects2 = self.aspect_extraction.aspects(
            self.text_contents.contents(output2)
        )

        coverage1 = _coverage(input_aspects, aspects1)
        coverage2 = _coverage(input_aspects, aspects2)

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
        input_aspects = self.aspect_extraction.aspects(
            self.text_contents.contents(input)
        )
        if len(input_aspects) == 0:
            return zeros((len(outputs), len(outputs)))

        contents = (self.text_contents.contents(output) for output in outputs)
        aspects = self.aspect_extraction.iter_aspects(contents)

        coverage = [
            _coverage(input_aspects, aspects)
            for aspects in tqdm(
                aspects,
                desc="Extract aspects",
                total=len(outputs),
            )
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

    margin_fraction: NoInject[float] = 0.1

    def preference(
        self,
        input: GenerationInput,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        input_aspects = self.aspect_extraction.aspects(
            self.text_contents.contents(input)
        )
        if len(input_aspects) == 0:
            return 0
        aspects2 = self.aspect_extraction.aspects(
            self.text_contents.contents(output1)
        )
        aspects2 = self.aspect_extraction.aspects(
            self.text_contents.contents(output2)
        )

        jaccard1 = _jaccard(input_aspects, aspects2)
        jaccard2 = _jaccard(input_aspects, aspects2)

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
        input_aspects = self.aspect_extraction.aspects(
            self.text_contents.contents(input)
        )
        if len(input_aspects) == 0:
            return zeros((len(outputs), len(outputs)))

        contents = (self.text_contents.contents(output) for output in outputs)
        aspects = self.aspect_extraction.iter_aspects(contents)

        jaccard = [
            _jaccard(input_aspects, aspects)
            for aspects in tqdm(
                aspects,
                desc="Extract aspects",
                total=len(outputs),
            )
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


COV2: Final = lazy_inject(AspectJaccardCoverageAxiom, injector)


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

    margin_fraction: NoInject[float] = 0.5

    def preference(
        self,
        input: GenerationInput,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        input_aspects = self.aspect_extraction.aspects(
            self.text_contents.contents(input)
        )

        aspects1 = self.aspect_extraction.aspects(
            self.text_contents.contents(output1),
        )
        aspects2 = self.aspect_extraction.aspects(
            self.text_contents.contents(output2),
        )

        similarity1 = self.sentence_similarity.average_similarity(
            aspects1, input_aspects
        )
        similarity2 = self.sentence_similarity.average_similarity(
            aspects2, input_aspects
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
        input_aspects = self.aspect_extraction.aspects(
            self.text_contents.contents(input)
        )

        contents = (self.text_contents.contents(output) for output in outputs)
        aspects = self.aspect_extraction.iter_aspects(contents)

        similarities = [
            self.sentence_similarity.average_similarity(aspects, input_aspects)
            for aspects in tqdm(
                aspects,
                desc="Extract aspects",
                total=len(outputs),
            )
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
                for similarity1 in similarities
                for similarity2 in similarities
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

    margin_fraction: NoInject[float] = 0.0

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        aspects1 = self.aspect_extraction.aspects(
            self.text_contents.contents(output1),
        )
        aspects2 = self.aspect_extraction.aspects(
            self.text_contents.contents(output2),
        )

        count1 = len(aspects1)
        count2 = len(aspects2)
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
        contents = (self.text_contents.contents(output) for output in outputs)
        aspects = self.aspect_extraction.iter_aspects(contents)

        counts = [
            len(aspects)
            for aspects in tqdm(
                aspects,
                desc="Extract aspects",
                total=len(outputs),
            )
        ]

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

    margin_fraction: NoInject[float] = 0.2

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        aspects1 = self.aspect_extraction.aspects(
            self.text_contents.contents(output1),
        )
        aspects2 = self.aspect_extraction.aspects(
            self.text_contents.contents(output2),
        )

        similarities1 = self.sentence_similarity.similarities(list(aspects1))
        similarities2 = self.sentence_similarity.similarities(list(aspects2))

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
        contents = (self.text_contents.contents(output) for output in outputs)
        aspects = self.aspect_extraction.iter_aspects(contents)
        similarities = (
            self.sentence_similarity.similarities(list(aspects))
            for aspects in tqdm(
                aspects,
                desc="Extract aspects",
                total=len(outputs),
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


COV5: Final = lazy_inject(AspectRedundancyCoverageAxiom, injector)
