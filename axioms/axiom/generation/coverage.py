"""
Coverage axioms for retrieval-augmented generation.

For a definition of this utility dimension, see: https://doi.org/10.1145/3626772.3657849

- Broad coverage: Does the response cover diverse information?
- Deep coverage: Does the response provide in-depth and highly informative content?
"""

from dataclasses import dataclass
from math import isclose
from typing import Final, Union, Sequence, Any, Iterable

from injector import inject, NoInject
from numpy import array, float_, zeros
from numpy.typing import NDArray
from tqdm.auto import tqdm

from axioms.axiom.base import Axiom
from axioms.axiom.utils import strictly_greater, strictly_less
from axioms.dependency_injection import injector
from axioms.model.base import Preference, PreferenceMatrix
from axioms.model.generation import GenerationInput, GenerationOutput
from axioms.tools import (
    TextContents,
    AspectExtraction,
    SentenceSimilarity,
    SentenceTokenizer,
)
from axioms.utils.lazy import lazy_inject


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

        similarities1 = self.sentence_similarity.self_similarities(list(aspects1))
        similarities2 = self.sentence_similarity.self_similarities(list(aspects2))

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
            self.sentence_similarity.self_similarities(list(aspects))
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


@inject
@dataclass(frozen=True, kw_only=True)
class AspectSimilaritySentenceCountCoverageAxiom(
    Axiom[GenerationInput, GenerationOutput]
):
    """
    Prefer text with extracted aspects from the input text mentioned in more sentences of the output, weighing by the sentence's similarity to the aspects.
    """

    text_contents: TextContents[Union[GenerationInput, GenerationOutput]]
    aspect_extraction: AspectExtraction
    sentence_tokenizer: SentenceTokenizer
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
        if len(input_aspects) == 0:
            return 0

        sentences1 = self.sentence_tokenizer.sentences(
            self.text_contents.contents(output1)
        )
        sentences2 = self.sentence_tokenizer.sentences(
            self.text_contents.contents(output2)
        )

        similarities1 = self.sentence_similarity.paired_similarities(
            list(sentences1), list(input_aspects)
        )
        similarities2 = self.sentence_similarity.paired_similarities(
            list(sentences2), list(input_aspects)
        )

        sentence_weights1: NDArray[float_] = similarities1.mean(axis=1)
        sentence_weights2: NDArray[float_] = similarities2.mean(axis=1)

        sentence_counts1 = sentence_weights1.sum()
        sentence_counts2 = sentence_weights2.sum()

        if isclose(
            sentence_counts1,
            sentence_counts2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_greater(sentence_counts1, sentence_counts2)

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

        sentences = (
            self.sentence_tokenizer.sentences(self.text_contents.contents(output))
            for output in outputs
        )

        similarities = (
            self.sentence_similarity.paired_similarities(
                list(sentences), list(input_aspects)
            )
            for sentences in tqdm(
                sentences,
                desc="Aspect-sentence similarities",
                total=len(outputs),
            )
        )

        sentence_weights: Iterable[NDArray[float_]] = (
            weights.mean(axis=1) for weights in similarities
        )

        sentence_counts = [weights.sum() for weights in sentence_weights]

        return array(
            [
                (
                    strictly_greater(sentence_count1, sentence_count2)
                    if not isclose(
                        sentence_count1,
                        sentence_count2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for sentence_count1 in sentence_counts
                for sentence_count2 in sentence_counts
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


COV10: Final = lazy_inject(AspectSimilaritySentenceCountCoverageAxiom, injector)

# TODO: Weighted aspect similarity sentence length axiom.
