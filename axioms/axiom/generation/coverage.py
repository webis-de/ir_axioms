# Coverage: how well the information need is addressed
# Broad coverage: response covers diverse information
# - [x] Cover diverse aspects of the query
# Deep coverage: response provides in-depth and informative content
# - [ ] Cover aspects mentioned in query.

from itertools import groupby
from dataclasses import dataclass
from functools import cached_property
from math import isclose, nan
from typing import Final, Union, Sequence, Any, AbstractSet, Iterable

from injector import inject, NoInject
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from numpy import array, float_, zeros
from numpy.typing import NDArray
from spacy import load as spacy_load
from spacy.language import Language
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
    TermTokenizer,
    SentenceTokenizer,
)
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
        aspects1 = self.aspect_extraction.aspects(self.text_contents.contents(output1))
        aspects2 = self.aspect_extraction.aspects(self.text_contents.contents(output2))

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
        aspects2 = self.aspect_extraction.aspects(self.text_contents.contents(output1))
        aspects2 = self.aspect_extraction.aspects(self.text_contents.contents(output2))

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


# TODO: Move to (topical) correctness axioms?
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

        similarity1 = self.sentence_similarity.paired_similarities(
            list(aspects1), list(input_aspects)
        ).mean()
        similarity2 = self.sentence_similarity.paired_similarities(
            list(aspects2), list(input_aspects)
        ).mean()
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
            self.sentence_similarity.paired_similarities(
                list(aspects), list(input_aspects)
            ).mean()
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
class BleuCoverageAxiom(Axiom[GenerationInput, GenerationOutput]):
    """
    Prefer text with higher BLEU score compared to the input text.
    """

    text_contents: TextContents[Union[GenerationInput, GenerationOutput]]
    term_tokenizer: TermTokenizer

    margin_fraction: NoInject[float] = 0.1

    @cached_property
    def _smoothing_function(self) -> SmoothingFunction:
        return SmoothingFunction().method1

    def preference(
        self,
        input: GenerationInput,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        input_terms = self.term_tokenizer.terms(self.text_contents.contents(input))
        terms1 = self.term_tokenizer.terms(self.text_contents.contents(output1))
        terms2 = self.term_tokenizer.terms(self.text_contents.contents(output2))

        bleu1: float = sentence_bleu(
            references=[input_terms],
            hypothesis=terms1,
            smoothing_function=self._smoothing_function,
        )
        bleu2: float = sentence_bleu(
            references=[input_terms],
            hypothesis=terms2,
            smoothing_function=self._smoothing_function,
        )
        if isclose(
            bleu1,
            bleu2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_greater(bleu1, bleu2)

    def preferences(
        self,
        input: GenerationInput,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        input_terms = self.term_tokenizer.terms(self.text_contents.contents(input))
        terms = (
            self.term_tokenizer.terms(self.text_contents.contents(output))
            for output in tqdm(
                outputs,
                desc="Tokenize",
                unit="output",
            )
        )

        bleus = [
            sentence_bleu(
                references=[input_terms],
                hypothesis=terms,
                smoothing_function=self._smoothing_function,
            )
            for terms in terms
        ]

        return array(
            [
                (
                    strictly_greater(bleu1, bleu2)
                    if not isclose(
                        bleu1,
                        bleu2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for bleu1 in bleus
                for bleu2 in bleus
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


COV6: Final = lazy_inject(BleuCoverageAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class AspectRedundancy2CoverageAxiom(Axiom[GenerationInput, GenerationOutput]):
    """
    Prefer text that has less more diverse extracted aspects according to sentence similarity between the aspects, but factor in similarity to the aspects extracted from the input query.
    """

    text_contents: TextContents[Union[GenerationInput, GenerationOutput]]
    aspect_extraction: AspectExtraction
    sentence_similarity: SentenceSimilarity

    margin_fraction: NoInject[float] = 0.0

    def preference(
        self,
        input: GenerationInput,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        input_aspects = list(
            self.aspect_extraction.aspects(self.text_contents.contents(input))
        )

        aspects1 = list(
            self.aspect_extraction.aspects(
                self.text_contents.contents(output1),
            )
        )
        aspects2 = list(
            self.aspect_extraction.aspects(
                self.text_contents.contents(output2),
            )
        )

        input_similarities1 = self.sentence_similarity.paired_similarities(
            aspects1, input_aspects
        )
        input_similarities2 = self.sentence_similarity.paired_similarities(
            aspects2, input_aspects
        )

        aspect_weights1: NDArray[float_] = input_similarities1.mean(axis=1)
        aspect_weights2: NDArray[float_] = input_similarities2.mean(axis=1)

        self_similarities1 = self.sentence_similarity.self_similarities(aspects1)
        self_similarities2 = self.sentence_similarity.self_similarities(aspects2)

        weighted_self_similarities1 = self_similarities1 * aspect_weights1
        weighted_self_similarities2 = self_similarities2 * aspect_weights2

        aggregate_similarity1 = weighted_self_similarities1.mean()
        aggregate_similarity2 = weighted_self_similarities2.mean()

        if isclose(
            aggregate_similarity1,
            aggregate_similarity2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_less(aggregate_similarity1, aggregate_similarity2)

    def preferences(
        self,
        input: GenerationInput,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        input_aspects = list(
            self.aspect_extraction.aspects(self.text_contents.contents(input))
        )

        contents = (self.text_contents.contents(output) for output in outputs)
        aspects = [
            list(aspects)
            for aspects in tqdm(
                self.aspect_extraction.iter_aspects(contents),
                desc="Extract aspects",
                total=len(outputs),
                unit="output",
            )
        ]

        input_similarities = [
            self.sentence_similarity.paired_similarities(aspects, input_aspects)
            for aspects in aspects
        ]
        aspects_weights: list[NDArray[float_]] = [
            similarities.mean(axis=1) for similarities in input_similarities
        ]

        self_similarities = [
            self.sentence_similarity.self_similarities(aspects) for aspects in aspects
        ]

        weighted_self_similarities = [
            similarities * weights
            for similarities, weights in zip(self_similarities, aspects_weights)
        ]

        aggregate_similarities = [
            similarities.mean() for similarities in weighted_self_similarities
        ]

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


COV7: Final = lazy_inject(AspectRedundancy2CoverageAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class EntitySentenceCountCoverageAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text where entities are mentioned in more sentences, on average.
    """

    text_contents: TextContents[GenerationOutput]

    language_name: NoInject[str] = "en_core_web_sm"
    margin_fraction: NoInject[float] = 0.0

    @cached_property
    def _language(self) -> Language:
        return spacy_load(self.language_name)

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        doc1 = self._language(self.text_contents.contents(output1))
        doc2 = self._language(self.text_contents.contents(output2))

        entity_sentence_counts1 = array(
            [
                sum(1 for _ in sents)
                for ent_id, sents in groupby(
                    ((ent.text.lower(), ent.sent) for ent in doc1.ents),
                    key=lambda x: x[0],
                )
            ]
        )
        entity_sentence_counts2 = array(
            [
                sum(1 for _ in sents)
                for ent_id, sents in groupby(
                    ((ent.text.lower(), ent.sent) for ent in doc2.ents),
                    key=lambda x: x[0],
                )
            ]
        )

        avg_entity_sentence_count1 = entity_sentence_counts1.mean()
        avg_entity_sentence_count2 = entity_sentence_counts2.mean()

        if isclose(
            avg_entity_sentence_count1,
            avg_entity_sentence_count2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_greater(avg_entity_sentence_count1, avg_entity_sentence_count2)

    def preferences(
        self,
        input: Any,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        docs = (
            self._language(self.text_contents.contents(output)) for output in outputs
        )

        entity_sentence_counts = (
            array(
                [
                    sum(1 for _ in sents)
                    for ent_id, sents in groupby(
                        ((ent.text.lower(), ent.sent) for ent in doc.ents),
                        key=lambda x: x[0],
                    )
                ]
            )
            for doc in docs
        )

        avg_entity_sentence_counts = [
            counts.mean()
            for counts in tqdm(
                entity_sentence_counts,
                total=len(outputs),
                desc="Count sentences per entity",
                unit="output",
            )
        ]

        return array(
            [
                (
                    strictly_greater(
                        avg_entity_sentence_count1, avg_entity_sentence_count2
                    )
                    if not isclose(
                        avg_entity_sentence_count1,
                        avg_entity_sentence_count2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for avg_entity_sentence_count1 in avg_entity_sentence_counts
                for avg_entity_sentence_count2 in avg_entity_sentence_counts
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


COV8: Final = lazy_inject(EntitySentenceCountCoverageAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class EntityOverlapCoverageAxiom(Axiom[GenerationInput, GenerationOutput]):
    """
    Prefer text with larger overlap of extracted entities to the entities extracted from the input text.
    """

    text_contents: TextContents[Union[GenerationInput, GenerationOutput]]

    language_name: NoInject[str] = "en_core_web_sm"
    margin_fraction: NoInject[float] = 0.1

    @cached_property
    def _language(self) -> Language:
        return spacy_load(self.language_name)

    def preference(
        self,
        input: GenerationInput,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        doc1 = self._language(self.text_contents.contents(output1))
        doc2 = self._language(self.text_contents.contents(output2))

        input_entities = {
            ent.ent_id_
            for ent in self._language(self.text_contents.contents(input)).ents
        }
        entities1 = {ent.ent_id_ for ent in doc1.ents}
        entities2 = {ent.ent_id_ for ent in doc2.ents}

        overlap1 = _coverage(input_entities, entities1)
        overlap2 = _coverage(input_entities, entities2)

        if isclose(
            overlap1,
            overlap2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_greater(overlap1, overlap2)

    def preferences(
        self,
        input: GenerationInput,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        input_entities = {
            ent.ent_id_
            for ent in self._language(self.text_contents.contents(input)).ents
        }

        docs = (
            self._language(self.text_contents.contents(output)) for output in outputs
        )
        entities = ({ent.ent_id_ for ent in doc.ents} for doc in docs)

        overlaps = [
            _coverage(input_entities, entities)
            for entities in tqdm(
                entities,
                desc="Extract entities",
                total=len(outputs),
            )
        ]

        return array(
            [
                (
                    strictly_greater(overlap1, overlap2)
                    if not isclose(
                        overlap1,
                        overlap2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for overlap1 in overlaps
                for overlap2 in overlaps
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


COV9: Final = lazy_inject(EntityOverlapCoverageAxiom, injector)


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