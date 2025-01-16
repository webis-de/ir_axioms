# Consistency: alignment with source/context and self-contradictions
# External consistency:
# - [x] aspect-based overlap
# - [x] aspect-based similarity
# Internal consistency:
# - [x] self-contradictions

from dataclasses import dataclass
from functools import cached_property
from math import isclose, nan
from typing import Final, Union, Sequence, Any, TYPE_CHECKING, AbstractSet

from injector import inject, NoInject
from negspacy.negation import Negex
from negspacy.termsets import termset
from numpy import array, float_, zeros
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
class AspectOverlapConsistenyAxiom(Axiom[GenerationInput, GenerationOutput]):
    """
    Prefer text with larger overlap of extracted aspects to the aspects extracted from the input contexts.
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
        if input.context is None:
            return 0
        context_unique_aspects = {
            aspect
            for context in input.context
            for aspect in self.aspect_extraction.unique_aspects(context)
        }
        if len(context_unique_aspects) == 0:
            return 0
        output1_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(output1)
        )
        output2_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(output2)
        )

        coverage1 = _coverage(context_unique_aspects, output1_unique_aspects)
        coverage2 = _coverage(context_unique_aspects, output2_unique_aspects)

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
        if input.context is None:
            return zeros((len(outputs), len(outputs)))
        context_unique_aspects = {
            aspect
            for context in input.context
            for aspect in self.aspect_extraction.unique_aspects(context)
        }
        if len(context_unique_aspects) == 0:
            return zeros((len(outputs), len(outputs)))
        output_unique_aspects = (
            self.aspect_extraction.unique_aspects(self.text_contents.contents(output))
            for output in tqdm(
                outputs,
                desc="Extract aspects",
            )
        )

        coverage = [
            _coverage(context_unique_aspects, aspects)
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


CONS1: Final = lazy_inject(AspectOverlapConsistenyAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class AspectJaccardConsistencyAxiom(Axiom[GenerationInput, GenerationOutput]):
    """
    Prefer text with larger Jaccard index of extracted aspects to the aspects extracted from the input contexts.
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
        if input.context is None:
            return 0
        context_unique_aspects = {
            aspect
            for context in input.context
            for aspect in self.aspect_extraction.unique_aspects(context)
        }
        if len(context_unique_aspects) == 0:
            return 0
        # FIXME: Penalize by the number of repeated aspects in the output.
        output1_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(output1)
        )
        output2_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(output2)
        )

        jaccard1 = _jaccard(context_unique_aspects, output1_unique_aspects)
        jaccard2 = _jaccard(context_unique_aspects, output2_unique_aspects)

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
        if input.context is None:
            return zeros((len(outputs), len(outputs)))
        context_unique_aspects = {
            aspect
            for context in input.context
            for aspect in self.aspect_extraction.unique_aspects(context)
        }
        if len(context_unique_aspects) == 0:
            return zeros((len(outputs), len(outputs)))
        output_unique_aspects = (
            self.aspect_extraction.unique_aspects(self.text_contents.contents(output))
            for output in tqdm(
                outputs,
                desc="Extract aspects",
            )
        )

        jaccard = [
            _jaccard(context_unique_aspects, aspects)
            for aspects in output_unique_aspects
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


# TODO: Less harsh penalization for the number of aspects in the output.

CONS2: Final = lazy_inject(AspectJaccardConsistencyAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class AspectSimilarityConsistencyAxiom(Axiom[GenerationInput, GenerationOutput]):
    """
    Prefer text with extracted aspects more similar to the aspects from in the input contexts.

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
        if input.context is None:
            return 0
        context_unique_aspects = {
            aspect
            for context in input.context
            for aspect in self.aspect_extraction.unique_aspects(context)
        }

        document1_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(output1),
        )
        document2_unique_aspects = self.aspect_extraction.unique_aspects(
            self.text_contents.contents(output2),
        )

        similarity1 = self.sentence_similarity.average_similarity(
            document1_unique_aspects, context_unique_aspects
        )
        similarity2 = self.sentence_similarity.average_similarity(
            document2_unique_aspects, context_unique_aspects
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
        if input.context is None:
            return zeros((len(outputs), len(outputs)))
        context_unique_aspects = {
            aspect
            for context in input.context
            for aspect in self.aspect_extraction.unique_aspects(context)
        }

        document_unique_aspects = (
            self.aspect_extraction.unique_aspects(self.text_contents.contents(output))
            for output in tqdm(
                outputs,
                desc="Extract aspects",
            )
        )

        similarity = [
            self.sentence_similarity.average_similarity(
                document_unique_aspects, context_unique_aspects
            )
            for document_unique_aspects in document_unique_aspects
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


CONS3: Final = lazy_inject(AspectSimilarityConsistencyAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class EntityContradictionConsistencyAxiom(Axiom[Any, GenerationOutput]):
    """
    Prefer text with entities less frequently mentioned in contradictory phrases.
    """

    text_contents: TextContents[GenerationOutput]

    language_name: NoInject[str] = "en_core_web_sm"
    margin_fraction: NoInject[float] = 0.2

    # TODO: Migrate to tool injected by DI.
    @cached_property
    def _language(self) -> Language:
        if TYPE_CHECKING:
            Negex  # To ignore the unused import warning.
        language = spacy_load(name=self.language_name)
        neg_termset = termset("en")
        language.add_pipe(
            factory_name="negex",
            config={
                "ent_types": [
                    # "CARDINAL",  # other numbers
                    # "DATE",
                    "EVENT",  # event names
                    "FAC",  # facilities
                    "GPE",  # geopolitical entities
                    "LANGUAGE",
                    "LAW",  # legal document titles
                    "LOC",  # other geographic locations
                    # "MONEY",
                    "NORP",  # nationalities, religious groups, political groups
                    # "ORDINAL",
                    "ORG",
                    # "PERCENT",
                    "PERSON",
                    "PRODUCT",
                    # "QUANTITY",
                    # "TIME",
                    "WORK_OF_ART",
                ],
                "neg_termset": neg_termset.get_patterns(),
            },
        )
        return language

    def _contradictions_ratio(self, text: str) -> float:
        document = self._language(text)
        entity_negated_pairs: set[tuple[str, bool]] = {
            (entity.text, entity._.negex)
            for entity in document.ents
            if entity._.negex is not None
        }
        num_contradictions = len(entity_negated_pairs) - len(
            {entity for entity, _ in entity_negated_pairs}
        )
        num_entities = sum(1 for _ in document.ents)
        return num_contradictions / num_entities if num_entities != 0 else nan

    def preference(
        self,
        input: Any,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        contents1 = self.text_contents.contents(output1)
        contents2 = self.text_contents.contents(output2)
        contradictions1 = self._contradictions_ratio(contents1)
        contradictions2 = self._contradictions_ratio(contents2)
        if isclose(
            contradictions1,
            contradictions2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_less(contradictions1, contradictions2)

    def preferences(
        self,
        input: Any,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        contents = (
            self.text_contents.contents(output)
            for output in tqdm(
                outputs,
                desc="Counting contradictions",
                unit="output",
            )
        )
        contradictions = [self._contradictions_ratio(content) for content in contents]
        return array(
            [
                (
                    strictly_less(contradictions1, contradictions2)
                    if not isclose(
                        contradictions1,
                        contradictions2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for contradictions1 in contradictions
                for contradictions2 in contradictions
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


CONS4: Final = lazy_inject(EntityContradictionConsistencyAxiom, injector)
