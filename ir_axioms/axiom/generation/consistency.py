"""
Consistency axioms for retrieval-augmented generation.

For a definition of this utility dimension, see: https://doi.org/10.1145/3626772.3657849

- External consistency: Is the statement accurately conveying from the sources?
- Internal consistency: Does the statement not contradict itself?
"""

from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from math import isclose, nan
from typing import Final, Union, Sequence, Any, TYPE_CHECKING, Iterable

from injector import inject, NoInject
from negspacy.negation import Negex
from negspacy.termsets import termset
from numpy import array, float_, zeros
from numpy.typing import NDArray
from rouge_score.rouge_scorer import RougeScorer
from rouge_score.scoring import Score
from spacy import load as spacy_load
from spacy.language import Language
from tqdm.auto import tqdm

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import strictly_greater, strictly_less
from ir_axioms.dependency_injection import injector
from ir_axioms.model.base import Preference, PreferenceMatrix
from ir_axioms.model.generation import GenerationInput, GenerationOutput
from ir_axioms.tools import (
    TextContents,
    AspectExtraction,
    SentenceSimilarity,
    TermTokenizer,
    SentenceTokenizer,
)
from ir_axioms.utils.lazy import lazy_inject


@inject
@dataclass(frozen=True, kw_only=True)
class AspectSimilaritySentenceCountConsistencyAxiom(
    Axiom[GenerationInput, GenerationOutput]
):
    """
    Prefer text with extracted aspects from the input context mentioned in more sentences of the output, weighing by the sentence's similarity to the aspects.
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
        if input.context is None:
            return 0
        context_aspects = set(
            chain.from_iterable(self.aspect_extraction.iter_aspects(input.context))
        )
        if len(context_aspects) == 0:
            return 0

        sentences1 = self.sentence_tokenizer.sentences(
            self.text_contents.contents(output1)
        )
        sentences2 = self.sentence_tokenizer.sentences(
            self.text_contents.contents(output2)
        )

        similarities1 = self.sentence_similarity.paired_similarities(
            list(sentences1), list(context_aspects)
        )
        similarities2 = self.sentence_similarity.paired_similarities(
            list(sentences2), list(context_aspects)
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
        if input.context is None:
            return zeros((len(outputs), len(outputs)))
        context_aspects = self.aspect_extraction.aspects(
            self.text_contents.contents(input)
        )
        if len(context_aspects) == 0:
            return zeros((len(outputs), len(outputs)))

        sentences = (
            self.sentence_tokenizer.sentences(self.text_contents.contents(output))
            for output in outputs
        )

        similarities = (
            self.sentence_similarity.paired_similarities(
                list(sentences), list(context_aspects)
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


CONS1: Final = lazy_inject(AspectSimilaritySentenceCountConsistencyAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class RougeConsistencyAxiom(Axiom[GenerationInput, GenerationOutput]):
    """
    Prefer text with higher ROUGE score compared to the input contexts.
    """

    text_contents: TextContents[Union[GenerationInput, GenerationOutput]]
    sentence_tokenizer: SentenceTokenizer
    term_tokenizer: TermTokenizer

    margin_fraction: NoInject[float] = 0.1

    @cached_property
    def _rouge_scorer(self) -> RougeScorer:
        return RougeScorer(
            rouge_types=["rougeLsum"],
            use_stemmer=False,
            split_summaries=True,
        )

    def preference(
        self,
        input: GenerationInput,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        if input.context is None:
            return 0

        context = "\n\n".join(input.context)

        contents1 = self.text_contents.contents(output1)
        contents2 = self.text_contents.contents(output2)

        rouge1: dict[str, Score] = self._rouge_scorer.score_multi(
            target=context,
            prediction=contents1,
        )
        rouge2: dict[str, Score] = self._rouge_scorer.score(
            target=context,
            prediction=contents2,
        )
        rouge_l_sum1 = rouge1["rougeLsum"].fmeasure
        rouge_l_sum2 = rouge2["rougeLsum"].fmeasure

        if isclose(
            rouge_l_sum1,
            rouge_l_sum2,
            rel_tol=self.margin_fraction,
        ):
            return 0
        return strictly_greater(rouge_l_sum1, rouge_l_sum2)

    def preferences(
        self,
        input: GenerationInput,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        if input.context is None:
            return zeros((len(outputs), len(outputs)))

        context = "\n\n".join(input.context)

        contents = (self.text_contents.contents(output) for output in outputs)

        rouges: Iterable[dict[str, Score]] = (
            self._rouge_scorer.score(
                target=context,
                prediction=content,
            )
            for content in tqdm(
                contents,
                desc="Compute ROUGE",
                total=len(outputs),
                unit="output",
            )
        )
        rouge_l_sums = [rouge["rougeLsum"].fmeasure for rouge in rouges]

        print(rouge_l_sums)

        return array(
            [
                (
                    strictly_greater(rouge_l_sum1, rouge_l_sum2)
                    if not isclose(
                        rouge_l_sum1,
                        rouge_l_sum2,
                        rel_tol=self.margin_fraction,
                    )
                    else 0
                )
                for rouge_l_sum1 in rouge_l_sums
                for rouge_l_sum2 in rouge_l_sums
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


CONS2: Final = lazy_inject(RougeConsistencyAxiom, injector)


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
                desc="Count contradictions",
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


CONS3: Final = lazy_inject(EntityContradictionConsistencyAxiom, injector)
