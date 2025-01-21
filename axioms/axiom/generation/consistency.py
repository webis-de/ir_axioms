# Consistency: alignment with source/context and self-contradictions
# External consistency:
# - [x] aspect-based overlap
# - [x] aspect-based similarity
# Internal consistency:
# - [x] self-contradictions

from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from math import isclose, nan
from typing import Final, Union, Sequence, Any, TYPE_CHECKING, AbstractSet, Iterable

from injector import inject, NoInject
from negspacy.negation import Negex
from negspacy.termsets import termset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from numpy import array, float_, zeros
from numpy.typing import NDArray
from rouge_score.rouge_scorer import RougeScorer
from rouge_score.scoring import Score
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
        context_aspects = set(
            chain.from_iterable(self.aspect_extraction.iter_aspects(input.context))
        )
        if len(context_aspects) == 0:
            return 0
        aspects1 = self.aspect_extraction.aspects(self.text_contents.contents(output1))
        aspects2 = self.aspect_extraction.aspects(self.text_contents.contents(output2))

        coverage1 = _coverage(context_aspects, aspects1)
        coverage2 = _coverage(context_aspects, aspects2)

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
        context_aspects = set(
            chain.from_iterable(self.aspect_extraction.iter_aspects(input.context))
        )
        if len(context_aspects) == 0:
            return zeros((len(outputs), len(outputs)))

        output_contents = (self.text_contents.contents(output) for output in outputs)
        aspects = self.aspect_extraction.iter_aspects(output_contents)

        coverage = [
            _coverage(context_aspects, aspects)
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
        context_aspects = set(
            chain.from_iterable(self.aspect_extraction.iter_aspects(input.context))
        )
        if len(context_aspects) == 0:
            return 0
        aspects1 = self.aspect_extraction.aspects(self.text_contents.contents(output1))
        aspects2 = self.aspect_extraction.aspects(self.text_contents.contents(output2))

        jaccard1 = _jaccard(context_aspects, aspects1)
        jaccard2 = _jaccard(context_aspects, aspects2)

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
        context_aspects = set(
            chain.from_iterable(self.aspect_extraction.iter_aspects(input.context))
        )
        if len(context_aspects) == 0:
            return zeros((len(outputs), len(outputs)))

        contents = (self.text_contents.contents(output) for output in outputs)
        aspects = self.aspect_extraction.iter_aspects(contents)

        jaccard = [
            _jaccard(context_aspects, aspects)
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
        context_aspects = set(
            chain.from_iterable(self.aspect_extraction.iter_aspects(input.context))
        )
        if len(context_aspects) == 0:
            return 0

        aspects1 = self.aspect_extraction.aspects(
            self.text_contents.contents(output1),
        )
        aspects2 = self.aspect_extraction.aspects(
            self.text_contents.contents(output2),
        )

        similarity1 = self.sentence_similarity.paired_similarities(
            list(aspects1), list(context_aspects)
        ).mean()
        similarity2 = self.sentence_similarity.paired_similarities(
            list(aspects2), list(context_aspects)
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
        if input.context is None:
            return zeros((len(outputs), len(outputs)))
        context_aspects = list(
            chain.from_iterable(self.aspect_extraction.iter_aspects(input.context))
        )
        if len(context_aspects) == 0:
            return zeros((len(outputs), len(outputs)))

        contents = (self.text_contents.contents(output) for output in outputs)
        aspects = self.aspect_extraction.iter_aspects(contents)

        similarity = [
            self.sentence_similarity.paired_similarities(
                list(aspects), context_aspects
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


CONS4: Final = lazy_inject(EntityContradictionConsistencyAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class BleuConsistencyAxiom(Axiom[GenerationInput, GenerationOutput]):
    """
    Prefer text with higher BLEU score compared to the input contexts.
    """

    text_contents: TextContents[Union[GenerationInput, GenerationOutput]]
    sentence_tokenizer: SentenceTokenizer
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
        if input.context is None:
            return 0

        context_sentences = chain.from_iterable(
            self.sentence_tokenizer.sentences(context) for context in input.context
        )
        context_sentence_terms = [
            self.term_tokenizer.terms(sentence) for sentence in context_sentences
        ]

        sentences1 = self.sentence_tokenizer.sentences(
            self.text_contents.contents(output1)
        )
        sentences2 = self.sentence_tokenizer.sentences(
            self.text_contents.contents(output2)
        )

        sentence_terms1 = [
            self.term_tokenizer.terms(sentence) for sentence in sentences1
        ]
        sentence_terms2 = [
            self.term_tokenizer.terms(sentence) for sentence in sentences2
        ]

        bleu1: float = corpus_bleu(
            list_of_references=[context_sentence_terms for _ in sentences1],
            hypotheses=sentence_terms1,
            smoothing_function=self._smoothing_function,
        )
        bleu2: float = corpus_bleu(
            list_of_references=[context_sentence_terms for _ in sentences2],
            hypotheses=sentence_terms2,
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
        if input.context is None:
            return zeros((len(outputs), len(outputs)))

        context_sentences = chain.from_iterable(
            self.sentence_tokenizer.sentences(context) for context in input.context
        )
        context_sentence_terms = [
            self.term_tokenizer.terms(sentence) for sentence in context_sentences
        ]

        sentences = (
            self.sentence_tokenizer.sentences(self.text_contents.contents(output))
            for output in outputs
        )
        sentence_terms = (
            [self.term_tokenizer.terms(sentence) for sentence in sentences]
            for sentences in sentences
        )

        bleus = [
            corpus_bleu(
                list_of_references=[context_sentence_terms for _ in sentence_terms],
                hypotheses=sentence_terms,
                smoothing_function=self._smoothing_function,
            )
            for sentence_terms in tqdm(
                sentence_terms,
                desc="Compute BLEU",
                total=len(outputs),
                unit="output",
            )
        ]

        print(bleus)

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


CONS5: Final = lazy_inject(BleuConsistencyAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class SentenceSimilarityConsistencyAxiom(Axiom[GenerationInput, GenerationOutput]):
    """
    Prefer text with sentences closer to sentences from the input contexts, as measured by sentence similarity.
    """

    text_contents: TextContents[Union[GenerationInput, GenerationOutput]]
    sentence_tokenizer: SentenceTokenizer
    sentence_similarity: SentenceSimilarity

    margin_fraction: NoInject[float] = 0.1

    def preference(
        self,
        input: GenerationInput,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        if input.context is None:
            return 0

        context_sentences = list(
            chain.from_iterable(
                self.sentence_tokenizer.sentences(context) for context in input.context
            )
        )

        sentences1 = self.sentence_tokenizer.sentences(
            self.text_contents.contents(output1)
        )
        sentences2 = self.sentence_tokenizer.sentences(
            self.text_contents.contents(output2)
        )

        max_sentence_similarities1: NDArray[float_] = (
            self.sentence_similarity.paired_similarities(
                sentences1, context_sentences
            ).max(axis=1)
        )
        max_sentence_similarities2: NDArray[float_] = (
            self.sentence_similarity.paired_similarities(
                sentences2, context_sentences
            ).max(axis=1)
        )

        similarity1 = max_sentence_similarities1.mean()
        similarity2 = max_sentence_similarities2.mean()

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

        context_sentences = list(
            chain.from_iterable(
                self.sentence_tokenizer.sentences(context) for context in input.context
            )
        )

        sentences = (
            self.sentence_tokenizer.sentences(self.text_contents.contents(output))
            for output in outputs
        )

        max_sentence_similarities: Iterable[NDArray[float_]] = (
            self.sentence_similarity.paired_similarities(
                sentences, context_sentences
            ).max(axis=1)
            for sentences in tqdm(
                sentences,
                total=len(outputs),
                desc="Compute similarities",
                unit="output",
            )
        )

        similarities = [
            max_sentence_similarity.mean()
            for max_sentence_similarity in max_sentence_similarities
        ]

        print(similarities)

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


CONS6: Final = lazy_inject(SentenceSimilarityConsistencyAxiom, injector)


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


CONS7: Final = lazy_inject(RougeConsistencyAxiom, injector)


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


CONS8: Final = lazy_inject(AspectSimilaritySentenceCountConsistencyAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class WeightedSentenceSimilarityConsistencyAxiom(
    Axiom[GenerationInput, GenerationOutput]
):
    """
    Prefer text with sentences closer to the most dissimilar sentences from the input contexts, as measured by sentence similarity.
    """

    text_contents: TextContents[Union[GenerationInput, GenerationOutput]]
    sentence_tokenizer: SentenceTokenizer
    sentence_similarity: SentenceSimilarity

    margin_fraction: NoInject[float] = 0.1

    def preference(
        self,
        input: GenerationInput,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        if input.context is None:
            return 0

        context_sentences = list(
            chain.from_iterable(
                self.sentence_tokenizer.sentences(context) for context in input.context
            )
        )
        if len(context_sentences) == 0:
            return 0

        sentences1 = self.sentence_tokenizer.sentences(
            self.text_contents.contents(output1)
        )
        sentences2 = self.sentence_tokenizer.sentences(
            self.text_contents.contents(output2)
        )

        max_sentence_similarities1: NDArray[float_] = (
            self.sentence_similarity.paired_similarities(
                sentences1, context_sentences
            ).max(axis=0)
        )
        max_sentence_similarities2: NDArray[float_] = (
            self.sentence_similarity.paired_similarities(
                sentences2, context_sentences
            ).max(axis=1)
        )

        similarity1 = max_sentence_similarities1.mean()
        similarity2 = max_sentence_similarities2.mean()

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

        context_sentences = list(
            chain.from_iterable(
                self.sentence_tokenizer.sentences(context) for context in input.context
            )
        )

        sentences = (
            self.sentence_tokenizer.sentences(self.text_contents.contents(output))
            for output in outputs
        )

        max_sentence_similarities: Iterable[NDArray[float_]] = (
            self.sentence_similarity.paired_similarities(
                sentences, context_sentences
            ).max(axis=1)
            for sentences in tqdm(
                sentences,
                total=len(outputs),
                desc="Compute similarities",
                unit="output",
            )
        )

        similarities = [
            max_sentence_similarity.mean()
            for max_sentence_similarity in max_sentence_similarities
        ]

        print(similarities)

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


# TODO: How many of the context documents were cited in the output?
