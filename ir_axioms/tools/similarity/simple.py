from dataclasses import dataclass
from injector import inject


from ir_axioms.tools.tokenizer.base import TermTokenizer
from ir_axioms.tools.similarity.base import SentenceSimilarity, TermSimilarity


@inject
@dataclass(frozen=True)
class AverageTermSimilaritySentenceSimilarity(SentenceSimilarity):
    """
    Sentence similarity based on the mean term similarity of the terms in the sentences.

    Note: This is only meant as a fallback if no other sentence similarity model is available.
    """

    term_tokenizer: TermTokenizer
    term_similarity: TermSimilarity

    def similarity(self, sentence1: str, sentence2: str) -> float:
        terms1 = self.term_tokenizer.terms_unordered(sentence1)
        terms2 = self.term_tokenizer.terms_unordered(sentence2)
        return self.term_similarity.average_similarity(terms1, terms2)
