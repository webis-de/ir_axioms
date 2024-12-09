from typing import Sequence

from nltk import word_tokenize

from axioms.utils.nltk import download_nltk_dependencies
from axioms.tools.tokenizer.base import TermTokenizer


class NltkTermTokenizer(TermTokenizer):
    def __init__(self):
        download_nltk_dependencies("punkt")
        download_nltk_dependencies("punkt_tab")

    def terms(self, text: str) -> Sequence[str]:
        return word_tokenize(text)
