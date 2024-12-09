from dataclasses import dataclass
from typing import Final, TypeVar

from injector import inject

from axioms.axiom.utils import approximately_equal
from axioms.dependency_injection import injector
from axioms.model import Document, Mask
from axioms.precondition.base import Precondition
from axioms.tools import TextContents, TermTokenizer
from axioms.utils.lazy import lazy_inject

Input = TypeVar("Input")


@inject
@dataclass(frozen=True, kw_only=True)
class LenPrecondition(Precondition[Input, Document]):
    text_contents: TextContents[Document]
    term_tokenizer: TermTokenizer
    margin_fraction: float = 0.1

    def precondition(
        self,
        input: Input,
        output1: Document,
        output2: Document,
    ) -> Mask:
        return approximately_equal(
            len(self.term_tokenizer.terms(self.text_contents.contents(output1))),
            len(self.term_tokenizer.terms(self.text_contents.contents(output2))),
            margin_fraction=self.margin_fraction,
        )


LEN: Final = lazy_inject(LenPrecondition, injector)
