from dataclasses import dataclass
from typing import Final, Sequence, TypeVar

from injector import inject
from numpy import array, bool_

from axioms.axiom.utils import approximately_equal
from axioms.dependency_injection import injector
from axioms.model import Document, Mask, MaskMatrix
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

    def preconditions(
        self,
        input: Input,
        outputs: Sequence[Document],
    ) -> MaskMatrix:
        lengths = [
            len(self.term_tokenizer.terms(self.text_contents.contents(output)))
            for output in outputs
        ]
        return array(
            [
                approximately_equal(
                    length1,
                    length2,
                    margin_fraction=self.margin_fraction,
                )
                for length1 in lengths
                for length2 in lengths
            ],
            dtype=bool_,
        ).reshape((len(outputs), len(outputs)))


LEN: Final = lazy_inject(LenPrecondition, injector)
