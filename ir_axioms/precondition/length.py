from dataclasses import dataclass
from itertools import product
from math import isclose
from typing import Final, Sequence, TypeVar

from injector import inject
from numpy import array, bool_
from tqdm.auto import tqdm

from ir_axioms.dependency_injection import injector
from ir_axioms.model import Document, Mask, MaskMatrix
from ir_axioms.precondition.base import Precondition
from ir_axioms.tools import TextContents, TermTokenizer
from ir_axioms.utils.lazy import lazy_inject

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
        return isclose(
            len(self.term_tokenizer.terms(self.text_contents.contents(output1))),
            len(self.term_tokenizer.terms(self.text_contents.contents(output2))),
            rel_tol=self.margin_fraction,
        )

    def preconditions(
        self,
        input: Input,
        outputs: Sequence[Document],
    ) -> MaskMatrix:
        lengths = [
            len(self.term_tokenizer.terms(self.text_contents.contents(output)))
            for output in tqdm(
                outputs,
                desc="Lengths",
                unit="document",
            )
        ]
        return array(
            [
                isclose(
                    length1,
                    length2,
                    rel_tol=self.margin_fraction,
                )
                for length1, length2 in tqdm(
                    product(lengths, repeat=2),
                    total=len(lengths) * len(lengths),
                    desc="Compare lengths",
                    unit="pair",
                )
            ],
            dtype=bool_,
        ).reshape((len(outputs), len(outputs)))


LEN: Final = lazy_inject(LenPrecondition, injector)
