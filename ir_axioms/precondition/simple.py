from dataclasses import dataclass
from typing import Final, Sequence, TypeVar

from injector import inject
from numpy import bool_, full

from ir_axioms.model import Document, Mask, MaskMatrix
from ir_axioms.precondition.base import Precondition
from ir_axioms.utils.lazy import lazy_inject

Input = TypeVar("Input")


@inject
@dataclass(frozen=True, kw_only=True)
class NopPrecondition(Precondition[Input, Document]):
    def precondition(
        self,
        input: Input,
        output1: Document,
        output2: Document,
    ) -> Mask:
        return True

    def preconditions(
        self,
        input: Input,
        outputs: Sequence[Document],
    ) -> MaskMatrix:
        return full((len(outputs), len(outputs)), True, dtype=bool_)


NOP: Final = lazy_inject(NopPrecondition)
