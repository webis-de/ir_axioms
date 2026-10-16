from dataclasses import dataclass
from functools import cached_property
from typing import Any, Optional, Sequence, TypeVar, Protocol, Final

from numpy import array, float_, floating, subtract, zeros
from numpy.random import Generator, default_rng

from ir_axioms.axiom.base import Axiom
from ir_axioms.model import Preference, PreferenceMatrix
from ir_axioms.utils.lazy import lazy_inject


@dataclass(frozen=True, kw_only=True)
class NopAxiom(Axiom[Any, Any]):
    def preference(
        self,
        input: Any,
        output1: Any,
        output2: Any,
    ) -> Preference:
        return 0

    def preferences(
        self,
        input: Any,
        outputs: Sequence[Any],
    ) -> PreferenceMatrix:
        return zeros((len(outputs), len(outputs)))


NOP: Final = lazy_inject(NopAxiom)


@dataclass(frozen=True, kw_only=True)
class RandomAxiom(Axiom[Any, Any]):
    seed: Any | None = None

    @cached_property
    def _generator(self) -> Generator:
        return default_rng(seed=self.seed)  # nosec: B311

    def preference(
        self,
        input: Any,
        output1: Any,
        output2: Any,
    ) -> Preference:
        return self._generator.integers(-1, 1)

    def preferences(
        self,
        input: Any,
        outputs: Sequence[Any],
    ) -> PreferenceMatrix:
        return self._generator.integers(-1, 1, (len(outputs), len(outputs))).astype(
            floating
        )


RANDOM: Final = lazy_inject(RandomAxiom)


_T_contra = TypeVar("_T_contra", contravariant=True)


class _SupportsComparison(Protocol[_T_contra]):
    def __lt__(self, other: _T_contra, /) -> bool: ...
    def __gt__(self, other: _T_contra, /) -> bool: ...


_SupportsComparisonT = TypeVar("_SupportsComparisonT", bound=_SupportsComparison)


@dataclass(frozen=True, kw_only=True)
class GreaterThanAxiom(Axiom[Any, _SupportsComparisonT]):
    def preference(
        self,
        input: Any,
        output1: _SupportsComparisonT,
        output2: _SupportsComparisonT,
    ) -> Preference:
        if output1 > output2:
            return 1
        elif output1 < output2:
            return -1
        return 0


GT: Final = lazy_inject(GreaterThanAxiom)


@dataclass(frozen=True, kw_only=True)
class LessThanAxiom(Axiom[Any, _SupportsComparisonT]):
    def preference(
        self,
        input: Any,
        output1: _SupportsComparisonT,
        output2: _SupportsComparisonT,
    ) -> Preference:
        if output1 < output2:
            return 1
        elif output1 > output2:
            return -1
        return 0


LT: Final = lazy_inject(LessThanAxiom)


_NumberT = TypeVar("_NumberT", int, float)


@dataclass(frozen=True, kw_only=True)
class DistanceAxiom(Axiom[Any, _NumberT]):
    def preference(
        self,
        input: Any,
        output1: _NumberT,
        output2: _NumberT,
    ) -> Preference:
        # Signed distance, so that flipping the outputs flips the preference.
        return output1 - output2

    def preferences(
        self,
        input: Any,
        outputs: Sequence[_NumberT],
    ) -> PreferenceMatrix:
        values = array(outputs, dtype=float_)
        return subtract.outer(values, values)


DIST: Final = lazy_inject(DistanceAxiom)
