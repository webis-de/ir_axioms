from dataclasses import dataclass
from dbm import open as dbm_open
from functools import cached_property
from pathlib import Path
from struct import pack, unpack
from typing import Iterable, Iterator, Protocol, Sequence, Tuple, TypeVar

from numpy import array, float_, ndarray, isnan, ones_like, nan
from optimask import OptiMask
from tqdm.auto import tqdm
from typing_extensions import TypeAlias  # type: ignore

from axioms.axiom.base import Axiom
from axioms.model import Preference, PreferenceMatrix


class SupportsRepr(Protocol):
    def __repr__(self) -> str:
        pass


_Input = TypeVar("_Input", bound=SupportsRepr)
_Output = TypeVar("_Output", bound=SupportsRepr)


@dataclass(frozen=True, kw_only=True)
class DbmCachedAxiom(Axiom[_Input, _Output]):
    axiom: Axiom[_Input, _Output]
    cache_path: Path

    def _iter_preferences(
        self,
        inputs_outputs: Iterable[Tuple[_Input, _Output, _Output]],
        only_cached: bool = False,
    ) -> Iterator[Preference]:
        self.cache_path.parent.mkdir(exist_ok=True, parents=True)
        with dbm_open(self.cache_path, flag="c") as cache:
            for input, output1, output2 in inputs_outputs:
                key = repr((input, output1, output2)).encode(encoding="utf-8")
                if key in cache:
                    preference_bytes = cache[key]
                    (preference,) = unpack("f", preference_bytes)
                    if isnan(preference):
                        raise RuntimeError(
                            f"Invalid cache. Please delete cache file at: {self.cache_path}"
                        )
                    yield preference
                elif only_cached:
                    yield nan
                else:
                    preference = self.axiom.preference(input, output1, output2)
                    preference_bytes = pack("f", preference)
                    cache[key] = preference_bytes
                    yield preference

    def preference(
        self,
        input: _Input,
        output1: _Output,
        output2: _Output,
    ) -> Preference:
        return next(
            self._iter_preferences(
                inputs_outputs=((input, output1, output2),),
            )
        )

    def _cached_preferences(
        self,
        input: _Input,
        outputs: Sequence[_Output],
    ) -> ndarray:
        return array(
            list(
                self._iter_preferences(
                    (
                        (input, output1, output2)
                        for output2 in outputs
                        for output1 in outputs
                    ),
                    only_cached=True,
                )
            ),
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))

    @cached_property
    def _opti_mask(self) -> OptiMask:
        return OptiMask()

    def preferences(
        self,
        input: _Input,
        outputs: Sequence[_Output],
    ) -> PreferenceMatrix:
        # First, only load the cached dependencies.
        preferences = self._cached_preferences(input, outputs)

        # Then, fill as-large-as-possible sub-blocks of the matrix by computing preference matrices.
        while isnan(preferences).any():
            if isnan(preferences).sum() == preferences.size:
                start = 0
                end = preferences.shape[0]
            else:
                mask = ones_like(preferences)
                mask[~isnan(preferences)] = nan
                rows, cols = self._opti_mask.solve(mask)
                indices = sorted(set(rows) & set(cols))
                if len(indices) <= 1:
                    break
                start = min(indices)
                end = max(indices) + 1
            if end - start <= 1:
                break
            sub_outputs = outputs[start:end]
            sub_preferences = self.axiom.preferences(input, sub_outputs)

            with dbm_open(self.cache_path, flag="c") as cache:
                for i1, output1 in enumerate(sub_outputs):
                    for i2, output2 in enumerate(sub_outputs):
                        key = repr((input, output1, output2)).encode(encoding="utf-8")
                        preference = float(sub_preferences[i1, i2])
                        if isnan(preference):
                            raise RuntimeError("Missing preferences.")
                        preference_bytes = pack("f", preference)
                        cache[key] = preference_bytes

            preferences[start:end, start:end] = sub_preferences

        # Last, for the remaining pairs, fill them iteratively.
        unfilled_pairs = [
            (i1, i2)
            for i1 in range(len(outputs))
            for i2 in range(len(outputs))
            if isnan(preferences[i1, i2])
        ]
        if len(unfilled_pairs) == 0:
            return preferences

        unfilled_inouts_outputs = (
            (input, outputs[i1], outputs[i2]) for i1, i2 in unfilled_pairs
        )
        unfilled_preferences = self._iter_preferences(unfilled_inouts_outputs)
        for (i1, i2), preference in tqdm(
            zip(unfilled_pairs, unfilled_preferences),
            total=len(unfilled_pairs),
            desc="Cache preferences",
            unit="pair",
        ):
            preferences[i1, i2] = preference

        return preferences

    def cached(self, cache_path: Path) -> Axiom[_Input, _Output]:
        if self.cache_path == cache_path:
            return self
        else:
            return self.axiom.cached(cache_path)


CachedAxiom: TypeAlias = DbmCachedAxiom
