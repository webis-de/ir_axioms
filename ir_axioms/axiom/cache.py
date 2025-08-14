from dataclasses import dataclass
from dbm import open as dbm_open
from pathlib import Path
from struct import pack, unpack
from typing import Iterable, Iterator, Protocol, Sequence, Tuple, TypeVar

from numpy import array, float_, ndarray, isnan, nan
from typing_extensions import TypeAlias  # type: ignore

from ir_axioms.axiom.base import Axiom
from ir_axioms.model import Preference, PreferenceMatrix


class SupportsRepr(Protocol):
    def __repr__(self) -> str: ...


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
                        for output1 in outputs
                        for output2 in outputs
                    ),
                    only_cached=True,
                )
            ),
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))

    def preferences(
        self,
        input: _Input,
        outputs: Sequence[_Output],
    ) -> PreferenceMatrix:
        # First, only load the cached dependencies.
        preferences = self._cached_preferences(input, outputs)

        if not (isnan(preferences).any()):
            return preferences

        preferences = self.axiom.preferences(input, outputs)
        with dbm_open(self.cache_path, flag="c") as cache:
            for i1, output1 in enumerate(outputs):
                for i2, output2 in enumerate(outputs):
                    key = repr((input, output1, output2)).encode(encoding="utf-8")
                    preference = float(preferences[i1, i2])
                    if isnan(preference):
                        raise RuntimeError("Missing preferences.")
                    preference_bytes = pack("f", preference)
                    cache[key] = preference_bytes
        return preferences

    def cached(self, cache_path: Path) -> Axiom[_Input, _Output]:
        if self.cache_path == cache_path:
            return self
        else:
            return self.axiom.cached(cache_path)


CachedAxiom: TypeAlias = DbmCachedAxiom
