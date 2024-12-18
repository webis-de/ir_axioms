from dataclasses import dataclass
from dbm import open as dbm_open
from functools import cached_property
from math import nan
from pathlib import Path
from sqlite3 import connect
from struct import pack, unpack
from typing import Iterable, Iterator, Protocol, Sequence, Tuple, TypeVar

from numpy import array, float_, ndarray, isnan, ones_like
from more_itertools import chunked
from optimask import OptiMask
from typing_extensions import TypeAlias  # type: ignore

from axioms.axiom.base import Axiom
from axioms.model import Preference, PreferenceMatrix


class SupportsRepr(Protocol):
    def __repr__(self) -> str:
        pass


_Input = TypeVar("_Input", bound=SupportsRepr)
_Output = TypeVar("_Output", bound=SupportsRepr)


@dataclass(frozen=True, kw_only=True)
class SqliteCachedAxiom(Axiom[_Input, _Output]):
    axiom: Axiom[_Input, _Output]
    cache_path: Path

    @cached_property
    def _cache_file_path(self) -> Path:
        if not self.cache_path.exists():
            connection = connect(self.cache_path)
            connection.execute(
                "CREATE TABLE preferences(key INTEGER NOT NULL PRIMARY KEY, preference)"
            )
            connection.commit()
            connection.close()
        return self.cache_path

    def _preferences_batch(
        self,
        inputs_outputs: Sequence[Tuple[_Input, _Output, _Output]],
    ) -> Sequence[Preference]:
        inputs_outputs_dict = {
            repr((input, output1, output2)): (input, output1, output2)
            for input, output1, output2 in inputs_outputs
        }

        connection = connect(self._cache_file_path)

        result = connection.execute(
            f"SELECT key, preference FROM preferences WHERE key IN ({','.join('?' * len(inputs_outputs_dict))})",  # nosec: B608
            list(inputs_outputs_dict.keys()),
        )
        found_preferences: Iterable[tuple[str, float]] = result.fetchall()

        existing_preferences = {
            key: preference for key, preference in found_preferences
        }

        new_preferences = {
            key: self.axiom.preference(
                input=input,
                output1=output1,
                output2=output2,
            )
            for key, (input, output1, output2) in inputs_outputs_dict.items()
            if key not in existing_preferences.keys()
        }
        connection.executemany(
            "INSERT INTO preferences(key, preference) VALUES(?, ?)",
            list(new_preferences.items()),
        )
        connection.commit()
        connection.close()

        all_preferences = {
            **existing_preferences,
            **new_preferences,
        }
        return [
            all_preferences[repr((input, output1, output2))]
            for input, output1, output2 in inputs_outputs
        ]

    def _iter_preferences(
        self,
        inputs_outputs: Iterable[Tuple[_Input, _Output, _Output]],
    ) -> Iterator[Preference]:
        self.cache_path.parent.mkdir(exist_ok=True, parents=True)
        for chunk in chunked(inputs_outputs, 100):
            yield from self._preferences_batch(chunk)

    def preference(
        self,
        input: _Input,
        output1: _Output,
        output2: _Output,
    ) -> Preference:
        return next(
            self._iter_preferences(
                inputs_outputs=[(input, output1, output2)],
            )
        )

    def preferences(
        self,
        input: _Input,
        outputs: Sequence[_Output],
    ) -> PreferenceMatrix:
        return array(
            list(
                self._iter_preferences(
                    (input, output1, output2)
                    for output2 in outputs
                    for output1 in outputs
                )
            ),
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))

    def cached(self, cache_path: Path) -> Axiom[_Input, _Output]:
        if self.cache_path == cache_path:
            return self
        else:
            return self.axiom.cached(cache_path)


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
            mask = ones_like(preferences)
            mask[~isnan(preferences)] = nan
            rows, cols = self._opti_mask.solve(mask)
            indices = sorted(set(rows) & set(cols))
            if len(indices) == 0:
                break
            start = min(indices)
            end = max(indices) + 1
            sub_outputs = outputs[start:end]
            sub_preferences = self.axiom.preferences(input, sub_outputs)

            with dbm_open(self.cache_path, flag="c") as cache:
                for i1, output1 in enumerate(sub_outputs):
                    for i2, output2 in enumerate(sub_outputs):
                        key = repr((input, output1, output2)).encode(encoding="utf-8")
                        preference_bytes = pack("f", sub_preferences[i1, i2])
                        cache[key] = preference_bytes

            preferences[start:end, start:end] = sub_preferences

        # Last, for the remaining pairs, fill them iteratively.
        unfilled_pairs = [
            (i1, i2)
            for i1 in range(len(outputs))
            for i2 in range(len(outputs))
            if isnan(preferences[i1,i2])
        ]
        unfilled_inouts_outputs = (
            (input, outputs[i1], outputs[i2])
            for i1, i2 in unfilled_pairs
        )
        unfilled_preferences = self._iter_preferences(unfilled_inouts_outputs)
        for (i1, i2), preference in zip(unfilled_pairs, unfilled_preferences):
            preferences[i1, i2] = preference

        return preferences

    def cached(self, cache_path: Path) -> Axiom[_Input, _Output]:
        if self.cache_path == cache_path:
            return self
        else:
            return self.axiom.cached(cache_path)


CachedAxiom: TypeAlias = DbmCachedAxiom
