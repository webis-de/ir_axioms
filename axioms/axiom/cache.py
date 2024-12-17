from dataclasses import dataclass
from dbm import open as dbm_open
from functools import cached_property
from pathlib import Path
from sqlite3 import connect
from struct import pack, unpack
from typing import Iterable, Iterator, Protocol, Sequence, Tuple, TypeVar

from numpy import array, float_
from more_itertools import chunked
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
        for chunk in chunked(inputs_outputs, 100):
            yield from self._preferences_batch(chunk)

    def preference(
        self,
        input: _Input,
        output1: _Output,
        output2: _Output,
    ) -> Preference:
        return self._preferences_batch(
            inputs_outputs=[(input, output1, output2)],
        )[0]

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
    ) -> Iterator[Preference]:
        with dbm_open(self.cache_path, flag="c") as cache:
            for input, output1, output2 in inputs_outputs:
                key = repr((input, output1, output2)).encode(encoding="utf-8")
                if key in cache:
                    preference_bytes = cache[key]
                    (preference,) = unpack("f", preference_bytes)
                    yield preference
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


CachedAxiom: TypeAlias = DbmCachedAxiom
