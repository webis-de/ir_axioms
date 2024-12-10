from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from sqlite3 import connect
from typing import Hashable, Iterable, Iterator, Sequence, Tuple, TypeVar

from numpy import array, float_
from more_itertools import chunked

from axioms.axiom.base import Axiom
from axioms.model import Preference, PreferenceMatrix


HashableInput = TypeVar("HashableInput", bound=Hashable)
HashableOutput = TypeVar("HashableOutput", bound=Hashable)


@dataclass(frozen=True, kw_only=True)
class CachedAxiom(Axiom[HashableInput, HashableOutput]):
    axiom: Axiom[HashableInput, HashableOutput]
    cache_path: Path

    @cached_property
    def _cache_file_path(self) -> Path:
        self.cache_path.mkdir(parents=True, exist_ok=True)
        cache_file_path = self.cache_path / "cache.sqlite"
        if not cache_file_path.exists():
            connection = connect(cache_file_path)
            connection.execute(
                "CREATE TABLE preferences(hash INTEGER NOT NULL PRIMARY KEY, preference)"
            )
            connection.commit()
            connection.close()
        return cache_file_path

    def _preferences_batch(
        self,
        inputs_outputs: Sequence[Tuple[HashableInput, HashableOutput, HashableOutput]],
    ) -> Sequence[Preference]:
        inputs_outputs_dict = {
            hash((input, output1, output2)): (input, output1, output2)
            for input, output1, output2 in inputs_outputs
        }

        connection = connect(self._cache_file_path)

        result = connection.execute(
            f"SELECT hash, preference FROM preferences WHERE hash IN ({','.join('?' * len(inputs_outputs_dict))})",  # nosec: B608
            list(inputs_outputs_dict.keys()),
        )
        found_preferences: Iterable[tuple[int, float]] = result.fetchall()

        existing_preferences = {
            hash: preference for hash, preference in found_preferences
        }

        new_preferences = {
            hash: self.axiom.preference(
                input=input,
                output1=output1,
                output2=output2,
            )
            for hash, (input, output1, output2) in inputs_outputs_dict.items()
            if hash not in existing_preferences.keys()
        }
        connection.executemany(
            "INSERT INTO preferences(hash, preference) VALUES(?, ?)",
            list(new_preferences.items()),
        )
        connection.commit()
        connection.close()

        all_preferences = {
            **existing_preferences,
            **new_preferences,
        }
        return [
            all_preferences[hash((input, output1, output2))]
            for input, output1, output2 in inputs_outputs
        ]

    def _iter_preferences(
        self,
        inputs_outputs: Iterable[Tuple[HashableInput, HashableOutput, HashableOutput]],
    ) -> Iterator[Preference]:
        for chunk in chunked(inputs_outputs, 100):
            yield from self._preferences_batch(chunk)

    def preference(
        self,
        input: HashableInput,
        output1: HashableOutput,
        output2: HashableOutput,
    ) -> Preference:
        return self._preferences_batch(
            inputs_outputs=[(input, output1, output2)],
        )[0]

    def preferences(
        self,
        input: HashableInput,
        outputs: Sequence[HashableOutput],
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

    def cached(self, cache_path: Path) -> Axiom[HashableInput, HashableOutput]:
        if self.cache_path == cache_path:
            return self
        else:
            return self.axiom.cached(cache_path)
