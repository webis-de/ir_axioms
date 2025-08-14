from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence, Any

from numpy import full

from ir_axioms.axiom import Axiom
from ir_axioms.model import Preference, PreferenceMatrix


@dataclass(kw_only=True)
class _MutableUniformAxiom(Axiom[Any, Any]):
    scalar: float

    def preference(
        self,
        input: Any,
        output1: Any,
        output2: Any,
    ) -> Preference:
        return self.scalar

    def preferences(
        self,
        input: Any,
        outputs: Sequence[Any],
    ) -> PreferenceMatrix:
        return full((len(outputs), len(outputs)), self.scalar)


def test_cache() -> Any:
    input = "i1"
    output1 = "o1"
    output2 = "o2"
    output3 = "o3"
    output4 = "o4"

    axiom = _MutableUniformAxiom(scalar=1)

    assert axiom.preference(input, output1, output2) == 1
    assert axiom.preference(input, output2, output1) == 1
    assert (axiom.preferences(input, [output1, output2]) == full((2, 2), 1)).all()

    axiom.scalar = 2

    assert axiom.preference(input, output1, output2) == 2
    assert axiom.preference(input, output2, output1) == 2
    assert (axiom.preferences(input, [output1, output2]) == full((2, 2), 2)).all()

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "cache"

        cached_axiom = axiom.cached(tmp_path)

        assert cached_axiom.preference(input, output1, output2) == 2
        assert cached_axiom.preference(input, output2, output1) == 2
        assert (
            cached_axiom.preferences(input, [output1, output2]) == full((2, 2), 2)
        ).all()

        axiom.scalar = 3

        # The cached preferences should still show the old scalar value.
        assert cached_axiom.preference(input, output1, output2) == 2
        assert cached_axiom.preference(input, output2, output1) == 2
        assert (
            cached_axiom.preferences(input, [output1, output2]) == full((2, 2), 2)
        ).all()

        # Newly computed preferences should pick up the new scalar value.
        assert cached_axiom.preference(input, output3, output4) == 3
        assert cached_axiom.preference(input, output4, output3) == 3
        assert (
            cached_axiom.preferences(input, [output3, output4]) == full((2, 2), 3)
        ).all()
