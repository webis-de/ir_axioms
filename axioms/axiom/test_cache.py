from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from numpy import full

from axioms.axiom import Axiom
from axioms.model import Input, Output, Preference, PreferenceMatrix


@dataclass(kw_only=True)
class _MutableUniformAxiom(Axiom[Input, Output]):
    scalar: float

    def preference(
        self,
        input: Input,
        output1: Output,
        output2: Output,
    ) -> Preference:
        return self.scalar

    def preferences(
        self,
        input: Input,
        outputs: Sequence[Output],
    ) -> PreferenceMatrix:
        return full((len(outputs), len(outputs)), self.scalar)


def test_cache():
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
        tmp_path = Path(tmp_dir)

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
