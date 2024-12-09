from typing import TypeAlias, TypeVar

from numpy import floating, bool_
from numpy.typing import NDArray

Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", contravariant=True)

Preference: TypeAlias = float
PreferenceMatrix: TypeAlias = NDArray[floating]

Mask: TypeAlias = bool
MaskMatrix: TypeAlias = NDArray[bool_]
