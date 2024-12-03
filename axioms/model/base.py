from typing import TypeAlias, TypeVar

from numpy import floating, bool_
from numpy.typing import NDArray

Input = TypeVar("Input")
Output = TypeVar("Output")

Preference: TypeAlias = float
PreferenceMatrix: TypeAlias = NDArray[floating]

Mask: TypeAlias = bool
MaskMatrix: TypeAlias = NDArray[bool_]
