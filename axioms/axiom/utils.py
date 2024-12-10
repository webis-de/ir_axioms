from typing import Union, TypeVar, Protocol, Sized


_T_contra = TypeVar("_T_contra", contravariant=True)


class _SupportsComparison(Protocol[_T_contra]):
    def __lt__(self, other: _T_contra, /) -> bool: ...
    def __gt__(self, other: _T_contra, /) -> bool: ...


_SupportsComparisonT = TypeVar("_SupportsComparisonT", bound=_SupportsComparison)


def strictly_greater(x: _SupportsComparisonT, y: _SupportsComparisonT) -> float:
    if x > y:
        return 1
    elif x < y:
        return -1
    return 0


def strictly_less(x: _SupportsComparisonT, y: _SupportsComparisonT) -> float:
    if x < y:
        return 1
    elif x > y:
        return -1
    return 0


def strictly_more(x: Sized, y: Sized) -> float:
    return strictly_greater(len(x), len(y))


def strictly_fewer(x: Sized, y: Sized) -> float:
    return strictly_less(len(x), len(y))


def approximately_equal(
    *items: Union[int, float], margin_fraction: float = 0.1
) -> bool:
    """
    True if all numeric args are
    within (100 * margin_fraction)% of the largest.
    """
    if len(items) == 0:
        return True

    abs_max = max(items, key=lambda item: abs(item))
    if abs_max == 0:
        # All values are 0.
        return True

    boundaries = (
        abs_max * (1 + margin_fraction),
        abs_max * (1 - margin_fraction),
    )
    boundary_min = min(boundaries)
    boundary_max = max(boundaries)

    return all(boundary_min <= item <= boundary_max for item in items)
