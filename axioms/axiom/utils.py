from typing import Union, overload


@overload
def strictly_greater(x: bool, y: bool) -> float: ...


@overload
def strictly_greater(x: int, y: int) -> float: ...


@overload
def strictly_greater(x: float, y: float) -> float: ...


def strictly_greater(x: int | float | bool, y: int | float | bool) -> float:
    if x > y:
        return 1
    elif y > x:
        return -1
    return 0


@overload
def strictly_less(x: bool, y: bool) -> float: ...


@overload
def strictly_less(x: int, y: int) -> float: ...


@overload
def strictly_less(x: float, y: float) -> float: ...


def strictly_less(x: int | float | bool, y: int | float | bool) -> float:
    if y > x:
        return 1
    elif x > y:
        return -1
    return 0


def approximately_equal(
        *items: Union[int, float],
        margin_fraction: float = 0.1
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

    return all(boundary_min < item < boundary_max for item in items)
