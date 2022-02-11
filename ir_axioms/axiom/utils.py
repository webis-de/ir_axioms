def strictly_greater(x, y):
    if x > y:
        return 1
    elif y > x:
        return -1
    return 0


def strictly_less(x, y):
    if y > x:
        return 1
    elif x > y:
        return -1
    return 0


def approximately_equal(*args, margin_fraction: float = 0.1):
    """
    True if all numeric args are
    within (100 * margin_fraction)% of the largest.
    """

    abs_max = max(args, key=lambda item: abs(item))
    if abs_max == 0:
        # All values are 0.
        return True

    boundaries = (
        abs_max * (1 + margin_fraction),
        abs_max * (1 - margin_fraction),
    )
    boundary_min = min(boundaries)
    boundary_max = max(boundaries)

    return all(boundary_min < item < boundary_max for item in args)
