from typing import Set

from pandas import DataFrame

from ir_axioms.backend.pyterrier.safe import Transformer


def _require_columns(
        transformer: Transformer,
        ranking: DataFrame,
        expected_columns: Set[str],
) -> None:
    columns: Set[str] = set(ranking.columns)
    missing_columns: Set[str] = expected_columns - columns
    if len(missing_columns) > 0:
        raise ValueError(
            f"{transformer.name} expected columns "
            f"{', '.join(expected_columns)} but got columns "
            f"{', '.join(columns)} (missing columns "
            f"{', '.join(missing_columns)})."
        )
