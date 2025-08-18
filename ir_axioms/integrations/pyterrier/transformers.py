from typing import TYPE_CHECKING
from ir_axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed() or TYPE_CHECKING:
    from dataclasses import dataclass, field
    from functools import cached_property, reduce
    from itertools import product
    from operator import mul
    from statistics import mean
    from typing import (
        Optional,
        Sequence,
        Callable,
        Any,
        Mapping,
        Hashable,
    )

    from numpy import apply_along_axis, stack, ndarray
    from pandas import DataFrame, concat
    from pyterrier import Transformer
    from pyterrier.model import query_columns, add_ranks
    from tqdm.auto import tqdm

    from ir_axioms.axiom.base import Axiom
    from ir_axioms.integrations.pyterrier.utils import (
        require_columns,
        load_documents,
        load_query,
    )
    from ir_axioms.model import Query, Document
    from ir_axioms.tools import PivotSelection, RandomPivotSelection

    @dataclass(frozen=True, kw_only=True)
    class KwikSortReranker(Transformer):
        axiom: Axiom[Query, Document]
        pivot_selection: PivotSelection = RandomPivotSelection()
        text_field: Optional[str] = "text"
        verbose: bool = False

        def _transform_group(
            self, group_keys: Mapping[Hashable, Any], res: DataFrame
        ) -> DataFrame:
            # Convert query and documents.
            query = load_query(group_keys)
            documents = load_documents(res, text_column=self.text_field)

            # Rerank documents.
            documents = self.axiom.rerank_kwiksort(
                input=query,
                ranking=documents,
                pivot_selection=self.pivot_selection,
            )

            # Remove original scores and ranks.
            res.drop(
                columns=set(["rank", "score"]).intersection(res.columns), inplace=True
            )

            # Add re-ranked scores and ranks.
            ranks = DataFrame(
                [
                    {"docno": document.id, "score": -rank}
                    for rank, document in enumerate(documents)
                ]
            )
            res = res.merge(ranks, on="docno")
            res = add_ranks(res, single_query=True)
            res = res.sort_values(by="rank")

            return res

        def transform(self, inp: DataFrame) -> DataFrame:
            require_columns(inp, {"qid", "docno"})
            query_cols = list(query_columns(inp))
            query_rankings = inp.groupby(
                by=query_cols,
                group_keys=True,
                sort=False,
            )
            if len(query_rankings) == 0:
                return inp
            return concat(
                [
                    self._transform_group(
                        group_keys=dict(zip(query_cols, grouping)),
                        res=ranking,
                    )
                    for grouping, ranking in tqdm(
                        query_rankings,
                        desc="KwikSort re-rank",
                        unit="query",
                        disable=not self.verbose,
                    )
                ]
            )

    @dataclass(frozen=True)
    class AggregatedAxiomaticPreferences(Transformer):
        axioms: Sequence[Axiom[Query, Document]]
        aggregations: Sequence[Callable[[Sequence[float]], float]] = field(
            default_factory=lambda: [max, min, mean]
        )
        text_field: Optional[str] = "text"
        verbose: bool = False

        def _transform_group(
            self, group_keys: Mapping[Hashable, Any], res: DataFrame
        ) -> DataFrame:
            # Convert query and documents.
            query = load_query(group_keys)
            documents = load_documents(res, text_column=self.text_field)

            # Compute the axiomatic preference matrices.
            # Shape: |documents| x |documents| x |axioms|
            preferences: ndarray = stack(
                tuple(
                    # Shape: |documents| x |documents|
                    axiom.preferences(
                        input=query,
                        outputs=documents,
                    )
                    for axiom in self.axioms
                ),
                axis=-1,
            )

            # Aggregate the preferences.
            # Shape: |documents| x |axioms| x |aggregations|
            aggregated_preferences = stack(
                tuple(
                    # Shape: |documents| x |axioms|
                    apply_along_axis(
                        lambda preferences: aggregation(preferences.tolist()),
                        0,
                        preferences,
                    )
                    for aggregation in self.aggregations
                ),
                axis=-1,
            )

            # Flatten the result to have one row per document.
            # Shape: |documents| x (|aggregations| * |axioms|)
            features = list(
                aggregated_preferences.reshape((aggregated_preferences.shape[0], -1))
            )

            res["features"] = features
            return res

        def transform(self, inp: DataFrame) -> DataFrame:
            require_columns(inp, {"qid", "docno"})
            query_cols = list(query_columns(inp))
            query_rankings = inp.groupby(
                by=query_cols,
                group_keys=True,
                sort=False,
            )
            if len(query_rankings) == 0:
                inp["features"] = None
                return inp
            return concat(
                [
                    self._transform_group(
                        group_keys=dict(zip(query_cols, grouping)),
                        res=ranking,
                    )
                    for grouping, ranking in tqdm(
                        query_rankings,
                        desc="Aggregate axiom preferences",
                        unit="query",
                        disable=not self.verbose,
                    )
                ]
            )

    @dataclass(frozen=True)
    class AxiomaticPreferences(Transformer):
        axioms: Sequence[Axiom]
        axiom_names: Optional[Sequence[str]] = None
        text_field: Optional[str] = "text"
        verbose: bool = False

        @cached_property
        def _axiom_names(self) -> Sequence[str]:
            # Get axiom names.
            if self.axiom_names is not None:
                if len(self.axiom_names) != len(self.axioms):
                    raise ValueError("Number of axioms and names must match.")
                return self.axiom_names
            else:
                return [str(axiom) for axiom in self.axioms]

        def _transform_group(
            self, group_keys: Mapping[Hashable, Any], res: DataFrame
        ) -> DataFrame:
            # Convert query and documents.
            query = load_query(group_keys)
            documents = load_documents(res, text_column=self.text_field)

            # Result cross product.
            res = res.merge(
                res,
                on=list(query_columns(res)),
                suffixes=("_a", "_b"),
                sort=False,
            )

            # Compute the axiomatic preference matrices.
            # Shape: |axioms| x |documents| x |documents|
            preferences: ndarray = stack(
                tuple(
                    # Shape: |documents| x |documents|
                    axiom.preferences(
                        input=query,
                        outputs=documents,
                    )
                    for axiom in self.axioms
                ),
                axis=0,
            )

            # Flatten the result to have one row per document pair.
            # Shape: |axioms| x (|documents| * |documents|)
            preferences = preferences.reshape((preferences.shape[0], -1))

            # Sanity checks.
            if not len(self._axiom_names) == len(preferences):
                raise ValueError(
                    f"Number of axioms ({len(preferences)}) does not match number of names ({len(self._axiom_names)})."
                )
            if not reduce(mul, preferences.shape[1:]) == len(res):
                raise ValueError(
                    f"Number of document pairs ({len(res)}) does not match number of preferences ({product(preferences.shape[1:])})."
                )

            # Insert preferences into result data frame.
            for axiom_name, axiom_preferences in zip(self._axiom_names, preferences):
                res[f"{axiom_name}_preference"] = axiom_preferences

            return res

        def transform(self, inp: DataFrame) -> DataFrame:
            require_columns(inp, {"qid", "docno"})
            query_cols = list(query_columns(inp))
            query_rankings = inp.groupby(
                by=query_cols,
                group_keys=True,
                sort=False,
            )
            if len(query_rankings) == 0:
                inp = inp.merge(
                    inp,
                    on=query_cols,
                    suffixes=("_a", "_b"),
                    sort=False,
                )
                for axiom_name in self._axiom_names:
                    inp[f"{axiom_name}_preference"] = None
                return inp
            return concat(
                [
                    self._transform_group(
                        group_keys=dict(zip(query_cols, grouping)),
                        res=ranking,
                    )
                    for grouping, ranking in tqdm(
                        query_rankings,
                        desc="Compute axiom preferences",
                        unit="query",
                        disable=not self.verbose,
                    )
                ]
            )

else:
    KwikSortReranker = NotImplemented  # type: ignore
    AggregatedAxiomaticPreferences = NotImplemented  # type: ignore
    AxiomaticPreferences = NotImplemented  # type: ignore
