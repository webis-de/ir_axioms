from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from functools import cached_property
from itertools import product, compress
from logging import DEBUG
from pathlib import Path
from typing import Union, Optional, Set, Sequence, final, Callable

from ir_datasets import Dataset
from numpy import apply_along_axis, stack, ndarray, array
from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy
from tqdm.auto import tqdm

from ir_axioms import logger
from ir_axioms.axiom import AxiomLike, to_axiom, to_axioms
from ir_axioms.axiom.base import Axiom
from ir_axioms.backend.pyterrier import TerrierIndexContext, ContentsAccessor
from ir_axioms.backend.pyterrier.safe import TransformerBase
from ir_axioms.backend.pyterrier.transformer_utils import (
    require_columns, load_documents
)
from ir_axioms.backend.pyterrier.util import IndexRef, Index, Tokeniser
from ir_axioms.model import Query, RankedDocument, IndexContext
from ir_axioms.modules.ranking import PivotSelection, RandomPivotSelection


class PerGroupTransformer(TransformerBase, ABC):
    group_columns: Set[str] = NotImplemented
    optional_group_columns: Set[str] = {}
    verbose: bool = False
    description: Optional[str] = None
    unit: str = "it"

    @abstractmethod
    def transform_group(self, topics_or_res: DataFrame) -> DataFrame:
        pass

    def _all_group_columns(self, topics_or_res: DataFrame) -> Set[str]:
        return self.group_columns | {
            column for column in self.optional_group_columns
            if column in topics_or_res.columns
        }

    @final
    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        require_columns(topics_or_res, self.group_columns)

        query_rankings: DataFrameGroupBy = topics_or_res.groupby(
            by=list(self._all_group_columns(topics_or_res)),
            as_index=False,
            sort=False,
        )
        if self.verbose:
            # Show progress during reranking queries.
            tqdm.pandas(
                desc=self.description,
                unit=self.unit,
            )
            query_rankings = query_rankings.progress_apply(
                self.transform_group
            )
        else:
            query_rankings = query_rankings.apply(self.transform_group)
        return query_rankings.reset_index(drop=True)


class AxiomTransformer(PerGroupTransformer, ABC):
    index: Union[Path, IndexRef, Index] = NotImplemented
    dataset: Optional[Union[Dataset, str]] = None
    contents_accessor: Optional[ContentsAccessor] = "text"
    tokeniser: Optional[Tokeniser] = None
    cache_dir: Optional[Path] = None
    verbose: bool = False
    description: Optional[str] = None

    group_columns = {"query"}
    optional_group_columns = {"qid", "name"}
    unit = "query"

    @cached_property
    def _context(self) -> IndexContext:
        return TerrierIndexContext(
            index_location=self.index,
            dataset=self.dataset,
            contents_accessor=self.contents_accessor,
            tokeniser=self.tokeniser,
            cache_dir=self.cache_dir,
        )

    @final
    def transform_group(self, topics_or_res: DataFrame) -> DataFrame:
        require_columns(topics_or_res, {"query", "docno", "rank", "score"})

        if len(topics_or_res.index) == 0:
            # Empty ranking, skip reranking.
            return topics_or_res

        # Convert query.
        # As we grouped per query, we don't expect multiple queries here.
        assert topics_or_res["query"].nunique() <= 1
        query = Query(topics_or_res.iloc[0]["query"])

        # Load document list.
        documents = load_documents(topics_or_res)

        return self.transform_query_ranking(query, documents, topics_or_res)

    @abstractmethod
    def transform_query_ranking(
            self,
            query: Query,
            documents: Sequence[RankedDocument],
            topics_or_res: DataFrame,
    ) -> DataFrame:
        pass


@dataclass(frozen=True)
class AxiomaticReranker(AxiomTransformer):
    name = "AxiomaticReranker"
    description = "Reranking query axiomatically"

    axiom: AxiomLike
    index: Union[Path, IndexRef, Index]
    dataset: Optional[Union[Dataset, str]] = None
    contents_accessor: Optional[ContentsAccessor] = "text"
    pivot_selection: PivotSelection = RandomPivotSelection(),
    tokeniser: Optional[Tokeniser] = None
    cache_dir: Optional[Path] = None
    verbose: bool = False

    @cached_property
    def _axiom(self) -> Axiom:
        return to_axiom(self.axiom)

    def transform_query_ranking(
            self,
            query: Query,
            documents: Sequence[RankedDocument],
            topics_or_res: DataFrame,
    ) -> DataFrame:
        # Rerank documents.
        reranked_documents = self._axiom.rerank_kwiksort(
            self._context, query, documents, self.pivot_selection
        )

        # Convert reranked documents back to data frame.
        reranked = DataFrame({
            "docno": [doc.id for doc in reranked_documents],
            "rank": [doc.rank for doc in reranked_documents],
            "score": [doc.score for doc in reranked_documents],
        })

        # Remove original scores and ranks.
        original_ranking = topics_or_res.copy()
        del original_ranking["rank"]
        del original_ranking["score"]

        # Merge with new scores.
        reranked = reranked.merge(original_ranking, on="docno")
        return reranked


@dataclass(frozen=True)
class AggregatedAxiomaticPreference(AxiomTransformer):
    name = "AggregatedAxiomaticPreference"
    description = "Aggregating query axiom preferences"

    axioms: Sequence[AxiomLike]
    index: Union[Path, IndexRef, Index]
    aggregations: Sequence[Callable[[Sequence[float]], float]] = field(
        default_factory=lambda: [sum]
    )
    dataset: Optional[Union[Dataset, str]] = None
    contents_accessor: Optional[ContentsAccessor] = "text"
    filter_pairs: Optional[Callable[
        [RankedDocument, RankedDocument],
        bool
    ]] = None
    tokeniser: Optional[Tokeniser] = None
    cache_dir: Optional[Path] = None
    verbose: bool = False

    @cached_property
    def _axioms(self) -> Sequence[Axiom]:
        return to_axioms(self.axioms)

    def transform_query_ranking(
            self,
            query: Query,
            documents: Sequence[RankedDocument],
            topics_or_res: DataFrame,
    ) -> DataFrame:
        axioms = self._axioms
        context = self._context
        aggregations = self.aggregations
        filter_pairs = self.filter_pairs

        # Shape: |documents| x |documents| x |axioms|
        features: ndarray = stack(tuple(
            # Shape: |documents| x |documents|
            axiom.preference_matrix(
                context,
                query,
                documents,
                filter_pairs,
            )
            for axiom in axioms
        ), -1)

        # Shape: |documents| x |axioms| x |aggregations|
        features = stack(tuple(
            # Shape: |documents| x |axioms|
            apply_along_axis(
                lambda preferences: aggregation(preferences.tolist()),
                0,
                features
            )
            for aggregation in aggregations
        ), -1)

        # Shape: |documents| x (|aggregations| * |axioms|)
        features = features.reshape((features.shape[0], -1))

        topics_or_res["features"] = list(map(array, features))
        return topics_or_res


@dataclass(frozen=True)
class AxiomaticPreferences(AxiomTransformer):
    name = "AxiomaticPreferences"
    description = "Computing query axiom preferences"

    axioms: Sequence[AxiomLike]
    index: Union[Path, IndexRef, Index]
    axiom_names: Optional[Sequence[str]] = None
    dataset: Optional[Union[Dataset, str]] = None
    contents_accessor: Optional[ContentsAccessor] = "text"
    filter_pairs: Optional[Callable[
        [RankedDocument, RankedDocument],
        bool
    ]] = None
    tokeniser: Optional[Tokeniser] = None
    cache_dir: Optional[Path] = None
    verbose: bool = False

    @cached_property
    def _axioms(self) -> Sequence[Axiom]:
        return to_axioms(self.axioms)

    def transform_query_ranking(
            self,
            query: Query,
            documents: Sequence[RankedDocument],
            results: DataFrame,
    ) -> DataFrame:
        context = self._context
        axioms = self._axioms
        filter_pairs = self.filter_pairs

        # Result cross product.
        results_pairs = results.merge(
            results,
            on=list(self._all_group_columns(results)),
            suffixes=("_a", "_b"),
        )

        # Document pairs.
        document_pairs = list(product(documents, documents))

        # Filter document pairs.
        if filter_pairs is not None:
            filter_mask = [
                filter_pairs(document1, document2)
                for document1, document2 in document_pairs
            ]
            results_pairs = results_pairs.loc[filter_mask]
            document_pairs = list(compress(document_pairs, filter_mask))

        # Compute axiom preferences.
        if self.verbose and 0 < logger.level <= DEBUG:
            axioms = tqdm(
                axioms,
                desc="Computing axiom preferences",
                unit="axiom",
            )

        names: Sequence[str]
        if (
                self.axiom_names is not None and
                len(self.axiom_names) == len(axioms)
        ):
            names = self.axiom_names
        else:
            names = [str(axiom) for axiom in axioms]

        columns = [f"{name}_preference" for name in names]

        for column, axiom in zip(columns, axioms):
            if self.verbose and 0 < logger.level <= DEBUG:
                # Very verbose progress bars.
                document_pairs = tqdm(
                    document_pairs,
                    desc="Computing axiom preference",
                    unit="pair",
                )
            results_pairs[column] = [
                axiom.preference(context, query, document1, document2)
                for document1, document2 in document_pairs
            ]

        return results_pairs
