from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Union, Optional, List, Set, Sequence, final, Callable

from ir_datasets import Dataset
from numpy import array
from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy
from tqdm.auto import tqdm

from ir_axioms.axiom import AxiomLike, to_axiom
from ir_axioms.axiom.base import Axiom
from ir_axioms.backend.pyterrier import TerrierIndexContext, ContentsAccessor
from ir_axioms.backend.pyterrier.safe import TransformerBase
from ir_axioms.backend.pyterrier.transformer_utils import _require_columns
from ir_axioms.backend.pyterrier.util import IndexRef, Index, Tokeniser
from ir_axioms.model import (
    Query, RankedDocument, RankedTextDocument, IndexContext
)


class PerGroupTransformer(TransformerBase, ABC):
    group_columns: Set[str] = NotImplemented
    optional_group_columns: Set[str] = {}
    verbose: bool = False
    description: Optional[str] = None
    unit: Optional[str] = None

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
        _require_columns(self, topics_or_res, self.group_columns)

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

    @property
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
        _require_columns(
            self,
            topics_or_res,
            {"query", "docno", "rank", "score"}
        )

        if len(topics_or_res.index) == 0:
            # Empty ranking, skip reranking.
            return topics_or_res

        # Convert query.
        # As we grouped per query, we don't expect multiple queries here.
        assert topics_or_res["query"].nunique() <= 1
        query = Query(topics_or_res.iloc[0]["query"])

        # Load document list.
        documents: List[RankedDocument]
        if (
                self.contents_accessor is not None and
                isinstance(self.contents_accessor, str) and
                self.contents_accessor in topics_or_res.columns
        ):
            # Load contents from dataframe column.
            documents = [
                RankedTextDocument(
                    id=row["docno"],
                    contents=row[self.contents_accessor],
                    score=row["score"],
                    rank=row["rank"],
                )
                for index, row in topics_or_res.iterrows()
            ]
        else:
            documents = [
                RankedDocument(
                    id=row["docno"],
                    score=row["score"],
                    rank=row["rank"],
                )
                for index, row in topics_or_res.iterrows()
            ]

        return self.transform_query_ranking(query, documents, topics_or_res)

    @abstractmethod
    def transform_query_ranking(
            self,
            query: Query,
            documents: List[RankedDocument],
            topics_or_res: DataFrame,
    ) -> DataFrame:
        pass


class SingleAxiomTransformer(AxiomTransformer, ABC):
    axiom: AxiomLike
    index: Union[Path, IndexRef, Index]
    dataset: Optional[Union[Dataset, str]] = None
    contents_accessor: Optional[ContentsAccessor] = "text"
    tokeniser: Optional[Tokeniser] = None
    cache_dir: Optional[Path] = None
    verbose: bool = False
    description: Optional[str] = None

    @cached_property
    def _axiom(self) -> Axiom:
        return to_axiom(self.axiom)


class MultiAxiomTransformer(AxiomTransformer, ABC):
    axioms: Sequence[AxiomLike]
    index: Union[Path, IndexRef, Index]
    dataset: Optional[Union[Dataset, str]] = None
    contents_accessor: Optional[ContentsAccessor] = "text"
    tokeniser: Optional[Tokeniser] = None
    cache_dir: Optional[Path] = None
    verbose: bool = False
    description: Optional[str] = None

    @cached_property
    def _axioms(self) -> Sequence[Axiom]:
        return [to_axiom(axiom) for axiom in self.axioms]


@dataclass(frozen=True)
class AxiomaticReranker(SingleAxiomTransformer):
    name = "AxiomaticReranker"
    description = "Reranking query axiomatically"

    axiom: AxiomLike
    index: Union[Path, IndexRef, Index]
    dataset: Optional[Union[Dataset, str]] = None
    contents_accessor: Optional[ContentsAccessor] = "text"
    tokeniser: Optional[Tokeniser] = None
    cache_dir: Optional[Path] = None
    verbose: bool = False

    def transform_query_ranking(
            self,
            query: Query,
            documents: List[RankedDocument],
            topics_or_res: DataFrame,
    ) -> DataFrame:
        # Rerank documents.
        reranked_documents = self.axiom.rerank(
            self._context, query, documents
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
class AggregatedAxiomaticPreference(MultiAxiomTransformer):
    name = "AggregatedAxiomaticPreference"
    description = "Aggregating query axiom preferences"

    axioms: Sequence[AxiomLike]
    index: Union[Path, IndexRef, Index]
    aggregation: Callable[[List[float]], float] = sum
    dataset: Optional[Union[Dataset, str]] = None
    contents_accessor: Optional[ContentsAccessor] = "text"
    tokeniser: Optional[Tokeniser] = None
    cache_dir: Optional[Path] = None
    verbose: bool = False

    def transform_query_ranking(
            self,
            query: Query,
            documents: List[RankedDocument],
            topics_or_res: DataFrame,
    ) -> DataFrame:
        axioms = self.axioms
        context = self._context
        aggregation = self.aggregation

        aggregated_preferences = [
            axiom.aggregated_preference(
                context,
                query,
                documents,
                aggregation
            )
            for axiom in axioms
        ]

        transposed = list(map(array, zip(*aggregated_preferences)))

        features = topics_or_res
        features["features"] = transposed
        return features


@dataclass(frozen=True)
class AxiomaticPreferences(MultiAxiomTransformer):
    name = "AxiomaticPreferences"
    description = "Computing query axiom preferences"

    axioms: Sequence[AxiomLike]
    index: Union[Path, IndexRef, Index]
    dataset: Optional[Union[Dataset, str]] = None
    contents_accessor: Optional[ContentsAccessor] = "text"
    tokeniser: Optional[Tokeniser] = None
    cache_dir: Optional[Path] = None
    verbose: bool = False

    def transform_query_ranking(
            self,
            query: Query,
            documents: List[RankedDocument],
            topics_or_res: DataFrame,
    ) -> DataFrame:
        # Cross product.
        pairs = topics_or_res.merge(
            topics_or_res,
            on=list(self._all_group_columns(topics_or_res)),
            suffixes=("_a", "_b"),
        )

        # Compute axiom preferences.
        context = self._context
        axioms = self.axioms
        if self.verbose:
            axioms = tqdm(
                axioms,
                desc="Computing axiom preferences",
                unit="axiom",
            )
        for axiom in axioms:
            pairs[f"{axiom.name}_preference"] = [
                axiom.cached().preference(context, query, document1, document2)
                for document1 in documents
                for document2 in documents
            ]

        return pairs
