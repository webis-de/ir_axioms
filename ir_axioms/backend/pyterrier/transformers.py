from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import cached_property
from itertools import product
from logging import DEBUG
from pathlib import Path
from typing import Union, Optional, Set, Sequence, final, Callable, Iterable

from ir_datasets import Dataset
from numpy import array
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy
from tqdm.auto import tqdm

from ir_axioms import logger
from ir_axioms.axiom import AxiomLike, to_axiom, to_axioms
from ir_axioms.axiom.base import Axiom
from ir_axioms.backend.pyterrier import TerrierIndexContext, ContentsAccessor
from ir_axioms.backend.pyterrier.safe import TransformerBase
from ir_axioms.backend.pyterrier.transformer_utils import require_columns
from ir_axioms.backend.pyterrier.util import IndexRef, Index, Tokeniser
from ir_axioms.model import (
    Query, RankedDocument, RankedTextDocument, IndexContext,
    JudgedRankedDocument, JudgedRankedTextDocument
)


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

    def _load_documents(
            self,
            columns: Sequence[str],
            rows: Iterable[Series],
    ) -> Sequence[RankedDocument]:
        if "label" in columns:
            if (
                    self.contents_accessor is not None and
                    isinstance(self.contents_accessor, str) and
                    self.contents_accessor in columns
            ):
                # Load contents from dataframe column.
                return [
                    JudgedRankedTextDocument(
                        id=str(row["docno"]),
                        contents=str(row[self.contents_accessor]),
                        score=float(row["score"]),
                        rank=int(row["rank"]),
                        relevance=int(row["label"]),
                    )
                    for row in rows
                ]
            else:
                return [
                    JudgedRankedDocument(
                        id=str(row["docno"]),
                        score=float(row["score"]),
                        rank=int(row["rank"]),
                        relevance=int(row["label"]),
                    )
                    for row in rows
                ]
        else:
            if (
                    self.contents_accessor is not None and
                    isinstance(self.contents_accessor, str) and
                    self.contents_accessor in columns
            ):
                # Load contents from dataframe column.
                return [
                    RankedTextDocument(
                        id=str(row["docno"]),
                        contents=str(row[self.contents_accessor]),
                        score=float(row["score"]),
                        rank=int(row["rank"]),
                    )
                    for row in rows
                ]
            else:
                return [
                    RankedDocument(
                        id=str(row["docno"]),
                        score=float(row["score"]),
                        rank=int(row["rank"]),
                    )
                    for row in rows
                ]

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
        documents = self._load_documents(
            topics_or_res.columns,
            (row for _, row in topics_or_res.iterrows())
        )

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
        reranked_documents = self._axiom.rerank(
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
class AggregatedAxiomaticPreference(AxiomTransformer):
    name = "AggregatedAxiomaticPreference"
    description = "Aggregating query axiom preferences"

    axioms: Sequence[AxiomLike]
    index: Union[Path, IndexRef, Index]
    aggregation: Callable[[Sequence[float]], float] = sum
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
        aggregation = self.aggregation
        filter_pairs = self.filter_pairs

        aggregated_preferences = [
            axiom.aggregated_preference(
                context,
                query,
                documents,
                aggregation,
                filter_pairs,
            )
            for axiom in axioms
        ]

        transposed = list(map(array, zip(*aggregated_preferences)))

        features = topics_or_res
        features["features"] = transposed
        return features


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
            topics_or_res: DataFrame,
    ) -> DataFrame:
        context = self._context
        axioms = self._axioms
        filter_pairs = self.filter_pairs

        # Cross product.
        document_pairs = list(product(documents, documents))
        pairs = topics_or_res.merge(
            topics_or_res,
            on=list(self._all_group_columns(topics_or_res)),
            suffixes=("_a", "_b"),
        )

        filter_mask = [
            filter_pairs is None or filter_pairs(document1, document2)
            for document1, document2 in document_pairs
        ]
        document_pairs = [
            document
            for include, document in zip(filter_mask, document_pairs)
            if include
        ]
        pairs = pairs.where(filter_mask)

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

        for name, axiom in zip(names, axioms):
            if self.verbose and 0 < logger.level <= DEBUG:
                # Very verbose progress bars.
                document_pairs = tqdm(
                    document_pairs,
                    desc="Computing axiom preference",
                    unit="pair",
                )
            pairs[f"{name}_preference"] = [
                axiom.preference(context, query, document1, document2)
                for document1, document2 in document_pairs
            ]

        return pairs
