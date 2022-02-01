from abc import abstractmethod, ABC
from pathlib import Path
from typing import (
    Union, Optional, List, Set, Callable, NamedTuple, Sequence, final
)

from ir_datasets import Dataset
from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy
from tqdm import tqdm

from ir_axioms.axiom import Axiom, AxiomLike, to_axiom
from ir_axioms.backend.pyterrier import IndexRerankingContext
from ir_axioms.backend.pyterrier.safe import TransformerBase
from ir_axioms.backend.pyterrier.transformer_utils import _require_columns
from ir_axioms.backend.pyterrier.util import (
    IndexRef, Index, Tokeniser, EnglishTokeniser
)
from ir_axioms.model import Query, RankedDocument, RankedTextDocument
from ir_axioms.model.context import RerankingContext


class PerGroupTransformer(TransformerBase, ABC):
    _group_columns: Set[str]
    _optional_group_columns: Set[str]
    _verbose: bool = False
    _description: Optional[str] = None
    _unit: Optional[str] = None

    def __init__(
            self,
            group_columns: Set[str],
            optional_group_columns: Set[str] = None,
            verbose: bool = False,
            description: Optional[str] = None,
            unit: Optional[str] = None,
    ) -> None:
        self._group_columns = group_columns
        self._optional_group_columns = (
            optional_group_columns
            if optional_group_columns is not None
            else {}
        )
        self._verbose = verbose
        self._description = description
        self._unit = unit

    @abstractmethod
    def transform_group(self, topics_or_res: DataFrame) -> DataFrame:
        pass

    def _all_group_columns(self, topics_or_res: DataFrame) -> Set[str]:
        return self._group_columns | {
            column for column in self._optional_group_columns
            if column in topics_or_res.columns
        }

    @final
    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        _require_columns(self, topics_or_res, self._group_columns)

        query_rankings: DataFrameGroupBy = topics_or_res.groupby(
            by=list(self._all_group_columns(topics_or_res)),
            as_index=False,
            sort=False,
        )
        if self._verbose:
            # Show progress during reranking queries.
            tqdm.pandas(
                desc=(
                    self._description
                    if self._description is not None
                    else self.name
                ),
                unit=self._unit,
            )
            query_rankings = query_rankings.progress_apply(
                self.transform_group
            )
        else:
            query_rankings = query_rankings.apply(self.transform_group)
        return query_rankings.reset_index(drop=True)


class AxiomTransformer(PerGroupTransformer, ABC):
    _context: RerankingContext
    _contents_accessor: Optional[
        Union[str, Callable[[NamedTuple], str]]
    ]

    def __init__(
            self,
            index: Union[Path, IndexRef, Index],
            dataset: Optional[Union[Dataset, str]] = None,
            contents_accessor: Optional[Union[
                str,
                Callable[[NamedTuple], str]
            ]] = "text",
            tokeniser: Tokeniser = EnglishTokeniser(),
            cache_dir: Optional[Path] = None,
            verbose: bool = False,
            description: Optional[str] = None,
    ):
        super().__init__(
            group_columns={"query"},
            optional_group_columns={"qid", "name"},
            verbose=verbose,
            description=description,
            unit="query"
        )

        self._context = IndexRerankingContext(
            index_location=index,
            dataset=dataset,
            contents_accessor=contents_accessor,
            tokeniser=tokeniser,
            cache_dir=cache_dir,
        )

        self._contents_accessor = contents_accessor

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
                self._contents_accessor is not None and
                isinstance(self._contents_accessor, str) and
                self._contents_accessor in topics_or_res.columns
        ):
            # Load contents from dataframe column.
            documents = [
                RankedTextDocument(
                    id=row["docno"],
                    contents=row[self._contents_accessor],
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
    axiom: Axiom

    def __init__(
            self,
            axiom: AxiomLike,
            index: Union[Path, IndexRef, Index],
            dataset: Optional[Union[Dataset, str]] = None,
            contents_accessor: Optional[Union[
                str,
                Callable[[NamedTuple], str]
            ]] = "text",
            tokeniser: Tokeniser = EnglishTokeniser(),
            cache_dir: Optional[Path] = None,
            verbose: bool = False,
            description: Optional[str] = None,
    ):
        super().__init__(
            index=index,
            dataset=dataset,
            contents_accessor=contents_accessor,
            tokeniser=tokeniser,
            cache_dir=cache_dir,
            verbose=verbose,
            description=description,
        )
        self.axiom = to_axiom(axiom)


class MultiAxiomTransformer(AxiomTransformer, ABC):
    axioms: Sequence[Axiom]

    def __init__(
            self,
            axioms: Sequence[AxiomLike],
            index: Union[Path, IndexRef, Index],
            dataset: Optional[Union[Dataset, str]] = None,
            contents_accessor: Optional[Union[
                str,
                Callable[[NamedTuple], str]
            ]] = "text",
            tokeniser: Tokeniser = EnglishTokeniser(),
            cache_dir: Optional[Path] = None,
            verbose: bool = False,
            description: Optional[str] = None,
    ):
        super().__init__(
            index=index,
            dataset=dataset,
            contents_accessor=contents_accessor,
            tokeniser=tokeniser,
            cache_dir=cache_dir,
            verbose=verbose,
            description=description,
        )
        self.axioms = [to_axiom(axiom) for axiom in axioms]


class AxiomaticReranker(SingleAxiomTransformer):
    name = "AxiomaticReranker"

    def transform_query_ranking(
            self,
            query: Query,
            documents: List[RankedDocument],
            topics_or_res: DataFrame,
    ) -> DataFrame:
        # Rerank documents.
        reranked_documents = self.axiom.rerank(self._context, query, documents)

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


class AxiomaticPreferences(MultiAxiomTransformer):
    name = "AxiomaticPreferences"

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
        for axiom in self.axioms:
            pairs[f"{axiom.name}_preference"] = [
                axiom.preference(self._context, query, document1, document2)
                for document1 in documents
                for document2 in documents
            ]

        return pairs
