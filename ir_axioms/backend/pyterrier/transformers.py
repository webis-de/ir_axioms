from abc import abstractmethod, ABC
from pathlib import Path
from typing import Union, Optional, List, Set, Callable, NamedTuple, Sequence

from ir_datasets import Dataset
from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy
from tqdm import tqdm

from ir_axioms.axiom import Axiom, AxiomLike, to_axiom
from ir_axioms.backend.pyterrier import IndexRerankingContext
from ir_axioms.backend.pyterrier.safe import TransformerBase, Transformer
from ir_axioms.backend.pyterrier.util import (
    IndexRef, Index, Tokeniser, EnglishTokeniser
)
from ir_axioms.model import Query, RankedDocument, RankedTextDocument
from ir_axioms.model.context import RerankingContext


def _require_columns(
        transformer: Transformer,
        ranking: DataFrame,
        *expected_columns: str,
) -> None:
    expected_columns: Set[str] = set(expected_columns)
    columns: Set[str] = set(ranking.columns)
    missing_columns: Set[str] = expected_columns - columns
    if len(missing_columns) > 0:
        raise ValueError(
            f"{transformer.name} expected columns "
            f"{', '.join(expected_columns)} but got columns "
            f"{', '.join(columns)} (missing columns "
            f"{', '.join(missing_columns)})."
        )


def _apply_per_query_run(
        ranking: DataFrame,
        function: Callable[[DataFrame], DataFrame],
        desc: Optional[str] = None,
        verbose: bool = False,
) -> DataFrame:
    group_columns = ["qid", "query"]
    if "name" in ranking.columns:
        group_columns.append("name")
    query_rankings: DataFrameGroupBy = ranking.groupby(
        by=group_columns,
        as_index=False,
        sort=False,
    )
    if verbose:
        # Show progress during reranking queries.
        tqdm.pandas(
            desc=desc,
            unit="query",
        )
        query_rankings = query_rankings.progress_apply(function)
    else:
        query_rankings = query_rankings.apply(function)
    return query_rankings.reset_index(drop=True)


def _query(ranking: DataFrame) -> Query:
    # As we grouped per query, we don't expect multiple queries here.
    assert ranking["qid"].nunique() <= 1
    assert ranking["query"].nunique() <= 1

    # Convert to typed data classes.
    return Query(ranking.iloc[0]["query"])


def _documents(
        ranking: DataFrame,
        contents_accessor: Optional[Union[
            str,
            Callable[[NamedTuple], str]
        ]]
) -> List[RankedDocument]:
    if (
            contents_accessor is not None and
            isinstance(contents_accessor, str) and
            contents_accessor in ranking.columns
    ):
        return [
            RankedTextDocument(
                id=row["docno"],
                contents=row[contents_accessor],
                score=row["score"],
                rank=row["rank"],
            )
            for index, row in ranking.iterrows()
        ]
    else:
        return [
            RankedDocument(
                id=row["docno"],
                score=row["score"],
                rank=row["rank"],
            )
            for index, row in ranking.iterrows()
        ]


def _merge_scores(
        ranking: DataFrame,
        reranked: List[RankedDocument]
) -> DataFrame:
    # Convert reranked documents back to data frame.
    reranked_ranking = DataFrame({
        'docno': [doc.id for doc in reranked],
        'rank': [doc.rank for doc in reranked],
        'score': [doc.score for doc in reranked],
    })

    # Remove old scores and ranks.
    ranking = ranking.copy()
    del ranking["rank"]
    del ranking["score"]

    # Merge with new scores.
    reranked_ranking = reranked_ranking.merge(ranking, on="docno")
    return reranked_ranking


class AxiomTransformer(TransformerBase, ABC):
    reranking_context: RerankingContext
    contents_accessor: Optional[Union[str, Callable[[NamedTuple], str]]]
    verbose: bool

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
    ):
        # If specified, fetch document contents from ir_datasets.
        self.reranking_context = IndexRerankingContext(
            index_location=index,
            dataset=dataset,
            contents_accessor=contents_accessor,
            tokeniser=tokeniser,
            cache_dir=cache_dir,
        )

        self.contents_accessor = contents_accessor
        self.verbose = verbose

    description: Optional[str] = None

    @abstractmethod
    def transform_ranking(self, ranking: DataFrame) -> DataFrame:
        pass

    def transform(self, ranking: DataFrame) -> DataFrame:
        _require_columns(
            self, ranking,
            "qid", "query", "docno", "rank", "score"
        )
        return _apply_per_query_run(
            ranking,
            self.transform_ranking,
            desc=self.description,
            verbose=self.verbose
        )


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
    ):
        super().__init__(
            index=index,
            dataset=dataset,
            contents_accessor=contents_accessor,
            tokeniser=tokeniser,
            cache_dir=cache_dir,
            verbose=verbose,
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
    ):
        super().__init__(
            index=index,
            dataset=dataset,
            contents_accessor=contents_accessor,
            tokeniser=tokeniser,
            cache_dir=cache_dir,
            verbose=verbose,
        )
        self.axioms = [to_axiom(axiom) for axiom in axioms]


class AxiomaticReranker(SingleAxiomTransformer):
    name = "AxiomaticReranker"
    description = "Reranking with axiom preferences"

    def transform_ranking(self, ranking: DataFrame) -> DataFrame:
        if len(ranking.index) == 0:
            # Empty ranking, skip reranking.
            return ranking

        # Convert to typed data classes.
        query = _query(ranking)
        documents = _documents(ranking, self.contents_accessor)

        # Rerank documents.
        reranked_documents = self.axiom.rerank(
            self.reranking_context,
            query,
            documents,
        )

        # Merge reranked documents back to data frame.
        reranked = _merge_scores(ranking, reranked_documents)
        return reranked


class AxiomaticPreferences(MultiAxiomTransformer):
    name = "AxiomaticPreferences"
    description = "Compute axiomatic preferences"

    def _unroll_axiom_preferences(
            self,
            axiom: Axiom,
            query: Query,
            ranking: List[RankedDocument],
    ):
        return [
            axiom.preference(self.reranking_context, query, document1, document2)
            for document1 in ranking
            for document2 in ranking
        ]

    def transform_ranking(self, ranking: DataFrame):
        if len(ranking.index) == 0:
            # Empty ranking, skip feature scoring.
            return ranking

        # Convert to typed data classes.
        query = _query(ranking)
        documents = _documents(ranking, self.contents_accessor)

        # Cross product.
        product_columns = ["qid", "query"]
        if "name" in ranking.columns:
            product_columns.append("name")
        pairs = ranking.merge(
            ranking,
            on=product_columns,
            suffixes=("_a", "_b"),
        )

        # Compute axiom preferences.
        for axiom in self.axioms:
            pairs[f"{axiom.name}_preference"] = self._unroll_axiom_preferences(
                axiom,
                query,
                documents,
            )

        return pairs

