from abc import abstractmethod, ABC
from itertools import chain
from pathlib import Path
from typing import Union, Optional, List, Set, Callable, Iterable

from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy
from tqdm import tqdm

from ir_axioms.axiom import Axiom, AxiomLike, to_axiom
from ir_axioms.backend import PyTerrierBackendContext
from ir_axioms.backend.pyterrier import (
    IndexRerankingContext, EnglishTokeniser, Index
)
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext

with PyTerrierBackendContext():
    from pyterrier import IndexRef
    from pyterrier.index import Tokeniser
    from pyterrier.transformer import TransformerBase


    def _require_columns(
            transformer: TransformerBase,
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


    def _apply_per_query(
            ranking: DataFrame,
            function: Callable[[DataFrame], DataFrame],
            desc: Optional[str] = None,
    ) -> DataFrame:
        tqdm.pandas(
            desc=desc,
            unit=" topics",
        )  # Show progress during reranking queries.
        query_rankings: DataFrameGroupBy = ranking.groupby(
            by=["qid", "query"],
            as_index=False,
            sort=False,
        )
        query_rankings = query_rankings.progress_apply(function)
        return query_rankings.reset_index(drop=True)


    def _query(ranking: DataFrame) -> Query:
        # As we grouped per query, we don't expect multiple queries here.
        assert ranking["qid"].nunique() <= 1
        assert ranking["query"].nunique() <= 1

        # Convert to typed data classes.
        return Query(ranking.iloc[0]["query"])


    def _documents(ranking: DataFrame) -> List[RankedDocument]:
        return [
            RankedDocument(
                id=row["docno"],
                content=row["text"],
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


    class AxiomaticTransformerBase(TransformerBase):
        axiom: Axiom
        reranking_context: RerankingContext

        def __init__(
                self,
                axiom: AxiomLike,
                index_location: Union[Path, IndexRef, Index],
                tokeniser: Tokeniser = EnglishTokeniser(),
                cache_dir: Optional[Path] = None,
        ):
            self.axiom = to_axiom(axiom)
            self.reranking_context = IndexRerankingContext(
                index_location,
                tokeniser,
                cache_dir
            )


    class AxiomaticRankingTransformerBase(AxiomaticTransformerBase, ABC):
        description: Optional[str] = None

        @abstractmethod
        def transform_ranking(self, ranking: DataFrame) -> DataFrame:
            pass

        def transform(self, ranking: DataFrame) -> DataFrame:
            _require_columns(
                self, ranking,
                "qid", "query", "docid", "docno", "rank", "score", "text"
            )
            return _apply_per_query(
                ranking,
                self.transform_ranking,
                desc=self.description,
            )


    class AxiomaticReranker(AxiomaticRankingTransformerBase):
        name = "AxiomaticReranker"
        description = "Reranking with axiom preferences"

        def transform_ranking(self, ranking: DataFrame) -> DataFrame:
            if len(ranking.index) == 0:
                # Empty ranking, skip reranking.
                return ranking

            # Convert to typed data classes.
            query = _query(ranking)
            documents = _documents(ranking)

            # Rerank documents.
            reranked_documents = self.axiom.rerank(
                self.reranking_context,
                query,
                documents,
            )

            # Merge reranked documents back to data frame.
            reranked = _merge_scores(ranking, reranked_documents)
            return reranked


    class AxiomaticPreferences(AxiomaticRankingTransformerBase):
        name = "AxiomaticPreferences"
        description = "Compute axiomatic preferences"

        def transform_ranking(self, ranking: DataFrame):
            if len(ranking.index) == 0:
                # Empty ranking, skip feature scoring.
                return ranking

            # Convert to typed data classes.
            query = _query(ranking)
            documents = _documents(ranking)

            # Cross product.
            pairs = ranking.merge(
                ranking,
                on=["qid", "query"],
                suffixes=("_a", "_b"),
            )

            # Compute axiom preferences.
            preferences: Iterable[float] = chain(
                *self.axiom.preferences(
                    self.reranking_context,
                    query,
                    documents,
                )
            )
            pairs["preference"] = preferences

            return pairs


    class AxiomaticPermutationsCount(AxiomaticRankingTransformerBase):
        name = "AxiomaticPermutationsCount"
        description = "Counting permutations compared to axiom preferences"

        def transform_ranking(self, ranking: DataFrame):
            if len(ranking.index) == 0:
                # Empty ranking, skip feature scoring.
                return ranking

            # Convert to typed data classes.
            query = _query(ranking)
            documents = _documents(ranking)

            # Count permutations.
            permutations = self.axiom.permutation_frequency(
                self.reranking_context,
                query,
                documents,
            )

            # Convert reranked documents back to data frame.
            ranking["permutations_count"] = permutations

            return ranking