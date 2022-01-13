from pathlib import Path
from typing import Union, Optional

from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy
from tqdm import tqdm

from ir_axioms.app import rerank_ranking
from ir_axioms.axiom import Axiom
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


    class AxiomaticReranker(TransformerBase):
        axiom: Axiom
        reranking_context: RerankingContext

        def __init__(
                self,
                axiom: Axiom,
                index_location: Union[Path, IndexRef, Index],
                tokeniser: Tokeniser = EnglishTokeniser(),
                cache_dir: Optional[Path] = None,
        ):
            self.axiom = axiom
            self.reranking_context = IndexRerankingContext(
                index_location,
                tokeniser,
                cache_dir
            )

        def _rerank(self, ranking: DataFrame):
            if len(ranking.index) == 0:
                # Empty ranking, skip reranking.
                return ranking

            # As we grouped per query, we don't expect multiple queries here.
            assert ranking["qid"].nunique() <= 1
            assert ranking["query"].nunique() <= 1

            # Convert to typed data classes.
            query = Query(ranking.iloc[0]["query"])
            documents = [
                RankedDocument(
                    id=row["docno"],
                    content=row["text"],
                    score=row["score"],
                    rank=row["rank"],
                )
                for index, row in ranking.iterrows()
            ]

            # Rerank documents.
            reranked_documents = rerank_ranking(
                self.axiom,
                documents,
                query,
                self.reranking_context
            )

            # Convert reranked documents back to data frame.
            reranked = DataFrame({
                'docno': [doc.id for doc in reranked_documents],
                'rank': [doc.rank for doc in reranked_documents],
                'score': [doc.score for doc in reranked_documents],
            })

            # Remove old scores and ranks.
            ranking = ranking.copy()
            del ranking["rank"]
            del ranking["score"]

            # Merge with new scores.
            reranked = reranked.merge(ranking, on="docno")
            return reranked

        def transform(self, ranking: DataFrame):
            if "text" not in ranking.columns:
                raise ValueError(
                    "Can only rerank when a 'text' field is given.")
            tqdm.pandas(
                desc="Reranking axiomatically",
                unit=" topics",
            )  # Show progress during reranking queries.
            query_rankings: DataFrameGroupBy = ranking.groupby(
                ["qid", "query"])
            query_rankings = query_rankings.progress_apply(self._rerank)
            return query_rankings.reset_index(drop=True)
