from itertools import groupby
from os import PathLike
from pathlib import Path
from random import randint
from typing import List, Union, Dict, Optional, Tuple, Iterable

from pandas import DataFrame, Series
from trectools import TrecRun, TrecTopics

from ir_axioms import logger
from ir_axioms.axiom import Axiom, AggregatedAxiom
from ir_axioms.backend import is_pyserini_installed, is_pyterrier_installed
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext


def _kwiksort(
        axiom: Axiom,
        query: Query,
        context: RerankingContext,
        vertices: List[RankedDocument]
) -> List[RankedDocument]:
    if len(vertices) == 0:
        return []

    vertices_left = []
    vertices_right = []

    # Select random pivot.
    logger.debug("Selecting reranking pivot.")
    pivot = vertices[randint(0, len(vertices) - 1)]

    for vertex in vertices:
        if vertex == pivot:
            continue

        preference = axiom.preference(context, query, vertex, pivot)
        if preference > 0:
            vertices_left.append(vertex)
        elif preference < 0:
            vertices_right.append(vertex)
        elif vertex.rank < pivot.rank:
            vertices_left.append(vertex)
        elif vertex.rank > pivot.rank:
            vertices_right.append(vertex)
        else:
            raise RuntimeError(
                f"Tie during reranking. "
                f"Document {vertex} has same preference "
                f"and rank as pivot document {pivot}."
            )

    vertices_left = _kwiksort(
        axiom,
        query,
        context,
        vertices_left
    )
    vertices_right = _kwiksort(
        axiom,
        query,
        context,
        vertices_right
    )

    return [*vertices_left, pivot, *vertices_right]


def _reset_score(ranking: List[RankedDocument]) -> List[RankedDocument]:
    length = len(ranking)
    return [
        RankedDocument(
            id=document.id,
            content=document.content,
            score=length - i,
            rank=i + 1,
        )
        for i, document in enumerate(ranking)
    ]


def _load_context(
        context: Union[RerankingContext, PathLike],
        cache_dir: Optional[PathLike] = None,
) -> RerankingContext:
    if cache_dir is not None:
        cache_dir = Path(cache_dir)

    if isinstance(context, PathLike):
        context = Path(context)
        if is_pyserini_installed():
            from ir_axioms.backend.pyserini import IndexRerankingContext
            return IndexRerankingContext(
                index_dir=context,
                cache_dir=cache_dir,
            )
        elif is_pyterrier_installed():
            from ir_axioms.backend.pyterrier import IndexRerankingContext
            return IndexRerankingContext(
                index_location=context,
                cache_dir=cache_dir,
            )
        else:
            raise NotImplementedError("No backend found.")

    assert isinstance(context, RerankingContext)
    return context


def _load_axiom(axiom: Union[Axiom, Iterable[Axiom]]) -> Axiom:
    if isinstance(axiom, Iterable):
        return AggregatedAxiom(axiom)
    assert isinstance(axiom, Axiom)
    return axiom


def _load_ranking(ranking: Union[List[RankedDocument]]) -> List[
    RankedDocument]:
    assert isinstance(ranking, List)
    return ranking


def _load_run(run: Union[TrecRun, PathLike]) -> TrecRun:
    if isinstance(run, PathLike):
        return TrecRun(run)
    assert isinstance(run, TrecRun)
    return run


def _load_topics(topics: Union[TrecTopics, PathLike, Dict]) -> TrecTopics:
    if isinstance(topics, PathLike):
        topics = TrecTopics()
        topics.read_topics_from_file(topics)
        return topics
    if isinstance(topics, Dict):
        return TrecTopics(topics=topics)
    assert isinstance(topics, TrecTopics)
    return topics


def _load_query(query: Union[Query, str]) -> Query:
    if isinstance(query, str):
        return Query(query)
    assert isinstance(query, Query)
    return query


def _load_rankings(
        run: TrecRun,
        topics: TrecTopics,
) -> Dict[str, List[RankedDocument]]:
    raise NotImplementedError()


def rerank_ranking(
        axiom: Union[Axiom, Iterable[Axiom]],
        ranking: Union[List[RankedDocument]],
        query: Union[Query, str],
        context: Union[RerankingContext, PathLike],
        cache_dir: Optional[PathLike] = None,
) -> List[RankedDocument]:
    axiom: Axiom = _load_axiom(axiom)
    ranking: List[RankedDocument] = _load_ranking(ranking)
    query: Query = _load_query(query)
    context: RerankingContext = _load_context(context, cache_dir)

    ranking = _kwiksort(axiom, query, context, ranking)
    ranking = _reset_score(ranking)

    return ranking


def rerank_run(
        axiom: Union[Axiom, Iterable[Axiom]],
        run: Union[TrecRun, PathLike],
        topics: Union[TrecTopics, PathLike, Dict],
        context: Union[RerankingContext, PathLike],
        tag: str = "ir_axioms",
        cache_dir: Optional[PathLike] = None,
) -> TrecRun:
    run: TrecRun = _load_run(run)
    run_data: DataFrame = run.run_data
    topics: TrecTopics = _load_topics(topics)
    context: RerankingContext = _load_context(context, cache_dir)

    # Read rows and query IDs from data frame.
    query_rows: List[Tuple[int, Series]] = [
        (int(row["query"]), row)
        for index, row in run_data.iterrows()
    ]

    # Group rankings per query.
    rankings: Dict[int, List[RankedDocument]] = {
        query_id: [
            RankedDocument(
                row["docid"],
                context.document_content(row["docid"]),
                float(row["score"]),
                int(row["rank"]),
            )
            for _, row in group
        ]
        for query_id, group in
        groupby(query_rows, lambda query_row: query_row[0])
    }

    # Rerank each query's ranking.
    rankings = {
        query_id: rerank_ranking(
            axiom,
            ranking,
            topics.topics[query_id],
            context,
            cache_dir,
        )
        for query_id, ranking in rankings.items()
    }

    # Convert back to TREC run.
    reranked_df = DataFrame(
        columns=run_data.columns,
    )
    for query_id, ranking in rankings.items():
        reranked_query_df = DataFrame(
            data=[
                (
                    query_id,
                    "Q0",
                    document.id,
                    document.rank,
                    document.score,
                    tag,
                )
                for document in ranking
            ],
            columns=reranked_df.columns,
        )
        reranked_df.append(reranked_query_df)

    reranked_df.sort_values(
        ["query", "score", "docid"],
        inplace=True,
        ascending=[True, False, True],
    )
    return TrecRun(reranked_df)

def rerank_ir_datasets_run(
        axiom: Union[Axiom, Iterable[Axiom]],
        run: Union[TrecRun, PathLike],
        dataset: str,
        tag: str = "ir_axioms",
        cache_dir: Optional[PathLike] = None,
) -> TrecRun:
    pass

def save_rerank_run(
        axiom: Union[Axiom, Iterable[Axiom]],
        run: Union[TrecRun, PathLike],
        reranked_run_path: PathLike,
        topics: Union[TrecTopics, PathLike, Dict],
        context: Union[RerankingContext, PathLike],
        tag: str = "ir_axioms",
        cache_dir: Optional[PathLike] = None,
) -> None:
    reranked_run = rerank_run(
        axiom,
        run,
        topics,
        context,
        tag,
        cache_dir,
    )
    run_data: DataFrame = reranked_run.run_data
    run_data.to_csv(
        reranked_run_path,
        sep=" ",
        header=False,
    )
