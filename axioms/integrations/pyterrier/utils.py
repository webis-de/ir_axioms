from axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed():
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Set, Optional, Sequence, Callable, Any, Union, cast

    from injector import singleton
    from ir_datasets import Dataset
    from pandas import DataFrame, Series
    from pyterrier import Transformer
    from pyterrier.datasets import IRDSDataset
    from pyterrier.java import (
        required as pt_java_required,
        autoclass as pt_java_autoclass,
    )

    from axioms.dependency_injection import injector
    from axioms.model import (
        RankedScoredDocument,
        RankedScoredTextDocument,
        JudgedRankedScoredTextDocument,
        JudgedRankedScoredDocument,
        Query,
        Document,
    )
    from axioms.tools import (
        TextContents,
        SimpleTextContents,
        IrdsQueryTextContents,
        IrdsDocumentTextContents,
        TerrierDocumentTextContents,
        TextStatistics,
        TerrierTextStatistics,
        TermTokenizer,
        TerrierTermTokenizer,
        IndexStatistics,
        TerrierIndexStatistics,
    )

    @pt_java_required
    def autoclass(*args, **kwargs) -> Any:
        return pt_java_autoclass(*args, **kwargs)

    Index = autoclass("org.terrier.structures.Index")
    IndexRef = autoclass("org.terrier.querying.IndexRef")
    Tokeniser = autoclass("org.terrier.indexing.tokenisation.Tokeniser")
    EnglishTokeniser = autoclass("org.terrier.indexing.tokenisation.EnglishTokeniser")

    def inject_pyterrier(
        index_location: Union[Index, IndexRef, Path, str],  # type: ignore
        text_field: Optional[str] = "text",
        tokeniser: Tokeniser = EnglishTokeniser(),  # type: ignore
        dataset: Optional[Union[Dataset, str, IRDSDataset]] = None,
    ) -> None:
        injector.binder.bind(
            interface=TextStatistics,
            to=TerrierTextStatistics(index_location=index_location),
            scope=singleton,
        )
        injector.binder.bind(
            interface=TermTokenizer,
            to=TerrierTermTokenizer(tokeniser=tokeniser),
            scope=singleton,
        )
        injector.binder.bind(
            interface=IndexStatistics,
            to=TerrierIndexStatistics(index_location=index_location),
            scope=singleton,
        )

        if text_field is not None:
            injector.binder.bind(
                interface=TextContents[Document],
                to=TerrierDocumentTextContents(
                    simple_text_contents=injector.get(SimpleTextContents),
                    index_location=index_location,
                    text_field=text_field,
                ),
                scope=singleton,
            )

        if dataset is not None:
            injector.binder.bind(
                interface=TextContents[Query],
                to=IrdsQueryTextContents(
                    simple_text_contents=injector.get(SimpleTextContents),
                    dataset=(
                        dataset
                        if not isinstance(dataset, IRDSDataset)
                        else cast(Dataset, dataset.irds_ref)
                    ),
                ),
                scope=singleton,
            )
            injector.binder.bind(
                interface=TextContents[Document],
                to=IrdsDocumentTextContents(
                    simple_text_contents=injector.get(SimpleTextContents),
                    dataset=(
                        dataset
                        if not isinstance(dataset, IRDSDataset)
                        else cast(Dataset, dataset.irds_ref)
                    ),
                ),
                scope=singleton,
            )

    def require_columns(
        ranking: DataFrame,
        expected_columns: Set[str],
    ) -> None:
        columns: Set[str] = set(ranking.columns)
        missing_columns: Set[str] = expected_columns - columns
        if len(missing_columns) > 0:
            raise ValueError(
                f"Expected columns "
                f"{', '.join(expected_columns)} but got columns "
                f"{', '.join(columns)} (missing columns "
                f"{', '.join(missing_columns)})."
            )

    def load_documents(
        ranking: DataFrame,
        text_column: Optional[str] = "text",
    ) -> Sequence[RankedScoredDocument]:
        require_columns(ranking, {"docno", "rank", "score"})

        parser: Callable[[Series], RankedScoredDocument]

        if "label" in ranking.columns:
            if text_column is not None:

                def parser(row: Series) -> RankedScoredDocument:
                    return JudgedRankedScoredTextDocument(
                        id=str(row["docno"]),
                        text=str(row[text_column]),
                        score=float(row["score"]),
                        rank=int(row["rank"]),
                        relevance=float(row["label"]),
                    )

            else:

                def parser(row: Series) -> RankedScoredDocument:
                    return JudgedRankedScoredDocument(
                        id=str(row["docno"]),
                        score=float(row["score"]),
                        rank=int(row["rank"]),
                        relevance=float(row["label"]),
                    )

        else:
            if text_column is not None:

                def parser(row: Series) -> RankedScoredDocument:
                    return RankedScoredTextDocument(
                        id=str(row["docno"]),
                        text=str(row[text_column]),
                        score=float(row["score"]),
                        rank=int(row["rank"]),
                    )

            else:

                def parser(row: Series) -> RankedScoredDocument:
                    return RankedScoredDocument(
                        id=str(row["docno"]),
                        score=float(row["score"]),
                        rank=int(row["rank"]),
                    )

        return [parser(row) for _, row in ranking.iterrows()]

    def load_queries(ranking: DataFrame) -> Sequence[Query]:
        require_columns(ranking, {"query"})
        return [Query(row["query"]) for _, row in ranking.iterrows()]

    @dataclass(frozen=True)
    class FilterTopicsTransformer(Transformer):
        """
        Retain only queries that are contained in the topics.
        """

        topics: DataFrame

        def transform(self, ranking: DataFrame) -> DataFrame:
            return ranking[ranking["qid"].isin(self.topics["qid"])]

    @dataclass(frozen=True)
    class FilterQrelsTransformer(Transformer):
        """
        Retain only query-document pairs that are contained in the qrels.
        """

        qrels: DataFrame

        def transform(self, ranking: DataFrame) -> DataFrame:
            return ranking[
                ranking["qid"].isin(self.qrels["qid"])
                & ranking["docno"].isin(self.qrels["docno"])
            ]

    @dataclass(frozen=True)
    class JoinQrelsTransformer(Transformer):
        """
        Join query-document pairs with their relevance labels.
        """

        qrels: DataFrame

        def transform(self, ranking: DataFrame) -> DataFrame:
            qrels = self.qrels
            require_columns(qrels, {"qid", "docno", "label"})
            return ranking.merge(self.qrels, on=["qid", "docno"], how="left")

    @dataclass(frozen=True)
    class AddNameTransformer(Transformer):
        """
        Add a colum containing the system's name.
        """

        system_name: str

        def transform(self, res: DataFrame) -> DataFrame:
            res["name"] = self.system_name
            return res

else:
    inject_pyterrier = NotImplemented
    require_columns = NotImplemented
    load_documents = NotImplemented
    load_queries = NotImplemented
    FilterTopicsTransformer = NotImplemented  # type: ignore
    FilterQrelsTransformer = NotImplemented  # type: ignore
    JoinQrelsTransformer = NotImplemented  # type: ignore
    AddNameTransformer = NotImplemented  # type: ignore
