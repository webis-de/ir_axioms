from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed() or TYPE_CHECKING:
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Set, Optional, Sequence, Union, Mapping, Hashable, Any

    from injector import singleton, Injector, InstanceProvider
    from ir_datasets import Dataset
    from pandas import DataFrame, Series
    from pyterrier import Transformer
    from typing_extensions import TypeAlias  # type: ignore

    from ir_axioms.dependency_injection import injector as _default_injector
    from ir_axioms.model import Query, Document
    from ir_axioms.tools import (
        TextContents,
        IrdsQueryTextContents,
        IrdsDocumentTextContents,
        TerrierDocumentTextContents,
        TextStatistics,
        TerrierDocumentTextStatistics,
        TermTokenizer,
        TerrierTermTokenizer,
        IndexStatistics,
        TerrierIndexStatistics,
    )
    from ir_axioms.utils.injection import reset_binding_scopes
    from ir_axioms.utils.pyterrier import (
        Index,
        IndexRef,
        Tokeniser,
        EnglishTokeniser,
    )

    _Index: TypeAlias = Index  # type: ignore
    _IndexRef: TypeAlias = IndexRef  # type: ignore
    _Tokeniser: TypeAlias = Tokeniser  # type: ignore

    def inject_pyterrier(
        index_location: Optional[Union[_Index, _IndexRef, Path, str]] = None,
        text_field: Optional[str] = "text",
        tokeniser: _Tokeniser = EnglishTokeniser(),
        dataset: Optional[Union[Dataset, str]] = None,
        injector: Injector = _default_injector,
    ) -> None:
        injector.binder.bind(
            interface=TermTokenizer,
            to=TerrierTermTokenizer(tokeniser=tokeniser),
            scope=singleton,
        )

        if index_location is not None:
            injector.binder.bind(
                interface=IndexStatistics,
                to=TerrierIndexStatistics(index_location=index_location),
                scope=singleton,
            )
            injector.binder.bind(
                interface=TextStatistics[Document],
                to=InstanceProvider(
                    TerrierDocumentTextStatistics(index_location=index_location)
                ),
                scope=singleton,
            )

            if text_field is not None:
                injector.binder.bind(
                    interface=TextContents[Document],
                    to=InstanceProvider(
                        TerrierDocumentTextContents(
                            index_location=index_location,
                            text_field=text_field,
                        )
                    ),
                    scope=singleton,
                )

        if dataset is not None:
            injector.binder.bind(
                interface=TextContents[Query],
                to=InstanceProvider(
                    IrdsQueryTextContents(
                        dataset=dataset,
                    )
                ),
                scope=singleton,
            )
            injector.binder.bind(
                interface=TextContents[Document],
                to=InstanceProvider(
                    IrdsDocumentTextContents(
                        dataset=dataset,
                    )
                ),
                scope=singleton,
            )

        reset_binding_scopes(injector)

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

    def load_document(
        row: Union[Series, Mapping[Hashable, Any]],
        text_column: Optional[str] = "text",
    ) -> Document:
        return Document(
            id=str(row["docno"]),
            text=str(row[text_column])
            if text_column is not None and text_column in row.keys()
            else None,
            score=float(row["score"]) if "score" in row.keys() else None,
            rank=int(row["rank"]) if "rank" in row.keys() else None,
            relevance=float(row["label"]) if "label" in row.keys() else None,
        )

    def load_documents(
        ranking: DataFrame,
        text_column: Optional[str] = "text",
    ) -> Sequence[Document]:
        return [
            load_document(
                row=row,
                text_column=text_column,
            )
            for _, row in ranking.iterrows()
        ]

    def load_query(row: Union[Series, Mapping[Hashable, Any]]) -> Query:
        return Query(
            id=str(row["qid"]),
            text=str(row["query"]) if "query" in row.keys() else None,
        )

    def load_queries(ranking: DataFrame) -> Sequence[Query]:
        return [load_query(row=row) for _, row in ranking.iterrows()]

    @dataclass(frozen=True)
    class FilterTopicsTransformer(Transformer):
        """
        Retain only queries that are contained in the topics.
        """

        topics: DataFrame

        def transform(self, inp: DataFrame) -> DataFrame:
            return inp[inp["qid"].isin(self.topics["qid"])]

    @dataclass(frozen=True)
    class FilterQrelsTransformer(Transformer):
        """
        Retain only query-document pairs that are contained in the qrels.
        """

        qrels: DataFrame

        def transform(self, inp: DataFrame) -> DataFrame:
            return inp[
                inp["qid"].isin(self.qrels["qid"])
                & inp["docno"].isin(self.qrels["docno"])
            ]

    @dataclass(frozen=True)
    class JoinQrelsTransformer(Transformer):
        """
        Join query-document pairs with their relevance labels.
        """

        qrels: DataFrame

        def transform(self, inp: DataFrame) -> DataFrame:
            qrels = self.qrels
            require_columns(qrels, {"qid", "docno", "label"})
            return inp.merge(self.qrels, on=["qid", "docno"], how="left")

    @dataclass(frozen=True)
    class AddNameTransformer(Transformer):
        """
        Add a colum containing the system's name.
        """

        system_name: str

        def transform(self, inp: DataFrame) -> DataFrame:
            inp["name"] = self.system_name
            return inp

else:
    inject_pyterrier = NotImplemented
    require_columns = NotImplemented
    load_document = NotImplemented
    load_documents = NotImplemented
    load_query = NotImplemented
    load_queries = NotImplemented
    FilterTopicsTransformer = NotImplemented  # type: ignore
    FilterQrelsTransformer = NotImplemented  # type: ignore
    JoinQrelsTransformer = NotImplemented  # type: ignore
    AddNameTransformer = NotImplemented  # type: ignore
