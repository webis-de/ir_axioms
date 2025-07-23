from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyserini_installed

if is_pyserini_installed() or TYPE_CHECKING:
    from pathlib import Path
    from typing import Optional, Union

    from injector import singleton, inject
    from ir_datasets import Dataset

    from ir_axioms.dependency_injection import injector
    from ir_axioms.model import (
        Query,
        Document,
    )
    from ir_axioms.tools import (
        TextContents,
        IrdsQueryTextContents,
        IrdsDocumentTextContents,
        AnseriniDocumentTextContents,
        TextStatistics,
        AnseriniTextStatistics,
        TermTokenizer,
        AnseriniTermTokenizer,
        IndexStatistics,
        AnseriniIndexStatistics,
    )
    from ir_axioms.tools.contents.simple import HasText
    from ir_axioms.utils.pyserini import Analyzer, default_analyzer

    def inject_pyserini(
        index_dir: Union[Path, str],
        analyzer: Optional[Analyzer] = None,
        text_contents: bool = False,
        dataset: Optional[Union[Dataset, str]] = None,
    ) -> None:
        analyzer_ = analyzer if analyzer is not None else default_analyzer()

        injector.binder.bind(
            interface=TextStatistics,
            to=AnseriniTextStatistics(
                index_dir=index_dir,
                analyzer=analyzer_,
            ),
            scope=singleton,
        )
        injector.binder.bind(
            interface=TermTokenizer,
            to=AnseriniTermTokenizer(analyzer=analyzer_),
            scope=singleton,
        )
        injector.binder.bind(
            interface=IndexStatistics,
            to=AnseriniIndexStatistics(
                index_dir=index_dir,
                analyzer=analyzer_,
            ),
            scope=singleton,
        )

        if text_contents:

            @inject
            def _make_terrier_document_text_contents(
                fallback_text_contents: TextContents[HasText],
            ) -> TextContents[Document]:
                return AnseriniDocumentTextContents(
                    fallback_text_contents=fallback_text_contents,
                    index_dir=index_dir,
                )

            injector.binder.bind(
                interface=TextContents[Document],
                to=_make_terrier_document_text_contents,
                scope=singleton,
            )

        if dataset is not None:

            @inject
            def _make_irds_query_text_contents(
                fallback_text_contents: TextContents[HasText],
            ) -> TextContents[Query]:
                return IrdsQueryTextContents(
                    fallback_text_contents=fallback_text_contents,
                    dataset=dataset,
                )

            @inject
            def _make_irds_document_text_contents(
                fallback_text_contents: TextContents[HasText],
            ) -> TextContents[Document]:
                return IrdsDocumentTextContents(
                    fallback_text_contents=fallback_text_contents,
                    dataset=dataset,
                )

            injector.binder.bind(
                interface=TextContents[Query],
                to=_make_irds_query_text_contents,
                scope=singleton,
            )
            injector.binder.bind(
                interface=TextContents[Document],
                to=_make_irds_document_text_contents,
                scope=singleton,
            )


else:
    inject_pyserini = NotImplemented
