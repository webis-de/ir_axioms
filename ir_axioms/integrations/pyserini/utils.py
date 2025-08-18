from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyserini_installed

if is_pyserini_installed() or TYPE_CHECKING:
    from pathlib import Path
    from typing import Optional, Union

    from injector import singleton, Injector
    from ir_datasets import Dataset

    from ir_axioms.dependency_injection import injector as _default_injector
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
    from ir_axioms.utils.injection import reset_binding_scopes
    from ir_axioms.utils.pyserini import Analyzer, default_analyzer

    def inject_pyserini(
        index_dir: Optional[Union[Path, str]] = None,
        analyzer: Optional[Analyzer] = None,
        text_contents: bool = False,
        dataset: Optional[Union[Dataset, str]] = None,
        injector: Injector = _default_injector,
    ) -> None:
        analyzer_ = analyzer if analyzer is not None else default_analyzer()
        injector.binder.bind(
            interface=TermTokenizer,
            to=AnseriniTermTokenizer(analyzer=analyzer_),
            scope=singleton,
        )

        if index_dir is not None:
            injector.binder.bind(
                interface=TextStatistics,
                to=AnseriniTextStatistics(
                    index_dir=index_dir,
                    analyzer=analyzer_,
                ),
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
                injector.binder.bind(
                    interface=TextContents[Document],
                    to=AnseriniDocumentTextContents(
                        index_dir=index_dir,
                    ),
                    scope=singleton,
                )

        if dataset is not None:
            injector.binder.bind(
                interface=TextContents[Query],
                to=IrdsQueryTextContents(
                    dataset=dataset,
                ),
                scope=singleton,
            )
            injector.binder.bind(
                interface=TextContents[Document],
                to=IrdsDocumentTextContents(
                    dataset=dataset,
                ),
                scope=singleton,
            )

        reset_binding_scopes(injector)


else:
    inject_pyserini = NotImplemented
