from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed() or TYPE_CHECKING:
    from dataclasses import dataclass, field
    from functools import cached_property
    from re import compile as re_compile
    from typing import Sequence, Any

    from ir_axioms.tools.tokenizer.base import TermTokenizer
    from ir_axioms.utils.pyterrier import (
        pt_java_required,
        Tokeniser,
        EnglishTokeniser,
        BaseTermPipelineAccessor,
        ApplicationSetup,
    )

    _TERM_PIPELINE_PATTERN = re_compile(r"\s*,\s*")

    @pt_java_required
    @dataclass(frozen=True, kw_only=True)
    class TerrierTermTokenizer(TermTokenizer):
        tokeniser: Tokeniser = field(  # type: ignore
            default_factory=lambda: EnglishTokeniser()
        )
        # TODO: Add optional index location arg to guess tokenizer and term pipelines from index configuration.

        @cached_property
        def _term_pipelines(self) -> Sequence[Any]:
            term_pipelines = str(
                ApplicationSetup.getProperty(
                    "termpipelines",
                    "Stopwords,PorterStemmer",
                )
            ).strip()
            return [
                BaseTermPipelineAccessor(pipeline)
                for pipeline in _TERM_PIPELINE_PATTERN.split(term_pipelines)
            ]

        def terms(self, text: str) -> Sequence[str]:
            from pyterrier.java import J

            reader = J.StringReader(text)
            terms = [
                str(term)
                for term in self.tokeniser.tokenise(reader)  # type: ignore
                if term is not None
            ]
            del reader

            for pipeline in self._term_pipelines:
                terms = [
                    str(term)
                    for term in map(pipeline.pipelineTerm, terms)
                    if term is not None
                ]
            return terms

else:
    TerrierTermTokenizer = NotImplemented  # type: ignore
