from axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed():

    from dataclasses import dataclass, field
    from functools import cached_property
    from re import compile as re_compile
    from typing import Sequence, Any

    from pyterrier.java import (
        required as pt_java_required,
        autoclass as pt_java_autoclass,
        J,
    )

    from axioms.tools.tokenizer.base import TermTokenizer

    @pt_java_required
    def autoclass(*args, **kwargs) -> Any:
        return pt_java_autoclass(*args, **kwargs)

    Tokeniser = autoclass("org.terrier.indexing.tokenisation.Tokeniser")
    EnglishTokeniser = autoclass("org.terrier.indexing.tokenisation.EnglishTokeniser")
    TermPipelineAccessor = autoclass("org.terrier.terms.TermPipelineAccessor")
    BaseTermPipelineAccessor = autoclass("org.terrier.terms.BaseTermPipelineAccessor")
    ApplicationSetup = autoclass("org.terrier.utility.ApplicationSetup")

    _TERM_PIPELINE_PATTERN = re_compile(r"\s*,\s*")

    @pt_java_required
    @dataclass(frozen=True, kw_only=True)
    class TerrierTermTokenizer(TermTokenizer):
        tokeniser: Tokeniser = field(  # type: ignore
            default_factory=lambda: EnglishTokeniser()
        )

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
