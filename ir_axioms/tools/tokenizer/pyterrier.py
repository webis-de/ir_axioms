from pathlib import Path
from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed() or TYPE_CHECKING:
    from collections.abc import Sequence, Set, Collection
    from dataclasses import dataclass
    from functools import cached_property
    from itertools import chain, repeat
    from re import compile as re_compile
    from typing import Any

    from pyterrier.java import JavaClasses, autoclass, cast as java_cast
    from pyterrier.terrier import TerrierTokeniser, TerrierIndex

    from ir_axioms.model.utils import TokenizedString
    from ir_axioms.tools.tokenizer.base import TermTokenizer

    # Terrier-specific classes
    _J = JavaClasses(
        BaseTermPipelineAccessor="org.terrier.terms.BaseTermPipelineAccessor",
        ApplicationSetup="org.terrier.utility.ApplicationSetup",
        PropertiesIndex="org.terrier.structures.PropertiesIndex",
        StringReader="java.io.StringReader",
    )

    _TOKENIZATION_PROPERTY_NAMES = ("termpipelines", "tokeniser")

    def _properties_to_dict(properties: Any) -> dict[str, str]:
        """
        Extract tokenization-relevant properties from a ``java.util.Properties`` object as a Python dictionary.

        Note: Empty values are kept, e.g., an empty ``termpipelines`` property explicitly disables the term pipeline.
        """
        return {
            str(name): str(properties.getProperty(name)).strip()
            for name in properties.stringPropertyNames()
            if str(name).startswith(_TOKENIZATION_PROPERTY_NAMES)
        }

    _DEFAULT_TOKENISER = TerrierTokeniser.english

    _TERM_PIPELINE_PATTERN = re_compile(r"\s*,\s*")

    if TYPE_CHECKING:
        from typing import TypeVar

        # Fix the wrong typing of PyTerrier's required decorator.
        T = TypeVar("T")

        def pt_java_required(fn: T) -> T:
            return fn
    else:
        from pyterrier.java import required as pt_java_required

    @dataclass(frozen=True, kw_only=True)
    class TerrierTermTokenizer(TermTokenizer):
        tokeniser: TerrierTokeniser | Any | None = None
        index_location: TerrierIndex | Path | str | Any | None = None

        @cached_property
        def _index(self) -> TerrierIndex | None:
            if self.index_location is None:
                return None
            return TerrierIndex.coerce(self.index_location)  # type: ignore

        @cached_property
        def _index_properties(self) -> dict[str, str]:
            if self._index is None:
                return {}
            index_obj = self._index.index_obj()
            if not _J.PropertiesIndex._class.isInstance(index_obj):
                return {}
            properties = java_cast(_J.PropertiesIndex, index_obj).getProperties()
            return _properties_to_dict(properties)

        @cached_property
        def _application_setup_properties(self) -> dict[str, str]:
            properties = _J.ApplicationSetup.getProperties()
            return _properties_to_dict(properties)

        @cached_property
        def _properties(self) -> dict[str, str]:
            return self._application_setup_properties | self._index_properties

        @cached_property
        def _tokeniser(self) -> Any:
            # If specified as enum, return the enum's corresponding Java class.
            if isinstance(self.tokeniser, TerrierTokeniser):
                return TerrierTokeniser.java_tokeniser(self.tokeniser)  # type: ignore
            # Return the Java class directly.
            if self.tokeniser is not None:
                return self.tokeniser
            # Read the tokeniser setting from the index or setup properties.
            tokeniser_name = self._properties.get("tokeniser")
            if tokeniser_name is not None:
                if "." not in tokeniser_name:
                    # Use the fully qualified Java class name.
                    tokeniser_name = (
                        f"org.terrier.indexing.tokenisation.{tokeniser_name}"
                    )
                return autoclass(tokeniser_name)()
            # Fall back to the default Terrier tokeniser.
            return TerrierTokeniser.java_tokeniser(_DEFAULT_TOKENISER)  # type: ignore

        @cached_property
        def _term_pipelines(self) -> Sequence[Any]:
            # Read the term pipelines setting from the index or setup properties.
            term_pipelines = self._properties.get("termpipelines")
            if term_pipelines is not None:
                return [
                    _J.BaseTermPipelineAccessor(term_pipeline)
                    for term_pipeline in _TERM_PIPELINE_PATTERN.split(term_pipelines)
                ]
            return []

        @pt_java_required
        def terms(self, text: str) -> Sequence[str]:
            # ADDITION: Is there a way to restore the *sequence* of tokens from a TokenizedString (i.e., that contains a mapping of tokens to frequencies but no position information)?

            # Tokenize the text with the configured tokenizer.
            reader = _J.StringReader(str(text))
            terms = [
                str(term)
                for term in self._tokeniser.tokenise(reader)  # type: ignore
                if term is not None
            ]

            # Post-process the text with the configured term pipeline.
            for pipeline in self._term_pipelines:
                terms = [
                    str(term)
                    for term in map(pipeline.pipelineTerm, terms)
                    if term is not None
                ]
            return terms

        def terms_unordered(self, text: str) -> Collection[str]:
            # If the text is a TokenizedString (e.g., from TerrierTextContents), we can directly use its tokens.
            if isinstance(text, TokenizedString):
                return list(
                    chain.from_iterable(
                        repeat(term, count) for term, count in text.tokens.items()
                    )
                )

            return super().terms_unordered(text)

        def unique_terms(self, text: str) -> Set[str]:
            # If the text is a TokenizedString (e.g., from TerrierTextContents), we can directly use its tokens.
            if isinstance(text, TokenizedString):
                return set(text.tokens.keys())

            return super().unique_terms(text)


else:
    TerrierTermTokenizer = NotImplemented  # type: ignore
