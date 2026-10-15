from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed() or TYPE_CHECKING:
    from collections.abc import Sequence
    from dataclasses import dataclass
    from functools import cached_property
    from pathlib import Path
    from typing import Any

    from pyterrier.terrier import TerrierIndex

    from ir_axioms.model.utils import TokenizedString
    from ir_axioms.model.retrieval import Document
    from ir_axioms.tools.contents.base import TextContents

    _END_OF_POSTINGS = 2**31 - 1
    """Value of Terrier's `IterablePosting.EOL` (i.e., Java's `Integer.MAX_VALUE`)."""

    @dataclass(frozen=True, kw_only=True)
    class TerrierDocumentTextContents(TextContents[Document]):
        index_location: TerrierIndex | Path | str | Any
        text_field: str = "text"

        @cached_property
        def _index(self) -> TerrierIndex:
            return TerrierIndex.coerce(self.index_location)  # type: ignore

        @cached_property
        def _meta_index(self) -> Any:
            meta_index = self._index.meta_index()
            if meta_index is None:
                raise ValueError(
                    f"Index {self.index_location} does not have a metaindex."
                )
            return meta_index

        @cached_property
        def _document_index(self) -> Any:
            document_index = self._index.document_index()
            if document_index is None:
                raise ValueError(
                    f"Index {self.index_location} does not have a document index."
                )
            return document_index

        @cached_property
        def _direct_index(self) -> Any:
            direct_index = self._index.direct_index()
            if direct_index is None:
                raise ValueError(
                    f"Index {self.index_location} does not have a direct index."
                )
            return direct_index

        @cached_property
        def _lexicon(self) -> Any:
            lexicon = self._index.lexicon()
            if lexicon is None:
                raise ValueError(
                    f"Index {self.index_location} does not have a lexicon."
                )
            return lexicon

        @cached_property
        def _meta_index_keys(self) -> Sequence[str]:
            return [str(key) for key in self._meta_index.getKeys()]

        @cached_property
        def _document_contents_cache(self) -> dict[str, str]:
            return {}

        def _document_contents(self, document_id: str) -> str:
            if document_id in self._document_contents_cache:
                return self._document_contents_cache[document_id]
            if self.text_field not in self._meta_index_keys:
                raise ValueError(
                    f"Index {self.index_location} did not have "
                    f'requested metaindex key "{self.text_field}". '
                    f"Keys present in metaindex "
                    f"are {self._meta_index_keys}."
                )

            doc_id = int(self._meta_index.getDocument("docno", document_id))
            # Get the original text contents of this document.
            contents = str(
                self._meta_index.getItem(
                    self.text_field,
                    doc_id,
                )
            )
            # Get the index terms of this document.
            document_pointer = self._document_index.getDocumentEntry(doc_id)
            term_posting = self._direct_index.getPostings(document_pointer)
            tokens = {}
            # The posting iterator starts before the first posting.
            # Note: `endOfPostings()` is already true at the last posting.
            while term_posting.next() != _END_OF_POSTINGS:
                term_id = term_posting.getId()
                term_frequency = term_posting.getFrequency()
                term_lexicon_entry = self._lexicon.getLexiconEntry(term_id)
                term = str(term_lexicon_entry.getKey())
                tokens[term] = term_frequency
            document_contents = TokenizedString(
                contents,
                tokens=tokens,
            )
            self._document_contents_cache[document_id] = document_contents
            return document_contents

        def contents(self, input: Document) -> str:
            if input.text is not None:
                return input.text
            return self._document_contents(input.id)

else:
    TerrierDocumentTextContents = NotImplemented  # type: ignore
