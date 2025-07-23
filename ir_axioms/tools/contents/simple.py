from typing import Protocol, TypeVar, runtime_checkable
from dataclasses import dataclass

from ir_axioms.tools.contents.base import TextContents


@runtime_checkable
class HasText(Protocol):
    text: str


class SimpleTextContents(TextContents[HasText]):
    def contents(self, input: HasText) -> str:
        return input.text


T = TypeVar("T")


@dataclass(frozen=True, kw_only=True)
class FallbackTextContentsMixin(TextContents[T]):
    fallback_text_contents: TextContents[HasText]

    def contents(self, input: T) -> str:
        if isinstance(input, HasText):
            return self.fallback_text_contents.contents(input)
        return super().contents(input)  # type: ignore[safe-super]
