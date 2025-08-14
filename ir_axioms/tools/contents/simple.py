from typing import Protocol, runtime_checkable

from ir_axioms.tools.contents.base import TextContents


@runtime_checkable
class HasText(Protocol):
    text: str | None


class SimpleTextContents(TextContents[HasText]):
    def contents(self, input: HasText) -> str:
        if input.text is not None:
            return input.text
        raise ValueError(f"Could not get text contents from: {input}")
