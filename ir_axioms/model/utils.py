from typing import Mapping


class TokenizedString(str):
    """A string that is tokenized, i.e., its tokens can be accessed."""

    tokens: Mapping[str, int]

    def __new__(cls, value: str, tokens: Mapping[str, int]) -> "TokenizedString":
        instance = super().__new__(cls, value)
        instance.tokens = tokens
        return instance

    def __eq__(self, value: object) -> bool:
        return str(self) == value
