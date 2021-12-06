from typing import Union

from ir_axioms.model import Query, Document


def text_content(query_or_document: Union[Query, Document]) -> str:
    if isinstance(query_or_document, Query):
        return query_or_document.title
    elif isinstance(query_or_document, Document):
        return query_or_document.content
    else:
        raise ValueError(
            f"Expected Query or Document "
            f"but got {type(query_or_document)}."
        )
