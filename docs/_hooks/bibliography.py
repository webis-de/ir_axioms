"""
MkDocs hook that turns ``docs/references.bib`` into the single source of truth
for all citations in the documentation.

- ``[@key]`` or ``[@key1; @key2]`` in any page is replaced by author-year links.
- Each page that cites at least one reference gets a "References" section at
  the bottom, listing the cited references, including the BibTeX of each entry.

Unknown citation keys fail the build, so that broken references are caught in CI.
"""

from pathlib import Path
from re import DOTALL, MULTILINE, compile as re_compile
from typing import Any

from mkdocs.exceptions import PluginError

_BIB_PATH = Path(__file__).parent.parent / "references.bib"

_PATTERN_ENTRY = re_compile(r"^@(\w+)\{([^,\s]+),", MULTILINE)
_PATTERN_AUTHOR_SEPARATOR = re_compile(r"\s+and\s+")
_PATTERN_FIELD_NAME = re_compile(r"\s*(\w+)\s*=\s*\{")
_PATTERN_CITATION = re_compile(r"\[(@[\w-]+(?:\s*;\s*@[\w-]+)*)\]")
# Fenced code blocks and inline code spans, in which citations are not replaced.
_PATTERN_CODE = re_compile(r"(^[ \t]*```.*?^[ \t]*```|`[^`\n]+`)", MULTILINE | DOTALL)

_LATEX_REPLACEMENTS = {
    r"{\"{o}}": "ö",
    r"{\"{a}}": "ä",
    r"{\"{u}}": "ü",
    r"{\'{e}}": "é",
    r"{\'{E}}": "É",
    r"{\'{o}}": "ó",
    r"{\'{a}}": "á",
    r"{\u{a}}": "ă",
    r"{\^{a}}": "â",
    r"\&": "&",
    r"{-}": "-",
    r"\_": "_",
}


def _clean(value: str) -> str:
    for latex, text in _LATEX_REPLACEMENTS.items():
        value = value.replace(latex, text)
    value = value.replace("{", "").replace("}", "")
    return " ".join(value.split())


def _read_braced(text: str, start: int) -> int:
    """Return the index just after the brace closing the one opened before ``start``."""
    depth = 1
    index = start
    while depth > 0:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1
    return index


def _load_bibliography() -> dict[str, dict[str, str]]:
    text = _BIB_PATH.read_text(encoding="utf-8")
    entries: dict[str, dict[str, str]] = {}
    for match in _PATTERN_ENTRY.finditer(text):
        entry_type, key = match.groups()
        entry_end = _read_braced(text, match.start() + len(entry_type) + 2)
        fields: dict[str, str] = {
            "ENTRYTYPE": entry_type,
            "bibtex": text[match.start() : entry_end],
        }
        position = match.end()
        while (field_match := _PATTERN_FIELD_NAME.match(text, position)) is not None:
            value_end = _read_braced(text, field_match.end())
            fields[field_match.group(1).lower()] = text[field_match.end() : value_end - 1]
            position = value_end + 1  # Skip the trailing comma.
        if key in entries:
            raise PluginError(f"Duplicate BibTeX key '{key}' in {_BIB_PATH.name}.")
        entries[key] = fields
    return entries


def _authors(entry: dict[str, str]) -> list[str]:
    return [_clean(author) for author in _PATTERN_AUTHOR_SEPARATOR.split(entry.get("author", ""))]


def _short_citation(entry: dict[str, str]) -> str:
    surnames = [author.split()[-1] for author in _authors(entry)]
    if len(surnames) == 1:
        names = surnames[0]
    elif len(surnames) == 2:
        names = f"{surnames[0]} and {surnames[1]}"
    else:
        names = f"{surnames[0]} et al."
    return f"{names}, {entry.get('year', 'n.d.')}"


def _full_citation(entry: dict[str, str]) -> str:
    authors = _authors(entry)
    author_text = (
        ", ".join(authors[:-1]) + ", and " + authors[-1]
        if len(authors) > 2
        else " and ".join(authors)
    )
    title = _clean(entry.get("title", ""))
    venue = _clean(
        entry.get("booktitle")
        or entry.get("journal")
        or entry.get("publisher")
        or entry.get("howpublished", "")
    )
    details = []
    if "volume" in entry and "journal" in entry:
        volume = entry["volume"]
        if "number" in entry:
            volume += f"({entry['number']})"
        details.append(volume)
    if "pages" in entry:
        details.append(_clean(entry["pages"]).replace("--", "–"))
    details.append(entry.get("year", ""))
    citation = f"{author_text}. **{title}.** _{venue}_"
    citation += ", " + ", ".join(detail for detail in details if detail) + "."
    if "url" in entry:
        citation += f" [{_clean(entry['url'])}]({_clean(entry['url'])})"
    return citation


def _sort_key(item: tuple[str, dict[str, str]]) -> tuple[str, str]:
    key, entry = item
    return (_authors(entry)[0].split()[-1].lower(), entry.get("year", ""))


def _render_references(keys: set[str]) -> str:
    entries = {key: _bibliography[key] for key in keys}
    blocks = ["## References\n"]
    for key, entry in sorted(entries.items(), key=_sort_key):
        bibtex = "\n".join("    " + line for line in entry["bibtex"].splitlines())
        blocks.append(
            f'<a id="{key}"></a>\n'
            f"{_full_citation(entry)}\n\n"
            f'??? quote "BibTeX"\n\n'
            f"    ```bibtex\n{bibtex}\n    ```\n"
        )
    return "\n".join(blocks)


_bibliography: dict[str, dict[str, str]] = {}


def on_pre_build(config: Any, **kwargs: Any) -> None:
    global _bibliography
    _bibliography = _load_bibliography()


def on_page_markdown(markdown: str, page: Any, **kwargs: Any) -> str:
    src_uri: str = page.file.src_uri
    cited_keys: set[str] = set()

    def replace_citation(match: Any) -> str:
        keys = [key.strip().lstrip("@") for key in match.group(1).split(";")]
        links = []
        for key in keys:
            if key not in _bibliography:
                raise PluginError(f"Unknown citation key '{key}' in {src_uri}.")
            cited_keys.add(key)
            links.append(f"[{_short_citation(_bibliography[key])}](#{key})")
        return "(" + "; ".join(links) + ")"

    # Odd indices of the split are code, which is kept as is.
    markdown = "".join(
        part if index % 2 == 1 else _PATTERN_CITATION.sub(replace_citation, part)
        for index, part in enumerate(_PATTERN_CODE.split(markdown))
    )
    if cited_keys:
        markdown = markdown.rstrip() + "\n\n" + _render_references(cited_keys)
    return markdown
