from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyserini_installed

if is_pyserini_installed() or TYPE_CHECKING:
    from dataclasses import dataclass, field
    from typing import Sequence

    from ir_axioms.tools.tokenizer.base import TermTokenizer
    from ir_axioms.utils.pyserini import Analyzer, default_analyzer

    @dataclass(frozen=True, kw_only=True)
    class AnseriniTermTokenizer(TermTokenizer):
        analyzer: Analyzer = field(default_factory=default_analyzer)

        def terms(self, text: str) -> Sequence[str]:
            return self.analyzer.analyze(text)

else:
    AnseriniTermTokenizer = NotImplemented  # type: ignore
