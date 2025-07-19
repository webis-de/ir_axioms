from ir_axioms.utils.libraries import is_pyserini_installed

if is_pyserini_installed():

    from dataclasses import dataclass, field
    from typing import Sequence

    from pyserini.analysis import Analyzer, get_lucene_analyzer

    from ir_axioms.tools.tokenizer.base import TermTokenizer

    @dataclass(frozen=True, kw_only=True)
    class AnseriniTermTokenizer(TermTokenizer):
        analyzer: Analyzer = field(
            default_factory=lambda: Analyzer(get_lucene_analyzer())
        )

        def terms(self, text: str) -> Sequence[str]:
            return self.analyzer.analyze(text)

else:
    AnseriniTermTokenizer = NotImplemented  # type: ignore
