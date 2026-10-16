# Roadmap

<p class="subtitle">Ideas, known issues, and cleanup work</p>

This page collects features we would like to add to `ir_axioms`, known bugs, and leftover cleanup work.
Contributions are very welcome: pick an item, open an [issue](https://github.com/webis-de/ir_axioms/issues) to discuss it, and see the [contribution guide](contributing.md).

## Known Bugs

!!! bug "Preconditions are ignored by several axioms"

    [`PreconditionMixin`][ir_axioms.axiom.precondition.PreconditionMixin] applies the precondition by wrapping `preference()` and `preferences()`.
    But the following axioms define these methods in the same class, which shadows the mixin's methods, so their `precondition` (by default, `LEN()`) is never checked:
    [TFC1](../axioms/retrieval.md#tfc1), [TFC3](../axioms/retrieval.md#tfc3), [REG](../axioms/retrieval.md#reg), ANTI-REG, [ArgUC](../axioms/retrieval.md#arguc), [QTArg](../axioms/retrieval.md#qtarg), [QTPArg](../axioms/retrieval.md#qtparg), and [aSL](../axioms/retrieval.md#asl).
    The `LEN_*` variants (e.g., [`LenAndAxiom`][ir_axioms.axiom.retrieval.query_aspects.LenAndAxiom]) are not affected, as they inherit the logic from a base class.

    **Fix:** Move the logic into a base class (like [`AndAxiom`][ir_axioms.axiom.retrieval.query_aspects.AndAxiom] → `LenAndAxiom`), or apply the precondition mask explicitly.
    Then add a unit test that checks that each `PreconditionMixin` subclass respects its precondition.
    Note that fixing this changes the preferences of these axioms (and thus results computed with them).

!!! bug "`RandomAxiom` never prefers the first output"

    `RandomAxiom` draws preferences with `integers(-1, 1)`, whose upper bound is exclusive, so it only ever returns −1 or 0.
    **Fix:** Use `integers(-1, 2)`, and make the preference matrix antisymmetric (mirror the upper triangle).

!!! bug "`AxiomaticExperiment.inconsistent_pairs` filter is always true"

    The filter "at least one axiom would have corrected the ranking error" also includes ORACLE itself, which by definition prefers the correct document.
    Hence, all wrongly ranked pairs are returned.
    **Fix:** Only consider the user-given axioms (and maybe ORIG) in the filter.

!!! bug "CLAR2 prefers less readable responses"

    [CLAR2](../axioms/generation.md#clar2) should prefer the more readable response, but both its formalization in the ICTIR 2025 paper (Eq. 11) and the implementation prefer the response with the _lower_ Flesch reading ease score, and lower scores indicate _harder_ text.
    **Fix:** Use `strictly_greater` in `FleschReadingEaseClarityAxiom`.
    Note that this changes the axiom's preferences compared to the published experiments.

!!! bug "Outdated docstring of `Axiom`"

    The docstring of `Axiom` mentions a `~` operator (formerly used for caching), which no longer exists, and a `rerank()` method, which is now called `rerank_kwiksort()`.

## Cleanup

### Outdated Notebooks

Several [example notebooks](https://github.com/webis-de/ir_axioms/tree/main/experiments) still use the legacy (< 1.0) API, e.g., `ir_axioms.backend.pyterrier` imports, `~axiom` caching, removed axioms (`RS_*`, `*_f`), passing `index=` and `dataset=` to transformers, or PyTerrier's deprecated `BatchRetrieve`.
The [walkthroughs](../walkthroughs/index.md) already show the current API and can serve as a template.

| Notebook | Legacy usages |
|:--|:--|
| `pyterrier_kwiksort.ipynb` | `backend`, `modules.pivot`, `~`, `RS_*`, `*_f`, `BatchRetrieve`, `index=` |
| `pyterrier_kwiksort_learned.ipynb` | `BatchRetrieve`, `index=` |
| `pyterrier_ltr_features.ipynb` | `backend`, `~`, `RS_*`, `*_f`, `BatchRetrieve`, `index=` |
| `pyterrier_post_hoc_analysis_of_runs_and_qrels.ipynb` | `backend`, `~`, `RS_*`, `*_f`, `BatchRetrieve`, `index=` |
| `pyterrier_preferences.ipynb` | `backend`, `~`, `RS_*`, `*_f`, `BatchRetrieve`, `index=` (duplicate of the post-hoc analysis notebook) |
| `pyterrier_preferences_parallel.ipynb` | `backend`, `RS_*`, `*_f`, `index=` |
| `pyterrier_reranking_comparison.ipynb` | `backend`, `modules.pivot`, `~`, `RS_*`, `*_f`, `BatchRetrieve`, `index=` |
| `sigir2022_showcase.ipynb` | `backend`, `BatchRetrieve` |
| `sigir2022/trec-dl-ms-marco-passage.ipynb` | `backend` |
| `ictir2025.ipynb` | deprecated [`GenerationInput`][ir_axioms.model.generation.GenerationInput]/[`GenerationOutput`][ir_axioms.model.generation.GenerationOutput] |

For the SIGIR 2022 reproducibility notebooks, it might be better to pin them to the legacy version instead (e.g., `pip install "ir_axioms<1.0"`).

### Code and Repository

- **Optional dependencies:** Many dependencies are only needed for some axioms or integrations (`ir-datasets`, `language-tool-python`, `negspacy`, `nltk`, `rouge-score`, `spacy`, `targer-api`, `textacy`, `trectools`; see the TODOs in `pyproject.toml`). Move them to extras and expose dependent interfaces only if installed.
- **NumPy 2:** The code uses `numpy.float_`, which was removed in NumPy 2.0, and pins `numpy~=1.26`. Replace it with `numpy.float64` and relax the pin.
- **TREC AutoJudge code:** Migrate the code in `trec-auto-judge/` to a proper place in the package (currently excluded from type checks).
- **Experiments linting:** Clean up the `experiments/` directory so that it can be included in `ruff` checks again.
- **Stray file:** `docs/term_frequency.py.diff` seems to be a leftover from a refactoring and can probably be removed.
- **Defaults pointing to untracked data:** [`TrecRagNuggetAxiom`][ir_axioms.axiom.generation.trec.TrecRagNuggetAxiom] defaults to `data/nugget_assignment.20241108.jsonl`, which is not in the repository (the tracked file is `nugget_assignment.20241218.jsonl`). Other RAG oracle axioms also default to relative `data/` paths.

### Documentation

- **Docstrings:** Most axioms and tools have no or only short docstrings. Adding them (with parameters and references) would greatly improve the [API reference](../api/index.md).
- **Cross-links:** Link class names in the [axiom catalog](../axioms/index.md) to their API reference entries.
- **Tested examples:** Run the code examples of the documentation in CI (e.g., with doctests or by executing the walkthroughs on a small dataset), so that they do not get outdated again.
- **Executed walkthroughs:** Render the updated notebooks directly in the documentation (e.g., with `mkdocs-jupyter`), including their outputs.

## Feature Ideas

### Axioms

- **Missing retrieval axioms:** Find practical pairwise reformulations of [TFC2, LNC2, LB2](../axioms/retrieval.md#tfc2), the original TDC, and [STMC3](../axioms/related.md#semantic-term-matching).
- **Further constraints from the literature:** E.g., term semantic similarity constraints (TSSC1/2), proximity constraints of Tao and Zhai, or constraints for pseudo-relevance feedback (see [related axioms](../axioms/related.md)).
- **More RAG axioms:** See the [list of missing axioms and ideas](../axioms/generation.md#missing-axioms-and-axiom-ideas), e.g., factual grounding, discourse-based clarity, or multi-modal outputs.
- **More axioms from our own papers:** QPHRA, RSIM, and P-RANK [from axiomatic re-ranking](../axioms/related.md#axiomatic-re-ranking), CompArg, CompPArg, and ArgQ for comparative questions, and the embedding-based QArgSim and QSenSim axioms for [argument retrieval](../axioms/related.md#argument-retrieval).
- **Generative variants for index-based axioms:** TFC3, M-TDC, and LB1 have no `GEN_*` counterpart, as they need collection statistics or retrieval scores. These could be derived, e.g., from the retrieved context.
- **LLM-based axioms:** Axioms that ask an LLM for a pairwise preference on a specific, narrow criterion, with caching by default.

### Integrations and Tools

- **Command line interface:** The `ir_axioms` command currently only has a placeholder command. A CLI for computing preferences, re-ranking run files, and running axiomatic experiments (like in the legacy version) would make the library usable without Python code, e.g., in [TIRA](https://www.tira.io/).
- **Pyserini transformers:** Re-ranking and preference computation for Pyserini, similar to the [PyTerrier transformers](../getting-started/integrations.md#pyterrier-transformers).
- **Cache keys:** [`CachedAxiom`][ir_axioms.axiom.cache.CachedAxiom] keys preferences by the `repr()` of the input and outputs, which includes ranks and scores. Caching only by IDs and texts would increase cache hits across runs.
- **Pivot selection:** Smarter pivot selection for KwikSort (e.g., by ORIG rank or by axiom "wins") to make re-rankings more stable.
