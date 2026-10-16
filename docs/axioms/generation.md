# RAG Axioms

<p class="subtitle">Constraints on generated responses for a request</p>

RAG axioms compare two generated responses for the same request (optionally including the retrieved context) and implement the interface `Axiom[GenerationRequest, GenerationResponse]`.
They were introduced in our ICTIR 2025 paper [@merker2025axioms].

Below, we provide an overview and mapping of various [utility dimensions](#utility-dimensions) described in related work to the [implemented axioms](#axioms-for-utility-dimensions).
Because this library not yet comprehensively covers all aspects of response quality in retrieval-augmented generation, we also provide a [list of "missing" axioms and ideas](#missing-axioms-and-axiom-ideas).

## Utility Dimensions

Several frameworks describe utility dimensions of retrieval-augmented generation.
The following list puts each into context with [the overall categorization](https://doi.org/10.1145/3626772.3657849) we used for our axioms [@gienapp2024evaluating].

- Gienapp et al. [@gienapp2024evaluating]:
    - coherence
    - coverage
    - consistency
    - correctness
    - clarity
    - (overall quality)
- Es et al. ([RAGAs](https://github.com/explodinggradients/ragas)) [@es2024ragas]:
    - [context precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/) (retrieval-focused)
    - [context recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/) (retrieval-focused)
    - [context entities recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_entities_recall/) (retrieval-focused)
    - [noise sensitivity](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/noise_sensitivity/) (≃ factual correctness)
    - [answer relevance / response relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/) (≃ deep coverage)
    - [faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) (≃ external consistency)
    - context relevance (≃ broad coverage)
    - [multi-modal faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/multi_modal_faithfulness/) (only textual context considered yet; submit a [pull request](https://github.com/webis-de/ir_axioms/compare))
    - [multi-modal relevance](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/multi_modal_relevance/) (only textual context considered yet; submit a [pull request](https://github.com/webis-de/ir_axioms/compare))
- Hashemi et al. (LLM-Rubric) [@hashemi2024llmrubric]:
    - satisfaction
    - naturalness (≃ language clarity / content clarity)
    - grounding sources (retrieval-focused)
    - citation presence (≃ factual correctness)
    - citation suitability (≃ factual correctness)
    - citation optimality (≃ factual correctness / external consistency)
    - redundancy (≃ broad coverage)
    - conciseness (≃ content clarity)
    - efficiency (architecture-specific)
- Saad-Falcon et al. (ARES) [@saadfalcon2024ares]:
    - context relevance (retrieval-related)
    - answer faithfulness (≃ external consistency)
    - answer relevance (≃ deep coverage)
- Wallat et al. [@wallat2025correctness]:
    - citation correctness (≃ factual correctness)
    - citation faithfulness (≃ factual correctness)
- Rosset et al. [@rosset2023axiomatic]:
    - usefulness (≃ overall quality)
    - relevance (≃ deep coverage / topical correctness)
    - groundedness (≃ factual correctness)
    - truthfulness (≃ external consistency)
    - relevant/irrelevant grounding (≃ external consistency / factual correctness)
    - thoroughness (≃ broad coverage)

## Axioms for Utility Dimensions

The following list matches our new proposed axioms to the utility aspects they address as per the categorization by Gienapp et al. (see above).
All axioms accept a `margin_fraction` parameter: values within this relative margin are considered equal, and no preference is expressed.

| Axiom | Class | Intuition | Dimension | Default margin |
|:--|:--|:--|:--|--:|
| <span class="axiom" id="coh1">COH1</span> | `WordLengthDeviationCoherenceAxiom` | Prefer less variance in average word length across sentences. | coherence | 0.0 |
| <span class="axiom" id="coh2">COH2</span> | `SubjectVerbClosenessCoherenceAxiom` | Prefer the response with subject–verb pairs closer together. | coherence | 0.1 |
| <span class="axiom" id="cov1">COV1</span> | `AspectCountCoverageAxiom` | Prefer the response containing more extracted aspects. | coverage | 0.0 |
| <span class="axiom" id="cov2">COV2</span> | `AspectRedundancyCoverageAxiom` | Prefer the response with less redundant extracted aspects. | coverage | 0.2 |
| <span class="axiom" id="cov3">COV3</span> | `AspectSimilaritySentenceCountCoverageAxiom` | Prefer if more sentences cover aspects from the query. | coverage | 0.5 |
| <span class="axiom" id="cons1">CONS1</span> | `AspectSimilaritySentenceCountConsistencyAxiom` | Prefer if more sentences cover aspects from the context. | consistency | 0.5 |
| <span class="axiom" id="cons2">CONS2</span> | `RougeConsistencyAxiom` | Prefer the response with higher textual overlap (ROUGE [@lin2004rouge]) with the contexts. | consistency | 0.1 |
| <span class="axiom" id="cons3">CONS3</span> | `EntityContradictionConsistencyAxiom` | Penalize entities mentioned in contradictory phrases. | consistency | 0.2 |
| <span class="axiom" id="corr1">CORR1</span> | `CitationSentenceCorrectnessAxiom` | Prefer the response with more sentences containing citations. | correctness | 0.1 |
| <span class="axiom" id="clar1">CLAR1</span> | `LanguageToolGrammarErrorProportionClarityAxiom` | Prefer a lower text proportion covered by grammar errors ([LanguageTool](https://languagetool.org/)). | clarity | — |
| <span class="axiom" id="clar2">CLAR2</span> | `FleschReadingEaseClarityAxiom` | Prefer the more readable response (Flesch reading ease [@flesch1948new]). | clarity | 0.0 |

The source code of the axioms is located in [`ir_axioms/axiom/generation/`](https://github.com/webis-de/ir_axioms/tree/main/ir_axioms/axiom/generation).
The aspect-based axioms (COV1–COV3, CONS1) use the configured `AspectExtraction` tool, which extracts spaCy noun chunks by default.
In our ICTIR 2025 experiments, we used spaCy entities instead:

```python
from ir_axioms.axiom import COV1
from ir_axioms.tools import SpacyEntitiesAspectExtraction

COV1(aspect_extraction=SpacyEntitiesAspectExtraction(), margin_fraction=0.5)
```

## Adapted Retrieval Axioms

Many [retrieval axioms](retrieval.md) can also be applied to generated responses, by treating the request as the query and each response as a document.
The `GEN_*` axioms wrap the corresponding retrieval axiom accordingly:

| Axiom | Class | Wrapped retrieval axiom |
|:--|:--|:--|
| <span class="axiom">GEN_TFC1</span> | `GenerativeTfc1Axiom` | [TFC1](retrieval.md#tfc1) |
| <span class="axiom">GEN_LNC1</span> | `GenerativeLnc1Axiom` | [LNC1](retrieval.md#lnc1) |
| <span class="axiom">GEN_TF_LNC</span> | `GenerativeTfLncAxiom` | [TF-LNC](retrieval.md#tf-lnc) |
| <span class="axiom">GEN_REG</span> | `GenerativeRegAxiom` | [REG](retrieval.md#reg) |
| <span class="axiom">GEN_ANTI_REG</span> | `GenerativeAntiRegAxiom` | [ANTI-REG](retrieval.md#reg) |
| <span class="axiom">GEN_AND</span> | `GenerativeAndAxiom` | [AND](retrieval.md#and) |
| <span class="axiom">GEN_LEN_AND</span> | `GenerativeLenAndAxiom` | [LEN-AND](retrieval.md#and) |
| <span class="axiom">GEN_M_AND</span> | `GenerativeModifiedAndAxiom` | [M-AND](retrieval.md#and) |
| <span class="axiom">GEN_DIV</span> | `GenerativeDivAxiom` | [DIV](retrieval.md#div) |
| <span class="axiom">GEN_LEN_DIV</span> | `GenerativeLenDivAxiom` | [LEN-DIV](retrieval.md#div) |
| <span class="axiom">GEN_STMC1</span> | `GenerativeStmc1Axiom` | [STMC1](retrieval.md#stmc1) |
| <span class="axiom">GEN_STMC2</span> | `GenerativeStmc2Axiom` | [STMC2](retrieval.md#stmc2) |
| <span class="axiom">GEN_PROX1</span> … <span class="axiom">GEN_PROX5</span> | `GenerativeProx1Axiom` … `GenerativeProx5Axiom` | [PROX1](retrieval.md#prox1) … [PROX5](retrieval.md#prox5) |
| <span class="axiom">GEN_ArgUC</span> | `GenerativeArgumentativeUnitsCountAxiom` | [ArgUC](retrieval.md#arguc) |
| <span class="axiom">GEN_QTArg</span> | `GenerativeQueryTermOccurrenceInArgumentativeUnitsAxiom` | [QTArg](retrieval.md#qtarg) |
| <span class="axiom">GEN_QTPArg</span> | `GenerativeQueryTermPositionInArgumentativeUnitsAxiom` | [QTPArg](retrieval.md#qtparg) |
| <span class="axiom">GEN_aSL</span> | `GenerativeAverageSentenceLengthAxiom` | [aSL](retrieval.md#asl) |

Retrieval axioms that need retrieval scores or collection statistics (e.g., LB1, TFC3, M-TDC) have no generative counterpart.

## Oracle and Reference Axioms

To evaluate RAG axioms, `ir_axioms` includes "pseudo-axioms" that derive preferences from human or LLM judgments:

| Class | Preferences derived from |
|:--|:--|
| `TrecRagNuggetAxiom` | Nugget assignments of the TREC 2024 RAG track [@pradeep2024initial] (`score_type`: `"all"`, `"vital"`, or `"weighted"`). |
| `TrecRagCrowdAxiom` | Crowdsourced utility ratings of RAG responses (`utility_type`: e.g., `"overall"`, `"coherence"`, `"broad coverage"`). |
| `TrecRagLlmOrigAxiom` | Response ratings of an LLM (ORIG-like baseline for RAG). |

These axioms read their judgments from local files; see the [RAG evaluation walkthrough](../walkthroughs/rag-evaluation.md) for an example.

## Missing Axioms and Axiom Ideas

The [above axioms](#axioms-for-utility-dimensions) not yet completely cover all individual aspects of response quality.
The below list contains ideas on how to extend the set of axioms further:

- Factual-grounding-based axioms (main obstacle: fact checking often uses opaque models)
- Axioms from the viewpoint of the query, like EXAM/RUBRIC
- Content clarity axioms based on discourse structure
- More aspects of stylistic coherence
- Penalize too many tense switches
- Prefer response with easier co-reference resolution
- Consider other citation format or URLs for citation count
- Axioms for multi-modal outputs (e.g., generated images)

We are happy to collaborate to get more diverse axioms implemented.
Please [create a pull request](https://github.com/webis-de/ir_axioms/compare) to propose new ideas and refer to our [guide on axiom development](../development/axiom-development.md) to realize an idea as code.
