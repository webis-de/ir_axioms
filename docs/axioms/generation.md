# RAG Axioms

<p class="subtitle">Constraints on generated responses for a request</p>

RAG axioms compare two generated responses \(r_1\), \(r_2\) for the same request \(q\) (optionally including the retrieved context \(C\)) and implement the interface `Axiom[GenerationRequest, GenerationResponse]` (see the [data model](../getting-started/concepts.md#axioms-and-preferences)).
They were introduced in our ICTIR 2025 paper [@merker2025axioms] and address the five utility dimensions of generated responses proposed by Gienapp et al. [@gienapp2024evaluating]: [coherence](#coherence), [coverage](#coverage), [consistency](#consistency), [correctness](#correctness), and [clarity](#clarity).
Below, each axiom is listed with its intuition, followed by how it is implemented in `ir_axioms`.

The formalizations follow the paper, where, e.g., \(>_{10\%}\) means that only a difference of more than 10 % counts as "larger".
In the implementation, this margin is the `margin_fraction` parameter of each axiom: if the compared values of both responses are within this relative margin, the axiom expresses no preference.
The aspect-based axioms use the configured [`AspectExtraction` tool](../getting-started/concepts.md#tools-and-dependency-injection), which extracts spaCy noun chunks by default.
In our ICTIR 2025 experiments, we used spaCy entities instead:

```python
from ir_axioms.axiom import COV1
from ir_axioms.tools import SpacyEntitiesAspectExtraction

COV1(aspect_extraction=SpacyEntitiesAspectExtraction(), margin_fraction=0.5)
```

To evaluate the axioms against human judgments, follow the [RAG evaluation walkthrough](../walkthroughs/rag-evaluation.md).
The [API reference](../api/axiom.md#rag-axioms) documents all classes and parameters.

## Coherence

_Logical coherence:_ Is the response well-structured?
_Stylistic coherence:_ Does the response have a uniform style of speech?

### COH1 { #coh1 }

<span class="axiom">COH1</span> [`WordLengthDeviationCoherenceAxiom`][ir_axioms.axiom.generation.coherence.WordLengthDeviationCoherenceAxiom] — _"Prefer responses with less variance in avg. word length across sentences."_ [@merker2025axioms]

Formalization
:   \(\sigma(\mathrm{SWAWL}_{r_1}) < \sigma(\mathrm{SWAWL}_{r_2}) \Rightarrow r_1 >_\text{COH1} r_2\), where \(\mathrm{SWAWL}_r\) is the set of the average word lengths of the sentences in \(r\).

Implementation
:   Computes the average word length of each sentence and prefers the response with the lower standard deviation of these averages, as a proxy for a uniform style.

Parameters
:   `margin_fraction=0.0`.

Tools
:   [`SentenceTokenizer`][ir_axioms.tools.tokenizer.SentenceTokenizer], [`TermTokenizer`][ir_axioms.tools.tokenizer.TermTokenizer].

### COH2 { #coh2 }

<span class="axiom">COH2</span> [`SubjectVerbClosenessCoherenceAxiom`][ir_axioms.axiom.generation.coherence.SubjectVerbClosenessCoherenceAxiom] — _"Prefer responses with subject–verb pairs closer together."_ [@merker2025axioms]

Formalization
:   \(\text{avg-SV-dist}(r_1) <_{10\%} \text{avg-SV-dist}(r_2) \Rightarrow r_1 >_\text{COH2} r_2\), where \(\text{avg-SV-dist}(r) = \frac{1}{|\mathrm{SVO}_r|} \sum_{(\mathit{sub}, \mathit{ver}, \mathit{obj}) \in \mathrm{SVO}_r} |\mathrm{pos}(\mathit{sub}) - \mathrm{pos}(\mathit{ver})|\) for the subject–verb–object triples \(\mathrm{SVO}_r\) in \(r\).

Implementation
:   Extracts subject–verb–object triples with spaCy and prefers the response with the lower average (maximum) distance between subjects and their verbs, as a proxy for easy-to-follow sentence structures.

Parameters
:   `margin_fraction=0.1`, `language_name="en_core_web_sm"` (spaCy model).

## Coverage

_Broad coverage:_ Does the response cover diverse information?
_Deep coverage:_ Does the response provide in-depth and highly informative content?

### COV1 { #cov1 }

<span class="axiom">COV1</span> [`AspectCountCoverageAxiom`][ir_axioms.axiom.generation.coverage.AspectCountCoverageAxiom] — _"Prefer responses with more extracted aspects."_ [@merker2025axioms]

Formalization
:   \(|\mathrm{ASP}_{r_1}| > |\mathrm{ASP}_{r_2}| \Rightarrow r_1 >_\text{COV1} r_2\), where \(\mathrm{ASP}_r\) is the set of aspects extracted from \(r\).

Implementation
:   Extracts the aspects of each response and prefers the response with more distinct aspects.

Parameters
:   `margin_fraction=0.0`.

Tools
:   [`AspectExtraction`][ir_axioms.tools.aspects.AspectExtraction].

### COV2 { #cov2 }

<span class="axiom">COV2</span> [`AspectRedundancyCoverageAxiom`][ir_axioms.axiom.generation.coverage.AspectRedundancyCoverageAxiom] — _"Prefer responses with less redundant extracted aspects."_ [@merker2025axioms]

Formalization
:   \(\text{avg-ASP-sim}(r_1) <_{20\%} \text{avg-ASP-sim}(r_2) \Rightarrow r_1 >_\text{COV2} r_2\), where \(\text{avg-ASP-sim}(r) = \frac{1}{|\mathrm{ASP}_r|^2} \sum_{a, a' \in \mathrm{ASP}_r} \mathrm{sim}(a, a')\).

Implementation
:   Extracts the aspects of each response and prefers the response with the lower mean pairwise similarity between its aspects.

Parameters
:   `margin_fraction=0.2`.

Tools
:   [`AspectExtraction`][ir_axioms.tools.aspects.AspectExtraction], [`SentenceSimilarity`][ir_axioms.tools.similarity.SentenceSimilarity].

### COV3 { #cov3 }

<span class="axiom">COV3</span> [`AspectSimilaritySentenceCountCoverageAxiom`][ir_axioms.axiom.generation.coverage.AspectSimilaritySentenceCountCoverageAxiom] — _"Prefer responses with more sentences covering aspects from the query."_ [@merker2025axioms]

Formalization
:   \(\text{avg-ASP}_q\text{-cov}(r_1) >_{50\%} \text{avg-ASP}_q\text{-cov}(r_2) \Rightarrow r_1 >_\text{COV3} r_2\), where \(\text{avg-ASP}_q\text{-cov}(r) = \frac{1}{|\mathrm{ASP}_q|} \sum_{a \in \mathrm{ASP}_q} \sum_{s \in r} \mathrm{sim}(a, s)\) for the sentences \(s\) of \(r\).

Implementation
:   Extracts the aspects of the request text and counts the response's sentences covering them, weighting each sentence by its mean similarity to the aspects.
    Prefers the response with the higher weighted count.

Parameters
:   `margin_fraction=0.5`.

Tools
:   [`AspectExtraction`][ir_axioms.tools.aspects.AspectExtraction], [`SentenceTokenizer`][ir_axioms.tools.tokenizer.SentenceTokenizer], [`SentenceSimilarity`][ir_axioms.tools.similarity.SentenceSimilarity].

## Consistency

_External consistency:_ Does the response accurately convey the information from the sources?
_Internal consistency:_ Does the response not contradict itself?

### CONS1 { #cons1 }

<span class="axiom">CONS1</span> [`AspectSimilaritySentenceCountConsistencyAxiom`][ir_axioms.axiom.generation.consistency.AspectSimilaritySentenceCountConsistencyAxiom] — _"Prefer responses with more sentences covering aspects from the context."_ [@merker2025axioms]

Formalization
:   \(\text{avg-ASP}_C\text{-cov}(r_1) >_{50\%} \text{avg-ASP}_C\text{-cov}(r_2) \Rightarrow r_1 >_\text{CONS1} r_2\), where \(\text{avg-ASP}_C\text{-cov}(r) = \frac{1}{|\mathrm{ASP}_C|} \sum_{a \in \mathrm{ASP}_C} \sum_{s \in r} \mathrm{sim}(a, s)\) for the aspects \(\mathrm{ASP}_C\) of the context documents \(C\).

Implementation
:   Like [COV3](#cov3), but extracts the aspects from the retrieved context \(C\) instead of the request text.
    Expresses no preference if the request has no context.

Parameters
:   `margin_fraction=0.5`.

Tools
:   [`AspectExtraction`][ir_axioms.tools.aspects.AspectExtraction], [`SentenceTokenizer`][ir_axioms.tools.tokenizer.SentenceTokenizer], [`SentenceSimilarity`][ir_axioms.tools.similarity.SentenceSimilarity].

### CONS2 { #cons2 }

<span class="axiom">CONS2</span> [`RougeConsistencyAxiom`][ir_axioms.axiom.generation.consistency.RougeConsistencyAxiom] — _"Prefer responses with higher text overlap with contexts."_ [@merker2025axioms]

Formalization
:   \(\text{ROUGE-L}(r_1, c') >_{10\%} \text{ROUGE-L}(r_2, c') \Rightarrow r_1 >_\text{CONS2} r_2\), where \(c'\) is the concatenated text of the context documents \(C\).

Implementation
:   Prefers the response with the higher ROUGE-Lsum F-measure [@lin2004rouge] against the concatenated retrieved context.
    Expresses no preference if the request has no context.

Parameters
:   `margin_fraction=0.1`.

### CONS3 { #cons3 }

<span class="axiom">CONS3</span> [`EntityContradictionConsistencyAxiom`][ir_axioms.axiom.generation.consistency.EntityContradictionConsistencyAxiom] — _"Penalize aspects mentioned in contradictory phrases."_ [@merker2025axioms]

Formalization
:   \(\frac{|\mathrm{ASP}^E_{r_1}|}{|\mathrm{ASP}_{r_1}|} <_{20\%} \frac{|\mathrm{ASP}^E_{r_2}|}{|\mathrm{ASP}_{r_2}|} \Rightarrow r_1 >_\text{CONS3} r_2\), where \(\mathrm{ASP}^E_r\) are the aspects of \(r\) that appear in both negated and non-negated passages of \(r\).

Implementation
:   Detects named entities and their negation (with [negspaCy](https://github.com/jenojp/negspacy)), counts entities that occur both negated and non-negated, and prefers the response with the lower proportion of such contradictory entities.

Parameters
:   `margin_fraction=0.2`, `language_name="en_core_web_sm"` (spaCy model).

## Correctness

_Factuality:_ Does the response state things that are verifiable?
_Topical correctness:_ Does the response state things within the scope of the user's information need?

### CORR1 { #corr1 }

<span class="axiom">CORR1</span> [`CitationSentenceCorrectnessAxiom`][ir_axioms.axiom.generation.correctness.CitationSentenceCorrectnessAxiom] — _"Prefer responses with more sentences referencing sources."_ [@merker2025axioms]

Formalization
:   \(\frac{|S^\text{ref}_{r_1}|}{|S_{r_1}|} >_{10\%} \frac{|S^\text{ref}_{r_2}|}{|S_{r_2}|} \Rightarrow r_1 >_\text{CORR1} r_2\), where \(S_r\) are the sentences of \(r\) and \(S^\text{ref}_r \subseteq S_r\) those containing a reference.

Implementation
:   Prefers the response with the higher proportion of sentences that contain a citation of the form `[1]`.

Parameters
:   `margin_fraction=0.1`.

Tools
:   [`SentenceTokenizer`][ir_axioms.tools.tokenizer.SentenceTokenizer].

## Clarity

_Language clarity:_ Is the response concise, comprehensible, lexically and grammatically correct, and accessible?
_Content clarity:_ Does the response clearly communicate the most salient information and where it stems from?

### CLAR1 { #clar1 }

<span class="axiom">CLAR1</span> [`LanguageToolGrammarErrorProportionClarityAxiom`][ir_axioms.axiom.generation.clarity.LanguageToolGrammarErrorProportionClarityAxiom] — _"Prefer responses with fewer grammar errors."_ [@merker2025axioms]

Formalization
:   \(\frac{\mathrm{length}(\mathrm{ERR}_{r_1})}{\mathrm{length}(r_1)} <_{10\%} \frac{\mathrm{length}(\mathrm{ERR}_{r_2})}{\mathrm{length}(r_2)} \Rightarrow r_1 >_\text{CLAR1} r_2\), where \(\mathrm{ERR}_r\) are the spans of grammatical errors in \(r\).

Implementation
:   Checks the response with [LanguageTool](https://languagetool.org/) (grammar rules only) and prefers the response with the lower proportion of characters covered by errors.

Parameters
:   `margin_fraction=0.1`, `language="en-US"`.

### CLAR2 { #clar2 }

<span class="axiom">CLAR2</span> [`FleschReadingEaseClarityAxiom`][ir_axioms.axiom.generation.clarity.FleschReadingEaseClarityAxiom] — _"Prefer responses with better readability."_ [@merker2025axioms]

Formalization
:   \(\text{Flesch}(r_1) < \text{Flesch}(r_2) \Rightarrow r_1 >_\text{CLAR2} r_2\) (Eq. 11 in the paper).

Implementation
:   Compares the Flesch reading ease scores [@flesch1948new] of the responses.

Parameters
:   `margin_fraction=0.0`, `language_name="en_core_web_sm"` (spaCy model).

!!! bug "Known issue"

    Both the formalization in the paper (Eq. 11) and the implementation prefer the response with the _lower_ Flesch reading ease score.
    As higher Flesch scores indicate _easier_ text, this prefers the less readable response, contradicting the intuition (see the [roadmap](../development/roadmap.md#known-bugs)).

## Adapted Retrieval Axioms

Many [retrieval axioms](retrieval.md) can also be applied to generated responses, by treating the request as the query and each response as a document [@merker2025axioms].
The `GEN_*` axioms wrap the corresponding retrieval axiom accordingly:

| Axiom | Class | Wrapped retrieval axiom |
|:--|:--|:--|
| <span class="axiom">GEN_TFC1</span> | [`GenerativeTfc1Axiom`][ir_axioms.axiom.generation.retrieval.GenerativeTfc1Axiom] | [TFC1](retrieval.md#tfc1) |
| <span class="axiom">GEN_LNC1</span> | [`GenerativeLnc1Axiom`][ir_axioms.axiom.generation.retrieval.GenerativeLnc1Axiom] | [LNC1](retrieval.md#lnc1) |
| <span class="axiom">GEN_TF_LNC</span> | [`GenerativeTfLncAxiom`][ir_axioms.axiom.generation.retrieval.GenerativeTfLncAxiom] | [TF-LNC](retrieval.md#tf-lnc) |
| <span class="axiom">GEN_REG</span> | [`GenerativeRegAxiom`][ir_axioms.axiom.generation.retrieval.GenerativeRegAxiom] | [REG](retrieval.md#reg) |
| <span class="axiom">GEN_ANTI_REG</span> | [`GenerativeAntiRegAxiom`][ir_axioms.axiom.generation.retrieval.GenerativeAntiRegAxiom] | [ANTI-REG](retrieval.md#reg) |
| <span class="axiom">GEN_AND</span> | [`GenerativeAndAxiom`][ir_axioms.axiom.generation.retrieval.GenerativeAndAxiom] | [AND](retrieval.md#and) |
| <span class="axiom">GEN_LEN_AND</span> | [`GenerativeLenAndAxiom`][ir_axioms.axiom.generation.retrieval.GenerativeLenAndAxiom] | [LEN-AND](retrieval.md#and) |
| <span class="axiom">GEN_M_AND</span> | [`GenerativeModifiedAndAxiom`][ir_axioms.axiom.generation.retrieval.GenerativeModifiedAndAxiom] | [M-AND](retrieval.md#and) |
| <span class="axiom">GEN_DIV</span> | [`GenerativeDivAxiom`][ir_axioms.axiom.generation.retrieval.GenerativeDivAxiom] | [DIV](retrieval.md#div) |
| <span class="axiom">GEN_LEN_DIV</span> | [`GenerativeLenDivAxiom`][ir_axioms.axiom.generation.retrieval.GenerativeLenDivAxiom] | [LEN-DIV](retrieval.md#div) |
| <span class="axiom">GEN_STMC1</span> | [`GenerativeStmc1Axiom`][ir_axioms.axiom.generation.retrieval.GenerativeStmc1Axiom] | [STMC1](retrieval.md#stmc1) |
| <span class="axiom">GEN_STMC2</span> | [`GenerativeStmc2Axiom`][ir_axioms.axiom.generation.retrieval.GenerativeStmc2Axiom] | [STMC2](retrieval.md#stmc2) |
| <span class="axiom">GEN_PROX1</span> | [`GenerativeProx1Axiom`][ir_axioms.axiom.generation.retrieval.GenerativeProx1Axiom] | [PROX1](retrieval.md#prox1) |
| <span class="axiom">GEN_PROX2</span> | [`GenerativeProx2Axiom`][ir_axioms.axiom.generation.retrieval.GenerativeProx2Axiom] | [PROX2](retrieval.md#prox2) |
| <span class="axiom">GEN_PROX3</span> | [`GenerativeProx3Axiom`][ir_axioms.axiom.generation.retrieval.GenerativeProx3Axiom] | [PROX3](retrieval.md#prox3) |
| <span class="axiom">GEN_PROX4</span> | [`GenerativeProx4Axiom`][ir_axioms.axiom.generation.retrieval.GenerativeProx4Axiom] | [PROX4](retrieval.md#prox4) |
| <span class="axiom">GEN_PROX5</span> | [`GenerativeProx5Axiom`][ir_axioms.axiom.generation.retrieval.GenerativeProx5Axiom] | [PROX5](retrieval.md#prox5) |
| <span class="axiom">GEN_ArgUC</span> | [`GenerativeArgumentativeUnitsCountAxiom`][ir_axioms.axiom.generation.retrieval.GenerativeArgumentativeUnitsCountAxiom] | [ArgUC](retrieval.md#arguc) |
| <span class="axiom">GEN_QTArg</span> | [`GenerativeQueryTermOccurrenceInArgumentativeUnitsAxiom`][ir_axioms.axiom.generation.retrieval.GenerativeQueryTermOccurrenceInArgumentativeUnitsAxiom] | [QTArg](retrieval.md#qtarg) |
| <span class="axiom">GEN_QTPArg</span> | [`GenerativeQueryTermPositionInArgumentativeUnitsAxiom`][ir_axioms.axiom.generation.retrieval.GenerativeQueryTermPositionInArgumentativeUnitsAxiom] | [QTPArg](retrieval.md#qtparg) |
| <span class="axiom">GEN_aSL</span> | [`GenerativeAverageSentenceLengthAxiom`][ir_axioms.axiom.generation.retrieval.GenerativeAverageSentenceLengthAxiom] | [aSL](retrieval.md#asl) |

Retrieval axioms that need retrieval scores or collection statistics (e.g., LB1, TFC3, M-TDC) have no generative counterpart.

## Oracle Axioms

To evaluate RAG axioms, `ir_axioms` includes pseudo-axioms that derive preferences from human or LLM judgments (similar to [ORACLE](utility.md#oracle) and [ORIG](utility.md#orig) for retrieval).
They read their judgments from local files, e.g., from the [`data/`](https://github.com/webis-de/ir_axioms/tree/main/data) directory of the repository.

### Nugget Oracle { #trec-rag-nugget }

[`TrecRagNuggetAxiom`][ir_axioms.axiom.generation.trec.TrecRagNuggetAxiom] — _"Prefer the response that covers more information nuggets."_

Implementation
:   Derives preferences from the nugget assignments of the TREC 2024 RAG track [@pradeep2024initial].

Parameters
:   `assignments_path` (nugget assignments file), `score_type` (`"all"`, `"vital"`, or `"weighted"` nuggets), `strict=False`, `margin_fraction=0.0`.

### Crowd Oracle { #trec-rag-crowd }

[`TrecRagCrowdAxiom`][ir_axioms.axiom.generation.crowd.TrecRagCrowdAxiom] — _"Prefer the response that crowd workers judged more useful."_

Implementation
:   Derives preferences from crowdsourced pairwise utility judgments of RAG responses for a utility dimension.

Parameters
:   `responses_path`, `ratings_path`, `utility_type` (e.g., `"overall"`, `"coherence"`, `"broad coverage"`, `"topical correctness"`), `margin_fraction=0.1`.

### LLM Ratings { #trec-rag-llm }

[`TrecRagLlmOrigAxiom`][ir_axioms.axiom.generation.llm.TrecRagLlmOrigAxiom] — _"Prefer the response that an LLM rated higher."_

Implementation
:   Derives preferences from response ratings of an LLM, as an ORIG-like baseline for RAG.

Parameters
:   `ratings_path`, `margin_fraction=0.0`.

## Utility Dimensions in Related Work

Several frameworks describe utility dimensions of retrieval-augmented generation.
The following table puts each into context with the utility dimensions of Gienapp et al. [@gienapp2024evaluating] that we used for our axioms.

| Framework | Dimension | Corresponds to |
|:--|:--|:--|
| Gienapp et al. [@gienapp2024evaluating] | coherence, coverage, consistency, correctness, clarity | (same) |
| | overall quality | — |
| RAGAs [@es2024ragas] | [context precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/), [context recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/), [context entities recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_entities_recall/) | retrieval-focused |
| | [noise sensitivity](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/noise_sensitivity/) | ≃ factual correctness |
| | [answer relevance / response relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/) | ≃ deep coverage |
| | [faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) | ≃ external consistency |
| | context relevance | ≃ broad coverage |
| | [multi-modal faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/multi_modal_faithfulness/), [multi-modal relevance](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/multi_modal_relevance/) | only textual context considered yet |
| LLM-Rubric [@hashemi2024llmrubric] | satisfaction | — |
| | naturalness | ≃ language clarity / content clarity |
| | grounding sources | retrieval-focused |
| | citation presence, citation suitability | ≃ factual correctness |
| | citation optimality | ≃ factual correctness / external consistency |
| | redundancy | ≃ broad coverage |
| | conciseness | ≃ content clarity |
| | efficiency | architecture-specific |
| ARES [@saadfalcon2024ares] | context relevance | retrieval-focused |
| | answer faithfulness | ≃ external consistency |
| | answer relevance | ≃ deep coverage |
| Wallat et al. [@wallat2025correctness] | citation correctness, citation faithfulness | ≃ factual correctness |
| Rosset et al. [@rosset2023axiomatic] | usefulness | ≃ overall quality |
| | relevance | ≃ deep coverage / topical correctness |
| | groundedness | ≃ factual correctness |
| | truthfulness | ≃ external consistency |
| | relevant/irrelevant grounding | ≃ external consistency / factual correctness |
| | thoroughness | ≃ broad coverage |

## Missing Axioms and Axiom Ideas

The [above axioms](#coherence) do not yet completely cover all individual aspects of response quality.
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
See also the [roadmap](../development/roadmap.md#feature-ideas) and [related axioms](related.md) from the literature.
