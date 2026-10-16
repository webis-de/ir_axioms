# Axioms

<p class="subtitle">Catalog of all axioms implemented in <code>ir_axioms</code></p>

`ir_axioms` implements [retrieval axioms](retrieval.md) that compare documents for a query, [RAG axioms](generation.md) that compare generated responses for a request, and [utility axioms](utility.md) that serve as reference points and building blocks.
For a broader overview of axiomatic thinking in information retrieval, including constraints that are not (yet) implemented here, see [related axioms](related.md) from the literature.

Each axiom implementation is tested, featuring an extensive test suite of over 400 unit tests.

To learn how to compute preferences with these axioms and combine them, see the [quickstart](../getting-started/quickstart.md) and [core concepts](../getting-started/concepts.md); for the full API, see the [API reference](../api/axiom.md).

All axioms can be imported from `ir_axioms.axiom`, either by their class name (e.g., [`Tfc1Axiom`][ir_axioms.axiom.retrieval.term_frequency.Tfc1Axiom]) or by their short alias (e.g., `TFC1`).

## Retrieval Axioms

| Axiom | Intuition | Objective | Source |
|:--|:--|:--|:--|
| [TFC1](retrieval.md#tfc1) | Prefer documents with more query term occurrences. | Term frequency | [@fang2004formal] |
| [TFC3](retrieval.md#tfc3) | Prefer documents with occurrences of more query terms. | Term frequency | [@fang2011diagnostic] |
| [M-TDC](retrieval.md#m-tdc), LEN-M-TDC | Prefer documents with more discriminative query terms. | Term frequency | [@shi2005gravitation] |
| [LNC1](retrieval.md#lnc1) | Penalize longer documents for non-relevant terms. | Document length | [@fang2004formal] |
| [TF-LNC](retrieval.md#tf-lnc) | Reward additional query terms more than document length is penalized. | Document length | [@fang2004formal] |
| [LB1](retrieval.md#lb1) | Do not override the term presence–absence gap with length normalization. | Lower-bounding TF | [@lv2011lower] |
| [REG](retrieval.md#reg), ANTI-REG, ASPECT-REG | Prefer documents covering more different query aspects. | Query aspects | [@zheng2010query] |
| [AND](retrieval.md#and), LEN-AND, M-AND, LEN-M-AND | Prefer documents containing all query terms. | Query aspects | [@wu2012relation] |
| [DIV](retrieval.md#div), LEN-DIV | Prefer documents that are less redundant with the query. | Query aspects | [@gollapudi2009axiomatic] |
| [STMC1](retrieval.md#stmc1) | Prefer documents with terms more similar to query terms. | Semantic similarity | [@fang2006semantic] |
| [STMC2](retrieval.md#stmc2) | Do not reward similar terms more than exact matches. | Semantic similarity | [@fang2006semantic] |
| [PROX1](retrieval.md#prox1) | Prefer documents with shorter distance between query term pairs. | Term proximity | [@hagen2016axiomatic] |
| [PROX2](retrieval.md#prox2) | Prefer documents with earlier query term occurrences. | Term proximity | [@hagen2016axiomatic] |
| [PROX3](retrieval.md#prox3) | Prefer documents where the query occurs earlier as a phrase. | Term proximity | [@hagen2016axiomatic] |
| [PROX4](retrieval.md#prox4) | Prefer documents that contain all query terms in a shorter substring. | Term proximity | [@hagen2016axiomatic] |
| [PROX5](retrieval.md#prox5) | Prefer documents where the query terms are closer together on average. | Term proximity | [@hagen2016axiomatic] |
| [ArgUC](retrieval.md#arguc) | Prefer more argumentative units. | Argumentativeness | [@bondarenko2018webis; @bondarenko2019webis] |
| [QTArg](retrieval.md#qtarg) | Prefer more query terms in argumentative units. | Argumentativeness | [@bondarenko2018webis; @bondarenko2019webis] |
| [QTPArg](retrieval.md#qtparg) | Prefer earlier query terms in argumentative units. | Argumentativeness | [@bondarenko2018webis; @bondarenko2019webis] |
| [aSL](retrieval.md#asl) | Prefer passages with 12–20 words per sentence. | Argumentativeness | [@bondarenko2018webis; @bondarenko2019webis] |

Not implemented, as they do not induce practically usable pairwise preferences: [TFC2](retrieval.md#tfc2), [TDC](retrieval.md#m-tdc), [LNC2](retrieval.md#lnc2), and [LB2](retrieval.md#lb2).

## RAG Axioms

| Axiom | Intuition | Dimension | Source |
|:--|:--|:--|:--|
| [COH1](generation.md#coh1) | Prefer responses with less variance in avg. word length across sentences. | Coherence | [@merker2025axioms] |
| [COH2](generation.md#coh2) | Prefer responses with subject–verb pairs closer together. | Coherence | [@merker2025axioms] |
| [COV1](generation.md#cov1) | Prefer responses with more extracted aspects. | Coverage | [@merker2025axioms] |
| [COV2](generation.md#cov2) | Prefer responses with less redundant extracted aspects. | Coverage | [@merker2025axioms] |
| [COV3](generation.md#cov3) | Prefer responses with more sentences covering aspects from the query. | Coverage | [@merker2025axioms] |
| [CONS1](generation.md#cons1) | Prefer responses with more sentences covering aspects from the context. | Consistency | [@merker2025axioms] |
| [CONS2](generation.md#cons2) | Prefer responses with higher text overlap with contexts. | Consistency | [@merker2025axioms] |
| [CONS3](generation.md#cons3) | Penalize aspects mentioned in contradictory phrases. | Consistency | [@merker2025axioms] |
| [CORR1](generation.md#corr1) | Prefer responses with more sentences referencing sources. | Correctness | [@merker2025axioms] |
| [CLAR1](generation.md#clar1) | Prefer responses with fewer grammar errors. | Clarity | [@merker2025axioms] |
| [CLAR2](generation.md#clar2) | Prefer responses with better readability. | Clarity | [@merker2025axioms] |
| [GEN_*](generation.md#adapted-retrieval-axioms) | Retrieval axioms applied to generated responses. | — | [@merker2025axioms] |

## Utility Axioms

| Axiom | Description | Source |
|:--|:--|:--|
| [ORIG](utility.md#orig) | Follow the original ranking (ranks or scores). | [@hagen2016axiomatic] |
| [ORACLE](utility.md#oracle) | Follow human relevance judgments. | [@bondarenko2022axiomatic] |
| [NOP](utility.md#nop), [RANDOM](utility.md#random), [GT](utility.md#gt), [LT](utility.md#lt), [DIST](utility.md#dist) | Baselines and helpers. | — |
| [Arithmetic axioms](utility.md#arithmetic-axioms) | Sum, product, vote, conjunction, cascade, negation, normalization. | — |
