# Retrieval Axioms

<p class="subtitle">Constraints on document rankings for a query</p>

Retrieval axioms compare two documents \(d_1\), \(d_2\) for a query \(q\) and implement the interface `Axiom[Query, Document]`.
The original formulations of most axioms are restricted to artificial settings, e.g., single-term queries or documents of exactly the same length.
Following Hagen et al. [@hagen2016axiomatic] and Völske et al. [@volske2021towards], the implementations in `ir_axioms` incorporate an axiom's formalized idea while also making it applicable in practical settings [@bondarenko2022axiomatic]:
they are reformulated to work for arbitrary queries, to express pairwise preferences, and they include parameters to relax or strengthen pre- and filter conditions (e.g., a relative `margin_fraction` within which values are considered equal).

The [API reference](../api/axiom.md#retrieval-axioms) documents all classes and parameters; the [integrations guide](../getting-started/integrations.md) shows how to apply the axioms to a retrieval index.

Below, each axiom is listed with its original intuition and formulation, followed by the changes made in `ir_axioms`.
In the formulations, \(\mathrm{tf}(t, d)\) denotes the frequency of term \(t\) in document \(d\), \(|d|\) the document length, \(\mathrm{idf}(t)\) the inverse document frequency of \(t\), and \(\mathrm{sim}(t, t')\) a term similarity.

!!! info "Axioms requiring a retrieval index"

    Axioms using collection statistics (TFC3, M-TDC, LEN-M-TDC, ASPECT-REG) require an index, provided by the [PyTerrier or Pyserini integration](../getting-started/integrations.md).
    LB1 requires retrieval scores, and ORIG and ORACLE require ranks/scores and relevance labels, respectively (see [utility axioms](utility.md)).

## Term Frequency

### TFC1 { #tfc1 }

<span class="axiom">TFC1</span> [`Tfc1Axiom`][ir_axioms.axiom.retrieval.term_frequency.Tfc1Axiom] — _"Prefer documents with more query term occurrences."_ [@fang2004formal]

Original formulation
:   Given \(q = \{t\}\) and \(d_1, d_2\) with \(|d_1| = |d_2|\): if \(\mathrm{tf}(t, d_1) > \mathrm{tf}(t, d_2)\), then \(d_1 >_\text{TFC1} d_2\).

Changes in `ir_axioms`
:   Term frequencies are summed over all query terms, to allow queries with more than one term.
    Term frequency sums within a relative margin are considered equal, i.e., the inequality requires at least a 10 % difference [@volske2021towards].

Parameters
:   `margin_fraction=0.1` (term frequency sum margin), `precondition=LEN()` (document length precondition with 10 % margin).

### TFC2 { #tfc2 }

<span class="axiom">TFC2</span> — _"Additional query term occurrences yield smaller retrieval score improvements."_ [@fang2011diagnostic]

**Not implemented**, because the axiom compares score differences of three documents and does not induce a pairwise preference.

### TFC3 { #tfc3 }

<span class="axiom">TFC3</span> [`Tfc3Axiom`][ir_axioms.axiom.retrieval.term_frequency.Tfc3Axiom] — _"Prefer documents with occurrences of more query terms."_ [@fang2011diagnostic]

Original formulation
:   Given \(q = \{t_1, t_2\}\) and \(d_1, d_2\) with \(|d_1| = |d_2|\) and \(\mathrm{idf}(t_1) = \mathrm{idf}(t_2)\):
    if \(\mathrm{tf}(t_1, d_2) = \mathrm{tf}(t_1, d_1) + \mathrm{tf}(t_2, d_1)\) and \(\mathrm{tf}(t_2, d_2) = 0\) and \(\mathrm{tf}(t_1, d_1) \neq 0\) and \(\mathrm{tf}(t_2, d_1) \neq 0\), then \(d_1 >_\text{TFC3} d_2\).

Changes in `ir_axioms`
:   Term discrimination uses the inverse document frequency with a relative margin.
    Preferences are counted over all pairs of query terms, to allow queries with more than two terms [@volske2021towards].

Parameters
:   `margin_fraction=0.1` (IDF margin), `precondition=LEN()`.

### TDC and M-TDC { #m-tdc }

<span class="axiom">M_TDC</span> [`ModifiedTdcAxiom`][ir_axioms.axiom.retrieval.term_frequency.ModifiedTdcAxiom] — _"Prefer documents with more discriminative query terms."_

Original formulation (TDC)
:   Given \(q = \{t_1, t_2\}\) and \(d_1, d_2\) with \(|d_1| = |d_2|\), let \(d\) be some document.
    If \(\mathrm{idf}(t_1) > \mathrm{idf}(t_2)\) and \(t_1 \in d_1\), \(t_2 \notin d_1\), \(t_1 \notin d_2\), \(t_2 \in d_2\), then \(d \circ d_1 >_\text{TDC} d \circ d_2\) [@fang2004formal; @fang2011diagnostic].

Changes in `ir_axioms`
:   The original TDC is not implemented.
    Instead, `ir_axioms` implements the modified TDC by Shi et al. [@shi2005gravitation], which compares documents with the same total query term frequency (but differently distributed over the query terms) and, for each pair of query terms with different IDF, gives a point to the document with more occurrences of the more discriminative term.
    The document with more points is preferred.

Variants
:   <span class="axiom">LEN_M_TDC</span> [`LenModifiedTdcAxiom`][ir_axioms.axiom.retrieval.term_frequency.LenModifiedTdcAxiom]: M-TDC with an additional document length precondition (`precondition=LEN()`).

## Document Length

### LNC1 { #lnc1 }

<span class="axiom">LNC1</span> [`Lnc1Axiom`][ir_axioms.axiom.retrieval.length_norm.Lnc1Axiom] — _"Penalize longer documents for non-relevant terms."_ [@fang2004formal]

Original formulation
:   Given \(q\) and \(d_1, d_2\), let \(t \notin q\) and \(t'\) denote some arbitrary terms.
    If \(\mathrm{tf}(t, d_1) + 1 = \mathrm{tf}(t, d_2)\) and \(\mathrm{tf}(t', d_1) = \mathrm{tf}(t', d_2)\) for all \(t' \neq t\), then \(d_1 \geq_\text{LNC1} d_2\).

Changes in `ir_axioms`
:   If all query terms occur equally often in both documents (within a relative margin), prefer the shorter document.
    The margin allows for relative term frequency implementations.

Parameters
:   `margin_fraction=0.1`.

### LNC2 { #lnc2 }

<span class="axiom">LNC2</span> — _"Do not prefer shorter documents when the matched query term ratio is the same."_ [@fang2004formal]

**Not implemented**, because the precondition (\(d_1\) being a \(c\)-fold concatenation of \(d_2\)) practically never holds for real documents.

### TF-LNC { #tf-lnc }

<span class="axiom">TF_LNC</span> [`TfLncAxiom`][ir_axioms.axiom.retrieval.length_norm.TfLncAxiom] — _"Reward additional query terms more than document length is penalized."_ [@fang2004formal]

Original formulation
:   Given \(q = \{t\}\) and \(d_1, d_2\): if \(\mathrm{tf}(t, d_1) > \mathrm{tf}(t, d_2)\) and \(|d_1| = |d_2| + \mathrm{tf}(t, d_1) - \mathrm{tf}(t, d_2)\), then \(d_1 >_\text{TF-LNC} d_2\).

Changes in `ir_axioms`
:   For each query term, if both documents have the same length when ignoring that term, the document with more occurrences of the term gets a point.
    Points are counted over all query terms, to allow queries with more terms.

## Lower-Bounding Term Frequency

### LB1 { #lb1 }

<span class="axiom">LB1</span> [`Lb1Axiom`][ir_axioms.axiom.retrieval.lower_bound.Lb1Axiom] — _"Do not override the term presence–absence gap with length normalization."_ [@lv2011lower]

Original formulation
:   Given \(t \in q\) and \(d_1, d_2\) with \(\mathrm{score}(q \setminus \{t\}, d_1) = \mathrm{score}(q \setminus \{t\}, d_2)\):
    if \(\mathrm{tf}(t, d_1) > 0\) and \(\mathrm{tf}(t, d_2) = 0\), then \(d_1 >_\text{LB1} d_2\).

Changes in `ir_axioms`
:   Following Hagen et al. [@hagen2016axiomatic], uses the documents' retrieval scores (with 10 % softening) instead of the scores without term \(t\).
    Prefers the document that contains a query term that the other document does not contain.

Parameters
:   `margin_fraction=0.1` (score margin). Requires documents with a `score`.

### LB2 { #lb2 }

<span class="axiom">LB2</span> — _"Repeated query term occurrence is less important than the first occurrence."_ [@lv2011lower]

**Not implemented**, because the axiom requires modifying documents (adding/removing terms) and does not induce a practical pairwise preference.

## Query Aspects

### REG { #reg }

<span class="axiom">REG</span> [`RegAxiom`][ir_axioms.axiom.retrieval.query_aspects.RegAxiom] — _"Prefer documents covering more different query aspects."_ [@zheng2010query]

Original formulation
:   Given \(q = \{t_1, t_2, t_3\}\) and \(d_1, d_2\) with \(|d_1| = |d_2|\), \(\mathrm{idf}(t_2) = \mathrm{idf}(t_3)\), \(\mathrm{sim}(t_1, t_2) > \mathrm{sim}(t_1, t_3)\), and \(\mathrm{sim}(t_1, t_2) > \mathrm{sim}(t_2, t_3)\):
    if \(\mathrm{tf}(t_1, d_1) = \mathrm{tf}(t_1, d_2) > 0\), \(\mathrm{tf}(t_3, d_1) = \mathrm{tf}(t_2, d_2) > 0\), and \(\mathrm{tf}(t_2, d_1) = \mathrm{tf}(t_3, d_2) = 0\), then \(d_1 >_\text{REG} d_2\).

Changes in `ir_axioms`
:   Determines the query term(s) with the minimum average similarity to the other query terms (i.e., the most "different" aspect) and prefers the document with more occurrences of these terms, as described by Völske et al. [@volske2021towards].
    The term similarity is configurable (WordNet synonym sets by default, fastText if installed).

Parameters
:   `precondition=LEN()`.

Variants
:   <span class="axiom">ANTI_REG</span> [`AntiRegAxiom`][ir_axioms.axiom.retrieval.query_aspects.AntiRegAxiom]: uses the query term(s) with the _maximum_ average similarity instead, as in the modification of REG by Hagen et al. [@hagen2016axiomatic].
:   <span class="axiom">ASPECT_REG</span> [`AspectRegAxiom`][ir_axioms.axiom.retrieval.query_aspects.AspectRegAxiom]: follows the query aspect clustering of Zheng and Fang [@zheng2010query] (requires equally discriminative query terms) and prefers documents covering at least one of the query aspects.

### AND { #and }

<span class="axiom">AND</span> [`AndAxiom`][ir_axioms.axiom.retrieval.query_aspects.AndAxiom] — _"Prefer documents containing all query terms."_ [@wu2012relation]

Original formulation
:   Given \(q = \{t_1, t_2\}\) with \(\mathrm{idf}(t_1) \geq \mathrm{idf}(t_2)\):
    if \(\mathrm{tf}(t_1, d_1) = 1\), \(\mathrm{tf}(t_2, d_1) = 1\), \(\mathrm{tf}(t_1, d_2) > 1\), and \(\mathrm{tf}(t_2, d_2) = 0\), then \(d_1 >_\text{AND} d_2\).

Changes in `ir_axioms`
:   Prefers the document that contains all query terms if the other document does not, for queries of arbitrary length [@hagen2016axiomatic].

Variants
:   <span class="axiom">LEN_AND</span> [`LenAndAxiom`][ir_axioms.axiom.retrieval.query_aspects.LenAndAxiom]: AND with a document length precondition (`precondition=LEN()`).
:   <span class="axiom">M_AND</span> [`ModifiedAndAxiom`][ir_axioms.axiom.retrieval.query_aspects.ModifiedAndAxiom]: prefers the document that contains a larger subset of the query terms.
:   <span class="axiom">LEN_M_AND</span> [`LenModifiedAndAxiom`][ir_axioms.axiom.retrieval.query_aspects.LenModifiedAndAxiom]: M-AND with a document length precondition.

### DIV { #div }

<span class="axiom">DIV</span> [`DivAxiom`][ir_axioms.axiom.retrieval.query_aspects.DivAxiom] — _"Prefer documents that are less redundant with the query."_ [@gollapudi2009axiomatic]

Changes in `ir_axioms`
:   The diversity axioms of Gollapudi and Sharma characterize result-set diversification functions and do not express pairwise document preferences.
    Following Hagen et al. [@hagen2016axiomatic], `ir_axioms` reformulates the idea as a pairwise preference: prefer the document with the lower vocabulary overlap (Jaccard coefficient) with the query.

Variants
:   <span class="axiom">LEN_DIV</span> [`LenDivAxiom`][ir_axioms.axiom.retrieval.query_aspects.LenDivAxiom]: DIV with a document length precondition (`precondition=LEN()`).

## Semantic Similarity

### STMC1 { #stmc1 }

<span class="axiom">STMC1</span> [`Stmc1Axiom`][ir_axioms.axiom.retrieval.term_similarity.Stmc1Axiom] — _"Prefer documents with terms more similar to query terms."_ [@fang2006semantic]

Original formulation
:   Given \(q = \{t\}\), \(d_1 = \{t_1\}\), and \(d_2 = \{t_2\}\) with \(t \neq t_1\), \(t \neq t_2\): if \(\mathrm{sim}(t, t_1) > \mathrm{sim}(t, t_2)\), then \(d_1 >_\text{STMC1} d_2\).

Changes in `ir_axioms`
:   Following Hagen et al. [@hagen2016axiomatic], compares the average similarity of all document terms to all query terms, to allow arbitrary queries and documents.
    The term similarity is configurable (WordNet synonym sets by default, fastText if installed).

### STMC2 { #stmc2 }

<span class="axiom">STMC2</span> [`Stmc2Axiom`][ir_axioms.axiom.retrieval.term_similarity.Stmc2Axiom] — _"Do not reward similar terms more than exact matches."_ [@fang2006semantic]

Original formulation
:   Given \(q = \{t\}\) and \(d_1, d_2\) with \(|d_1| = 1\), \(|d_2| = c > 0\), and a term \(t'\) with \(\mathrm{sim}(t, t') > 0\): if \(\mathrm{tf}(t, d_1) = 1\) and \(\mathrm{tf}(t', d_2) = c\), then \(d_1 \geq_\text{STMC2} d_2\).

Changes in `ir_axioms`
:   Following Hagen et al. [@hagen2016axiomatic], given the most similar pair of a query term \(t\) and a non-query term \(t'\), prefer the document matching \(t\) exactly if the other document's frequency of \(t'\) relative to its length is similar (within a margin) to the first document's frequency of \(t\) relative to its length.

Parameters
:   `margin_fraction=0.2`.

## Term Proximity

The proximity axioms were proposed by Hagen et al. [@hagen2016axiomatic], inspired by the proximity measures studied by Tao and Zhai [@tao2007exploration].
Tao and Zhai's own constraints describe properties of distance measures and do not induce ranking preferences, so Hagen et al. derived the following pairwise axioms from the measures instead.
All of them require that both documents contain the same subset of query terms (PROX1–PROX3) or all query terms (PROX4–PROX5).

### PROX1 { #prox1 }

<span class="axiom">PROX1</span> [`Prox1Axiom`][ir_axioms.axiom.retrieval.proximity.Prox1Axiom] — _"Prefer documents with shorter distance between query term pairs."_

Given \(q\) and \(d_1, d_2\) with \(d_1 \cap q = d_2 \cap q\), let \(P = \{(t, t') : t, t' \in q, t \neq t'\}\) and \(\delta(d, t, t')\) the average number of words between \(t\) and \(t'\) in \(d\).
If \(\sum_{(t, t') \in P} \delta(d_1, t, t') < \sum_{(t, t') \in P} \delta(d_2, t, t')\), then \(d_1 >_\text{PROX1} d_2\).

### PROX2 { #prox2 }

<span class="axiom">PROX2</span> [`Prox2Axiom`][ir_axioms.axiom.retrieval.proximity.Prox2Axiom] — _"Prefer documents with earlier query term occurrences."_

Given \(q\) and \(d_1, d_2\) with \(d_1 \cap q = d_2 \cap q\), let \(\mathrm{first}(t, d)\) denote the position of the first occurrence of \(t\) in \(d\).
If \(\sum_{t \in q} \mathrm{first}(t, d_1) < \sum_{t \in q} \mathrm{first}(t, d_2)\), then \(d_1 >_\text{PROX2} d_2\).

### PROX3 { #prox3 }

<span class="axiom">PROX3</span> [`Prox3Axiom`][ir_axioms.axiom.retrieval.proximity.Prox3Axiom] — _"Prefer documents where the query occurs earlier as a phrase."_

Given \(q\) and \(d_1, d_2\) with \(d_1 \cap q = d_2 \cap q\), let \(\tau(d, q)\) denote the first position of the whole query \(q\) as one phrase in \(d\) (\(\infty\) if \(q\) does not occur as a phrase).
If \(\tau(d_1, q) < \tau(d_2, q)\), then \(d_1 >_\text{PROX3} d_2\).

### PROX4 { #prox4 }

<span class="axiom">PROX4</span> [`Prox4Axiom`][ir_axioms.axiom.retrieval.proximity.Prox4Axiom] — _"Prefer documents that contain all query terms in a shorter substring."_

Given \(q\) and \(d_1, d_2\) with \(d_1 \cap q = d_2 \cap q = q\), let \(\omega(d, q)\) denote the length of the shortest substring of \(d\) that contains all query terms.
If \(\omega(d_1, q) < \omega(d_2, q)\), then \(d_1 >_\text{PROX4} d_2\).
Following Hagen et al., \(\omega(d, q)\) is the pair of the number of non-query words within the closest grouping of all query terms and the frequency of this gap, so ties are broken by the number of such shortest substrings.

### PROX5 { #prox5 }

<span class="axiom">PROX5</span> [`Prox5Axiom`][ir_axioms.axiom.retrieval.proximity.Prox5Axiom] — _"Prefer documents where the query terms are closer together on average."_

Given \(q\) and \(d_1, d_2\) with \(d_1 \cap q = d_2 \cap q = q\), let \(\bar{s}(d, q)\) denote the average length of the shortest substrings of \(d\) that contain all query terms (averaged over all occurrences of query terms in \(d\)).
If \(\bar{s}(d_1, q) < \bar{s}(d_2, q)\), then \(d_1 >_\text{PROX5} d_2\).

## Argumentativeness

The argumentativeness axioms aim at queries that probably require argumentative results (e.g., _"mercy killing"_) [@bondarenko2018webis; @bondarenko2019webis].
They identify argumentative units (premises and claims) with the [TARGER](https://github.com/uhh-lt/targer) argument mining API and apply a document length precondition by default (`precondition=LEN()`).
Later work extended them for comparative questions [@reimer2022grimjack] and argument retrieval with embeddings [@heinrich2025axiomatic] (see [related axioms](related.md#argument-retrieval)).

### ArgUC { #arguc }

<span class="axiom">ArgUC</span> [`ArgumentativeUnitsCountAxiom`][ir_axioms.axiom.retrieval.argumentative.ArgumentativeUnitsCountAxiom] — _"Prefer more argumentative units."_ [@bondarenko2022axiomatic; @reimer2022grimjack]

Intuition
:   Favors documents with more argument units, i.e., premises and claims [@bondarenko2022axiomatic].

### QTArg { #qtarg }

<span class="axiom">QTArg</span> [`QueryTermOccurrenceInArgumentativeUnitsAxiom`][ir_axioms.axiom.retrieval.argumentative.QueryTermOccurrenceInArgumentativeUnitsAxiom] — _"Prefer more query terms in argumentative units."_ [@bondarenko2022axiomatic; @reimer2022grimjack]

Intuition
:   Favors documents where the query terms appear in (i.e., close to) the argument units [@bondarenko2022axiomatic].

### QTPArg { #qtparg }

<span class="axiom">QTPArg</span> [`QueryTermPositionInArgumentativeUnitsAxiom`][ir_axioms.axiom.retrieval.argumentative.QueryTermPositionInArgumentativeUnitsAxiom] — _"Prefer earlier query terms in argumentative units."_ [@bondarenko2022axiomatic; @reimer2022grimjack]

Intuition
:   Favors documents where the first occurrence of a query term in an argumentative unit is closer to the beginning of the document,
    based on the general observation that query terms occur "earlier" in relevant documents [@troy2007enhancing; @mitra2017learning].

Parameters
:   `penalty=100000`: penalty for the average query term position if a query term is not found in any argumentative unit (`None` to use the maximum length of the two documents).

### aSL { #asl }

<span class="axiom">aSL</span> [`AverageSentenceLengthAxiom`][ir_axioms.axiom.retrieval.argumentative.AverageSentenceLengthAxiom] (alias: `aSLDoc`) — _"Prefer passages with 12–20 words per sentence."_ [@bondarenko2022axiomatic; @reimer2022grimjack]

Intuition
:   Favors more "readable" documents [@bondarenko2022axiomatic], based on the general observation for text readability and good writing style [@markel2010technical; @newell2014editing].

Parameters
:   `min_sentence_length=12`, `max_sentence_length=20`.
