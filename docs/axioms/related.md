# Related Axioms

<p class="subtitle">Axiomatic constraints from the literature, and which of them <code>ir_axioms</code> implements</p>

!!! abstract "Source"

    This page is adapted from the resource page [**Axiomatic Thinking for Information Retrieval**](https://www.eecis.udel.edu/~hfang/AX.html) by [Hui Fang](https://www.eecis.udel.edu/~hfang/) and [ChengXiang Zhai](http://czhai.cs.illinois.edu/).
    We revised the list for this documentation: fixed typos and dates, replaced outdated links with DOIs, marked which constraints are implemented in `ir_axioms`, and added more recent work (marked with :material-new-box:).
    Please refer to the [original page](https://www.eecis.udel.edu/~hfang/AX.html) for the authoritative version.

Axiomatic thinking has been adopted for the development of both retrieval models and evaluation metrics.
The general idea is to seek a set of desirable properties, expressed mathematically as formal constraints, to guide the search for an optimal solution.
The explicit expression of desirable properties makes it possible to analytically address issues that would otherwise appear to be purely empirical, provides theoretical guidance on how to optimize a retrieval model or evaluation metric, and allows applying the identified constraints directly in many practical applications.

Below, each constraint is listed with its intuition and source.
For constraints on retrieval models, the field "In `ir_axioms`" links to the corresponding implementation in the [retrieval axioms](retrieval.md) or [RAG axioms](generation.md), if any.
The [roadmap](../development/roadmap.md#feature-ideas) lists constraints that we would like to add.

## Resources

Tutorials
:   [SIGIR 2014 tutorial](https://www.eecis.udel.edu/~hfang/pubs/sigir14-axiomatic.pdf) "Axiomatic Analysis and Optimization of Information Retrieval Models" by Hui Fang and ChengXiang Zhai ([slides as PPTX](https://www.eecis.udel.edu/~hfang/pubs/sigir14-axiomatic.pptx)).
:   [ICTIR 2013 tutorial](https://www.eecis.udel.edu/~hfang/ictir13-ax4ir.pdf) by ChengXiang Zhai and Hui Fang [@zhai2013axiomatic].

Workshop
:   SIGIR 2017 Workshop on Axiomatic Thinking for Information Retrieval and Related Tasks (ATIR) [@amigo2017axiomatic] ([website](https://www.eecis.udel.edu/~hfang/ATIR.html), report [@amigo2018report]).

Mailing list
:   [AX4IR Google Group](https://groups.google.com/forum/#!forum/ax4ir) (`ax4ir@googlegroups.com`).

## Constraints for Retrieval Models

### Term Frequency, Term Discrimination, and Length Normalization

<span class="badge">TFC1</span> — _"Give a higher score to a document with more occurrences of a query term."_ [@fang2004formal]

In `ir_axioms`
:   [TFC1](retrieval.md#tfc1)

<span class="badge">TFC2</span> — _"The increase in the retrieval score due to an increase in TF should be smaller for larger TFs."_ [@fang2004formal; @fang2011diagnostic]

In `ir_axioms`
:   Not implemented ([why?](retrieval.md#tfc2)).

<span class="badge">TFC3</span> — _"If two documents have the same total occurrences of all query terms and all query terms have the same term discrimination value, give a higher score to the document covering more distinct query terms."_ [@fang2011diagnostic]

In `ir_axioms`
:   [TFC3](retrieval.md#tfc3)

<span class="badge">TDC</span> — _"Penalize terms that are popular in the collection."_ [@fang2004formal]

In `ir_axioms`
:   [M-TDC](retrieval.md#m-tdc), the modified TDC by Shi et al. [@shi2005gravitation].

<span class="badge">LNC1</span> — _"The score of a document should decrease if an occurrence of a non-relevant word is added."_ [@fang2004formal]

In `ir_axioms`
:   [LNC1](retrieval.md#lnc1)

<span class="badge">LNC2</span> — _"Avoid over-penalizing long relevant documents."_ [@fang2004formal]

In `ir_axioms`
:   Not implemented ([why?](retrieval.md#lnc2)).

<span class="badge">TF-LNC</span> — _"Regulate the interaction between TF and document length."_ [@fang2004formal]

In `ir_axioms`
:   [TF-LNC](retrieval.md#tf-lnc)

### Lower-Bounding Term Frequency

<span class="badge">LB1</span> — _"The presence–absence gap (0–1 gap) should not be closed due to length normalization."_ [@lv2011lower]

In `ir_axioms`
:   [LB1](retrieval.md#lb1)

<span class="badge">LB2</span> — _"Repeated occurrences of an already matched query term are not as important as the first occurrence of an otherwise absent query term."_ [@lv2011lower]

In `ir_axioms`
:   Not implemented ([why?](retrieval.md#lb2)).

### Semantic Term Matching

<span class="badge">STMC1</span> — _"Give a higher score to a document with a term that is more semantically related to a query term."_ [@fang2006semantic]

In `ir_axioms`
:   [STMC1](retrieval.md#stmc1)

<span class="badge">STMC2</span> — _"Avoid over-favoring semantically similar terms."_ [@fang2006semantic]

In `ir_axioms`
:   [STMC2](retrieval.md#stmc2)

<span class="badge">STMC3</span> — _"Favor semantically similar terms."_ [@fang2006semantic]

In `ir_axioms`
:   Not implemented.

<span class="badge">TSSC1, TSSC2</span> — _Term semantic similarity constraints for query expansion._ [@fang2008reexamination]

In `ir_axioms`
:   Not implemented.

### Term Proximity

<span class="badge">Proximity</span> — _"Term proximity should positively contribute to the retrieval score of a document."_ [@tao2007exploration]

In `ir_axioms`
:   Not implemented as such; see PROX1–PROX5 below.

<span class="badge">Convex curve</span> — _"The contribution of a distance measure should drop quickly when the distance is small and become nearly constant as the distance grows."_ [@tao2007exploration]

In `ir_axioms`
:   Not implemented.

<span class="badge">PROX1–PROX5</span> :material-new-box: — _"Prefer documents with query terms closer together or earlier."_ [@hagen2016axiomatic]

In `ir_axioms`
:   [PROX1–PROX5](retrieval.md#term-proximity)

### Query Term Relations

<span class="badge">REG</span> — _"Give a higher score to a document that covers more query aspects."_ [@zheng2010query]

In `ir_axioms`
:   [REG](retrieval.md#reg), ANTI-REG, ASPECT-REG

<span class="badge">AND</span> — _"If two query terms have an AND relation, documents with both terms should be ranked higher than those with only one of them."_ [@wu2012relation]

In `ir_axioms`
:   [AND](retrieval.md#and), M-AND, LEN-AND, LEN-M-AND

<span class="badge">Diversity</span> :material-new-box: — _Axioms characterizing result diversification functions._ [@gollapudi2009axiomatic]

In `ir_axioms`
:   [DIV](retrieval.md#div) (pairwise reformulation)

### Pseudo-Relevance Feedback

<span class="badge">Document score</span> — _"Documents with higher scores should be given more weight in the feedback weight function."_ [@clinchant2010information; @clinchant2013theoretical]

<span class="badge">Proximity</span> — _"Feedback terms should be close to query terms in documents."_ [@clinchant2010information; @clinchant2013theoretical]

<span class="badge">Document frequency</span> — _"Feedback terms should receive higher weights when they occur more often in the feedback set."_ [@clinchant2011document; @clinchant2011frequency]

In `ir_axioms`
:   Not implemented (`ir_axioms` does not cover query expansion yet).

### Translation Language Models

Constraints for translation language models for retrieval [@karimzadehgan2012axiomatic]:

<span class="badge">General 1</span> — _"For all translation language models, the self-translation probability should be the same."_

<span class="badge">General 2</span> — _"The self-translation probability should be larger than the probability of translating any other word to this word."_

<span class="badge">General 3</span> — _"A word is more likely to be translated to itself than to any other word."_

<span class="badge">Additional 4</span> — _"If word \(u\) occurs more often than word \(v\) in the context of word \(w\), and both co-occur similarly with all other words, the probability of translating \(u\) to \(w\) should be higher."_

<span class="badge">Additional 5</span> — _"If both \(u\) and \(v\) co-occur equally with word \(w\), but \(v\) co-occurs with many more other words than \(u\), the probability of translating \(u\) to \(w\) should be higher."_

In `ir_axioms`
:   Not applicable (constraints on the translation model rather than on rankings).

### Multi-Criteria Relevance Ranking

<span class="badge">Compatibility</span> — _Compatibility constraint for score transformations in linear combinations of relevance criteria._ [@gerani2012score]

In `ir_axioms`
:   Not implemented.

### Axiomatic Re-Ranking :material-new-box: { #axiomatic-re-ranking }

Besides PROX1–PROX5 and ORIG, Hagen et al. [@hagen2016axiomatic] proposed further axioms for re-ranking:

<span class="badge">QPHRA</span> — _"A document containing all the query phrases is favored over a document not containing all phrases."_ [@hagen2016axiomatic]

In `ir_axioms`
:   Not implemented (see also [PROX3](retrieval.md#prox3)).

<span class="badge">RSIM</span> — _"From any cluster of similar (near-duplicate) documents, favor only one document over all the others."_ [@hagen2016axiomatic]

In `ir_axioms`
:   Not implemented.

<span class="badge">P-RANK</span> — _"Prefer a document with higher PageRank."_ [@hagen2016axiomatic]

In `ir_axioms`
:   Not implemented.

<span class="badge">ORIG</span> — _"Give some voting power to the reasonable ideas underlying the basis retrieval model."_ [@hagen2016axiomatic]

In `ir_axioms`
:   [ORIG](utility.md#orig)

### Argument Retrieval :material-new-box: { #argument-retrieval }

<span class="badge">ArgUC</span> — _"Prefer more argumentative units."_ [@bondarenko2018webis; @bondarenko2019webis]

<span class="badge">QTArg</span> — _"Prefer more query terms in argumentative units."_ [@bondarenko2018webis; @bondarenko2019webis]

<span class="badge">QTPArg</span> — _"Prefer earlier query terms in argumentative units."_ [@bondarenko2018webis; @bondarenko2019webis]

<span class="badge">aSLDoc</span> — _"Prefer passages with 12–20 words per sentence."_ [@bondarenko2018webis; @bondarenko2019webis]

In `ir_axioms`
:   [ArgUC, QTArg, QTPArg, aSL](retrieval.md#argumentativeness)

<span class="badge">CompArg</span> — _"Prefer more comparative objects in argumentative units."_ [@reimer2022grimjack]

<span class="badge">CompPArg</span> — _"Prefer earlier comparative objects in argumentative units."_ [@reimer2022grimjack]

<span class="badge">ArgQ</span> — _"Prefer higher argument quality."_ [@reimer2022grimjack]

In `ir_axioms`
:   Not implemented (proposed for comparative questions).

<span class="badge">QArgSim<sub>avg</sub></span> — _"Prefer the document whose argumentative units on average are more similar to the query."_ [@heinrich2025axiomatic]

<span class="badge">QArgSim<sub>max</sub></span> — _"Prefer the document whose most similar argumentative unit is more similar to the query."_ [@heinrich2025axiomatic]

<span class="badge">QSenSim<sub>avg</sub></span> — _"Prefer the document whose sentences on average are more similar to the query."_ [@heinrich2025axiomatic]

<span class="badge">QSenSim<sub>max</sub></span> — _"Prefer the document whose most similar sentence is more similar to the query."_ [@heinrich2025axiomatic]

In `ir_axioms`
:   Not implemented. These refine [STMC1](retrieval.md#stmc1) from terms to argumentative units and sentences, each in an exact and a relaxed (10 % margin) variant.

### Generated Responses :material-new-box:

<span class="badge">Long-form QA</span> — _Axioms for preference modeling of long-form answers, e.g., usefulness, groundedness, and thoroughness._ [@rosset2023axiomatic]

In `ir_axioms`
:   Not implemented; related to the [RAG axioms](generation.md#utility-dimensions-in-related-work).

<span class="badge">RAG utility</span> — _Axioms for coherence, coverage, consistency, correctness, and clarity of generated responses._ [@merker2025axioms]

In `ir_axioms`
:   [RAG axioms](generation.md)

## Axioms for Neural Ranking Models :material-new-box:

Retrieval axioms have also been used to analyze, regularize, and explain neural ranking models:

Diagnosis
:   Diagnostic datasets derived from axioms reveal which constraints neural models satisfy [@rennings2019axiomatic; @camara2020diagnosing; @macavaney2022abnirml], also with relaxed term matching [@parry2026relaxed].

Regularization
:   Axioms serve as an additional loss when training neural models [@rosset2019axiomatic].

Explanation
:   Axiomatic preferences explain the rankings of neural models [@volske2021towards].

Re-ranking
:   Combined axiomatic preferences re-rank initial results [@hagen2016axiomatic; @bondarenko2022axiomatic].

## Constraints for Evaluation Metrics

These constraints characterize evaluation metrics rather than retrieval models and are thus not applicable to `ir_axioms`.

### General Constraints

<span class="badge">Strong definiteness</span> — _"The measure must be definable for any gold standard and system output."_ [@sebastiani2015axiomatically]

<span class="badge">Fixed range</span> — _"The metric score must be bounded from below and above."_ [@sebastiani2015axiomatically; @moffat2013seven]

### Classification Metrics

<span class="badge">Strict monotonicity</span> — _"Replacing an incorrect decision by a correct decision increases the score."_ [@sebastiani2015axiomatically; @sokolova2006assessing]

<span class="badge">Continuous differentiability</span> — _"The measure must be continuous and differentiable over the true positives and true negatives."_ [@sebastiani2015axiomatically]

<span class="badge">Symmetry</span> — _"The measure should be invariant with respect to switching the roles of the class and its complement."_ [@sebastiani2015axiomatically; @sokolova2006assessing]

<span class="badge">Absolute weighting</span> — _"A parameter of the measure determines whether adding one document of each class into a system's categorization improves the score."_ (task-dependent) [@amigo2017account]

<span class="badge">Non-informativeness fixed score</span> — _"Any non-informative (random) classification output achieves a fixed score."_ (task-dependent) [@amigo2017account]

<span class="badge">Non-informativeness growing score</span> — _"The score of a non-informative output for a class correlates with the output class size."_ (task-dependent) [@amigo2017account]

### Clustering Metrics

<span class="badge">Cluster homogeneity</span> — _"Joining clusters with items from different categories decreases the score."_ [@rosenberg2007vmeasure; @amigo2009comparison]

<span class="badge">Cluster completeness</span> — _"Items belonging to the same category should appear in the same cluster."_ [@rosenberg2007vmeasure; @dom2001information; @amigo2009comparison; @meila2003comparing]

<span class="badge">Rag bag</span> — _"Adding noise (single-item categories) to a clean cluster is worse than joining noisy items in the same cluster."_ [@amigo2009comparison]

<span class="badge">Cluster size vs. quantity</span> — _"A small error in a big cluster is preferable to many small errors in small clusters."_ [@amigo2009comparison; @meila2003comparing]

### Ranking Metrics

<span class="badge">Priority</span> — _"Swapping two documents in the ranking according to their relevance increases the score."_ [@moffat2013seven; @ferrante2015towards; @amigo2013general]

<span class="badge">Deepness</span> — _"There is an area of the ranking that is never explored by the user; the relevance of documents at the top has more effect."_ [@moffat2013seven; @ferrante2015towards; @amigo2013general]

<span class="badge">Deepness threshold</span> — _"One relevant document at the top is better than many relevant documents after many non-relevant documents."_ [@amigo2013general]

<span class="badge">Closeness threshold</span> — _"There is an area that is always explored by the user: there exists a small \(n\) such that \(n\) relevant documents in the first \(2n\) positions are better than one relevant document in the first position."_ [@amigo2013general]

<span class="badge">Confidence</span> — _"Adding noise at the end of the ranking decreases the score."_ [@amigo2013general]

See also the axiometrics framework [@busin2013axiometrics].

## Constraints for Similarity

These constraints characterize similarity functions and are also not applicable to `ir_axioms`.

### Metric-Space-Based Similarity

<span class="badge">Maximality</span>, <span class="badge">Symmetry</span>, <span class="badge">Triangle inequality</span> — _Similarity as a metric space._ [@shepard1987toward]

### Feature Contrast Model

<span class="badge">Matching</span> — _"Similarity is a function of the common and distinctive features of the objects."_ [@tversky1977features]

<span class="badge">Monotonicity</span> — _"Similarity increases whenever the common features increase or the differences decrease."_ [@tversky1977features]

<span class="badge">Independence</span> — _"Object features affect similarity independently."_ [@tversky1977features]

### Information-Theoretic Similarity

<span class="badge">Identity</span> — _"Adding or removing features to an object decreases its similarity to the original."_ [@amigo2017account; @amigo2020foundations]

<span class="badge">Identity specificity</span> — _"Adding new features increases the self-similarity."_ [@amigo2017account; @amigo2020foundations]

<span class="badge">Unexpectedness</span> — _"When adding a feature, similarity decreases to a greater extent if the added feature is less expected."_ [@amigo2017account; @amigo2020foundations]

<span class="badge">Dependency</span> — _"Adding new features to both objects increases their similarity whenever their respective conditional probabilities grow."_ [@amigo2017account; @amigo2020foundations]

<span class="badge">Asymmetry</span> — _"An object is more similar to any of its parts than vice versa."_ [@amigo2017account; @amigo2020foundations]

## Axioms Beyond Retrieval :material-new-box:

### Framing

A first axiomatization of frames in communication [@allein2022towards]:

<span class="badge">Axiom 1</span> — _"Frames exist."_

<span class="badge">Axiom 2</span> — _"Exposure to frames has measurable effects."_

<span class="badge">Axiom 3</span> — _"A frame can be defined by what belongs to the frame, or by what does not belong to the frame (frame dualism)."_

## Further Reading

More axiomatic analyses of retrieval models, as listed on the [original page](https://www.eecis.udel.edu/~hfang/AX.html):

Foundations
:   Aboutness axioms [@bruza1994investigating], the exploration of axiomatic approaches to retrieval [@fang2005exploration], and their diagnostic evaluation [@fang2011diagnostic].

Constraint analyses
:   Learned term-weighting schemes [@cummins2007axiomatic], measuring constraint violations [@cummins2009measuring], analysing ranking functions [@cummins2011analysing], length normalization [@cummins2012constraint], and heuristic constraints in function discovery [@goswami2015study].

Term frequency normalization
:   Multi-topical documents [@na2008improving], adaptive and log-logistic interpretations of BM25 [@lv2011adaptive; @lv2012loglogistic], very long documents [@lv2011documents], and query length heuristics [@lv2015study].

Information-based models
:   Information-based retrieval models [@clinchant2010information; @clinchant2010retrieval] and cross-language retrieval [@li2012information; @rahimi2014axiomatic].

Pseudo-relevance feedback
:   Document frequency [@clinchant2011document; @clinchant2011frequency], theoretical analyses [@clinchant2013theoretical], and smoothing methods [@hazimeh2015axiomatic].

Other tasks
:   Rating-based ranking of items [@zhang2011count], medical records search [@wang2014study], and query performance prediction [@makarenkov2015theoretical].
