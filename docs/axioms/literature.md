# Axioms in the Literature

<p class="subtitle">A revised list of axiomatic constraints in information retrieval</p>

!!! abstract "Source"

    This page is adapted from the resource page [**Axiomatic Thinking for Information Retrieval**](https://www.eecis.udel.edu/~hfang/AX.html) by [Hui Fang](https://www.eecis.udel.edu/~hfang/) and [ChengXiang Zhai](http://czhai.cs.illinois.edu/).
    We revised the list for this documentation: fixed typos and dates, replaced outdated links with DOIs, marked which constraints are implemented in `ir_axioms`, and added more recent work (marked with :material-new-box:).
    Please refer to the [original page](https://www.eecis.udel.edu/~hfang/AX.html) for the authoritative version.

Axiomatic thinking has been adopted for the development of both retrieval models and evaluation metrics.
The general idea is to seek a set of desirable properties, expressed mathematically as formal constraints, to guide the search for an optimal solution.
The explicit expression of desirable properties makes it possible to analytically address issues that would otherwise appear to be purely empirical, provides theoretical guidance on how to optimize a retrieval model or evaluation metric, and allows applying the identified constraints directly in many practical applications.

## Resources

- **Tutorials:**
  [SIGIR 2014 tutorial](https://www.eecis.udel.edu/~hfang/pubs/sigir14-axiomatic.pdf) "Axiomatic Analysis and Optimization of Information Retrieval Models" by Hui Fang and ChengXiang Zhai
  ([slides as PPTX](https://www.eecis.udel.edu/~hfang/pubs/sigir14-axiomatic.pptx));
  [ICTIR 2013 tutorial](https://doi.org/10.1145/2499178.2499205) ([slides](https://www.eecis.udel.edu/~hfang/ictir13-ax4ir.pdf)).
- **Workshop:**
  [SIGIR 2017 Workshop on Axiomatic Thinking for Information Retrieval and Related Tasks (ATIR)](https://doi.org/10.1145/3077136.3084369)
  ([website](https://www.eecis.udel.edu/~hfang/ATIR.html), [report](https://doi.org/10.1145/3190580.3190596)).
- **Mailing list:** [AX4IR Google Group](https://groups.google.com/forum/#!forum/ax4ir) (`ax4ir@googlegroups.com`).

## Constraints for Retrieval Models

The column "`ir_axioms`" links to the corresponding implementation, if any.

### Basic TF-IDF-LN Constraints

Proposed by Fang et al. ([2004](#ref-fang2004), [2011](#ref-fang2011)).

| Constraint | Intuition | `ir_axioms` |
|:--|:--|:--|
| TFC1 | Give a higher score to a document with more occurrences of a query term. | [TFC1](retrieval.md#tfc1) |
| TFC2 | The increase in the retrieval score due to an increase in TF should be smaller for larger TFs. | — |
| TFC3 | If two documents have the same total occurrences of all query terms and all query terms have the same term discrimination value, give a higher score to the document covering more distinct query terms. | [TFC3](retrieval.md#tfc3) |
| TDC | Penalize terms that are popular in the collection. | [M-TDC](retrieval.md#m-tdc) (modified by [Shi et al., 2005](#ref-shi2005)) |
| LNC1 | The score of a document should decrease if an occurrence of a non-relevant word is added. | [LNC1](retrieval.md#lnc1) |
| LNC2 | Avoid over-penalizing long relevant documents. | — |
| TF-LNC | Regulate the interaction between TF and document length. | [TF-LNC](retrieval.md#tf-lnc) |

### Lower-Bounding TF Constraints

Proposed by [Lv and Zhai (2011a)](#ref-lv2011a).

| Constraint | Intuition | `ir_axioms` |
|:--|:--|:--|
| LB1 | The presence–absence gap (0–1 gap) should not be closed due to length normalization. | [LB1](retrieval.md#lb1) |
| LB2 | Repeated occurrences of an already matched query term are not as important as the first occurrence of an otherwise absent query term. | — |

### Semantic Term Matching Constraints

Proposed by [Fang and Zhai (2006)](#ref-fang2006) and [Fang (2008)](#ref-fang2008).

| Constraint | Intuition | `ir_axioms` |
|:--|:--|:--|
| STMC1 | Give a higher score to a document with a term that is more semantically related to a query term. | [STMC1](retrieval.md#stmc1) |
| STMC2 | Avoid over-favoring semantically similar terms. | [STMC2](retrieval.md#stmc2) |
| STMC3 | Favor semantically similar terms. | — |
| TSSC1, TSSC2 | Term semantic similarity constraints for query expansion ([Fang, 2008](#ref-fang2008)). | — |

### Term Proximity Constraints

| Constraint | Intuition | `ir_axioms` |
|:--|:--|:--|
| Proximity heuristic | Term proximity should positively contribute to the retrieval score of a document ([Tao and Zhai, 2007](#ref-tao2007)). | — |
| Convex curve | The contribution of a distance measure should drop quickly when the distance is small and become nearly constant as the distance grows ([Tao and Zhai, 2007](#ref-tao2007)). | — |
| :material-new-box: PROX1–PROX5 | Prefer documents with query terms closer together or earlier ([Hagen et al., 2016](#ref-hagen2016)). | [PROX1–PROX5](retrieval.md#term-proximity) |

### Query Term Relation Constraints

| Constraint | Intuition | `ir_axioms` |
|:--|:--|:--|
| REG | Give a higher score to a document that covers more query aspects ([Zheng and Fang, 2010](#ref-zheng2010)). | [REG](retrieval.md#reg), ANTI-REG, ASPECT-REG |
| AND | If two query terms have an AND relation, documents with both terms should be ranked higher than those with only one of them ([Wu and Fang, 2012](#ref-wu2012)). | [AND](retrieval.md#and), M-AND, LEN-AND, LEN-M-AND |
| :material-new-box: Diversity | Axioms characterizing result diversification functions ([Gollapudi and Sharma, 2009](#ref-gollapudi2009)). | [DIV](retrieval.md#div) (pairwise reformulation) |

### Constraints for Pseudo-Relevance Feedback

Proposed by Clinchant and Gaussier ([2010](#ref-clinchant2010), [2011a](#ref-clinchant2011a), [2011b](#ref-clinchant2011b), [2013](#ref-clinchant2013)).

- **Document score constraint:** Documents with higher scores should be given more weight in the feedback weight function.
- **Proximity constraint:** Feedback terms should be close to query terms in documents.
- **Document frequency constraint:** Feedback terms should receive higher weights when they occur more often in the feedback set.

### Constraints for Translation Language Models

Proposed by [Karimzadehgan and Zhai (2012)](#ref-karimzadehgan2012).

1. For all translation language models, the self-translation probability should be the same.
2. The self-translation probability should be larger than the probability of translating any other word to this word.
3. A word is more likely to be translated to itself than to any other word.
4. If word \(u\) occurs more often than word \(v\) in the context of word \(w\), and both \(u\) and \(v\) co-occur similarly with all other words, the probability of translating \(u\) to \(w\) should be higher.
5. If both \(u\) and \(v\) co-occur equally with word \(w\), but \(v\) co-occurs with many more other words than \(u\), the probability of translating \(u\) to \(w\) should be higher.

### Constraints for Multi-Criteria Relevance Ranking

- **Compatibility constraint** ([Gerani et al., 2012](#ref-gerani2012)).

### :material-new-box: Further Retrieval Constraints

| Constraint | Intuition | `ir_axioms` |
|:--|:--|:--|
| Argumentativeness | Prefer documents with more argumentative units, query terms in arguments, and readable sentences ([Bondarenko et al., 2018](#ref-bondarenko2018), [2019](#ref-bondarenko2019)). | [ArgUC, QTArg, QTPArg, aSL](retrieval.md#argumentativeness) |
| Long-form QA | Axioms for preference modeling of long-form answers, e.g., usefulness, groundedness, and thoroughness ([Rosset et al., 2023](#ref-rosset2023)). | — |
| RAG utility | Axioms for coherence, coverage, consistency, correctness, and clarity of generated responses ([Merker et al., 2025](#ref-merker2025)). | [RAG axioms](generation.md) |

## :material-new-box: Axioms for Neural Ranking Models

Retrieval axioms have also been used to analyze, regularize, and explain neural ranking models:

- **Diagnosing neural models** with diagnostic datasets derived from axioms ([Rennings et al., 2019](#ref-rennings2019); [Câmara and Hauff, 2020](#ref-camara2020); [MacAvaney et al., 2022](#ref-macavaney2022)).
- **Regularizing neural models** with axioms as an additional loss ([Rosset et al., 2019](#ref-rosset2019)).
- **Explaining neural rankings** by axiomatic preferences ([Völske et al., 2021](#ref-volske2021)).
- **Re-ranking** with combined axiomatic preferences ([Hagen et al., 2016](#ref-hagen2016); [Bondarenko et al., 2022](#ref-bondarenko2022)).

## Evaluation Metric Axiomatics

### General Constraints

- **Strong definiteness:** The measure must be definable for any gold standard and system output ([Sebastiani, 2015](#ref-sebastiani2015)).
- **Fixed range:** The metric score must be bounded from below and above ([Sebastiani, 2015](#ref-sebastiani2015); [Moffat, 2013](#ref-moffat2013)).

### Classification Metrics

- **Strict monotonicity:** Replacing an incorrect decision by a correct decision increases the score ([Sebastiani, 2015](#ref-sebastiani2015); [Sokolova, 2006](#ref-sokolova2006)).
- **Continuous differentiability:** The measure must be continuous and differentiable over the true positives and true negatives ([Sebastiani, 2015](#ref-sebastiani2015)).
- **Symmetry:** The measure should be invariant with respect to switching the roles of the class and its complement ([Sebastiani, 2015](#ref-sebastiani2015); [Sokolova, 2006](#ref-sokolova2006)).
- **Absolute weighting** (task-dependent): A parameter of the measure determines whether adding one document of each class into a system's categorization improves the score (Amigó et al., 2017).
- **Non-informativeness fixed score** (task-dependent): Any non-informative (random) classification output achieves a fixed score (Amigó et al., 2017).
- **Non-informativeness growing score** (task-dependent): The score of a non-informative output for a class correlates with the output class size (Amigó et al., 2017).

### Clustering Metrics

- **Cluster homogeneity:** Joining clusters with items from different categories decreases the score ([Rosenberg and Hirschberg, 2007](#ref-rosenberg2007); [Amigó et al., 2009](#ref-amigo2009)).
- **Cluster completeness:** Items belonging to the same category should appear in the same cluster ([Rosenberg and Hirschberg, 2007](#ref-rosenberg2007); Dom, 2001; [Amigó et al., 2009](#ref-amigo2009); [Meilă, 2003](#ref-meila2003)).
- **Rag bag:** Adding noise (single-item categories) to a clean cluster is worse than joining noisy items in the same cluster ([Amigó et al., 2009](#ref-amigo2009)).
- **Cluster size vs. quantity:** A small error in a big cluster is preferable to many small errors in small clusters ([Amigó et al., 2009](#ref-amigo2009); [Meilă, 2003](#ref-meila2003)).

### Ranking Metrics

- **Priority:** Swapping two documents in the ranking according to their relevance increases the score ([Moffat, 2013](#ref-moffat2013); [Ferrante et al., 2015](#ref-ferrante2015); [Amigó et al., 2013](#ref-amigo2013)).
- **Deepness:** There is an area of the ranking that is never explored by the user; the relevance of documents at the top has more effect ([Moffat, 2013](#ref-moffat2013); [Ferrante et al., 2015](#ref-ferrante2015); [Amigó et al., 2013](#ref-amigo2013)).
- **Deepness threshold:** One relevant document at the top is better than many relevant documents after many non-relevant documents ([Amigó et al., 2013](#ref-amigo2013)).
- **Closeness threshold:** There is an area that is always explored by the user: there exists a small \(n\) such that \(n\) relevant documents in the first \(2n\) positions are better than one relevant document in the first position ([Amigó et al., 2013](#ref-amigo2013)).
- **Confidence:** Adding noise at the end of the ranking decreases the score ([Amigó et al., 2013](#ref-amigo2013)).

See also the axiometrics framework by [Busin and Mizzaro (2013)](#ref-busin2013).

## Similarity Axiomatics

- **Metric-space-based similarity** ([Shepard, 1987](#ref-shepard1987)): maximality, symmetry, triangle inequality.
- **Feature contrast model** ([Tversky, 1977](#ref-tversky1977)):
    - _Matching:_ Similarity is a function of the common and distinctive features of the objects.
    - _Monotonicity:_ Similarity increases whenever the common features increase or the differences decrease.
    - _Independence:_ Object features affect similarity independently.
- **Information-theoretic similarity** (Amigó et al., 2017; see also [Amigó et al., 2020](#ref-amigo2020)):
    - _Identity:_ Adding or removing features to an object decreases its similarity to the original.
    - _Identity specificity:_ Adding new features increases the self-similarity.
    - _Unexpectedness:_ When adding a feature, similarity decreases to a greater extent if the added feature is less expected.
    - _Dependency:_ Adding new features to both objects increases their similarity whenever their respective conditional probabilities grow.
    - _Asymmetry:_ An object is more similar to any of its parts than vice versa.

## References

### Retrieval Models

- <a id="ref-bruza1994"></a>Bruza and Huibers (1994). Investigating Aboutness Axioms using Information Fields. SIGIR 1994. [doi:10.1007/978-1-4471-2099-5_12](https://doi.org/10.1007/978-1-4471-2099-5_12)
- <a id="ref-fang2004"></a>Fang, Tao, and Zhai (2004). A Formal Study of Information Retrieval Heuristics. SIGIR 2004. [doi:10.1145/1008992.1009004](https://doi.org/10.1145/1008992.1009004)
- <a id="ref-fang2005"></a>Fang and Zhai (2005). An Exploration of Axiomatic Approaches to Information Retrieval. SIGIR 2005. [doi:10.1145/1076034.1076116](https://doi.org/10.1145/1076034.1076116)
- <a id="ref-shi2005"></a>:material-new-box: Shi, Wen, Yu, Song, and Ma (2005). Gravitation-Based Model for Information Retrieval. SIGIR 2005. [doi:10.1145/1076034.1076117](https://doi.org/10.1145/1076034.1076117)
- <a id="ref-fang2006"></a>Fang and Zhai (2006). Semantic Term Matching in Axiomatic Approaches to Information Retrieval. SIGIR 2006. [doi:10.1145/1148170.1148193](https://doi.org/10.1145/1148170.1148193)
- <a id="ref-tao2007"></a>Tao and Zhai (2007). An Exploration of Proximity Measures in Information Retrieval. SIGIR 2007. [doi:10.1145/1277741.1277794](https://doi.org/10.1145/1277741.1277794)
- <a id="ref-cummins2007"></a>Cummins and O'Riordan (2007). An Axiomatic Comparison of Learned Term-Weighting Schemes in Information Retrieval: Clarifications and Extensions. Artificial Intelligence Review. [doi:10.1007/s10462-008-9074-5](https://doi.org/10.1007/s10462-008-9074-5)
- <a id="ref-fang2008"></a>Fang (2008). A Re-examination of Query Expansion Using Lexical Resources. ACL 2008. [aclanthology:P08-1017](https://aclanthology.org/P08-1017/)
- <a id="ref-na2008"></a>Na, Kang, and Lee (2008). Improving Term Frequency Normalization for Multi-topical Documents and Application to Language Modeling Approaches. ECIR 2008. [doi:10.1007/978-3-540-78646-7_35](https://doi.org/10.1007/978-3-540-78646-7_35)
- <a id="ref-gollapudi2009"></a>Gollapudi and Sharma (2009). An Axiomatic Approach for Result Diversification. WWW 2009. [doi:10.1145/1526709.1526761](https://doi.org/10.1145/1526709.1526761)
- <a id="ref-cummins2009"></a>Cummins and O'Riordan (2009). Measuring Constraint Violations in Information Retrieval. SIGIR 2009. [doi:10.1145/1571941.1572096](https://doi.org/10.1145/1571941.1572096)
- <a id="ref-zheng2010"></a>Zheng and Fang (2010). Query Aspect Based Term Weighting Regularization in Information Retrieval. ECIR 2010. [doi:10.1007/978-3-642-12275-0_31](https://doi.org/10.1007/978-3-642-12275-0_31)
- <a id="ref-clinchant2010"></a>Clinchant and Gaussier (2010). Information-Based Models for Ad Hoc IR. SIGIR 2010. [doi:10.1145/1835449.1835490](https://doi.org/10.1145/1835449.1835490)
- <a id="ref-clinchant2010b"></a>Clinchant and Gaussier (2010). Retrieval Constraints and Word Frequency Distributions: A Log-Logistic Model for IR. Information Retrieval. [doi:10.1007/s10791-010-9143-7](https://doi.org/10.1007/s10791-010-9143-7)
- <a id="ref-fang2011"></a>Fang, Tao, and Zhai (2011). Diagnostic Evaluation of Information Retrieval Models. ACM TOIS. [doi:10.1145/1961209.1961210](https://doi.org/10.1145/1961209.1961210)
- <a id="ref-lv2011a"></a>Lv and Zhai (2011a). Lower-Bounding Term Frequency Normalization. CIKM 2011. [doi:10.1145/2063576.2063584](https://doi.org/10.1145/2063576.2063584)
- <a id="ref-lv2011b"></a>Lv and Zhai (2011b). Adaptive Term Frequency Normalization for BM25. CIKM 2011. [doi:10.1145/2063576.2063871](https://doi.org/10.1145/2063576.2063871)
- <a id="ref-lv2011c"></a>Lv and Zhai (2011c). When Documents Are Very Long, BM25 Fails! SIGIR 2011. [doi:10.1145/2009916.2010070](https://doi.org/10.1145/2009916.2010070)
- <a id="ref-clinchant2011a"></a>Clinchant and Gaussier (2011a). Is Document Frequency Important for PRF? ICTIR 2011. [doi:10.1007/978-3-642-23318-0_10](https://doi.org/10.1007/978-3-642-23318-0_10)
- <a id="ref-clinchant2011b"></a>Clinchant and Gaussier (2011b). A Document Frequency Constraint for Pseudo-Relevance Feedback Models. CORIA 2011. [PDF](http://www.asso-aria.org/coria/2011/73.pdf)
- <a id="ref-zhang2011"></a>Zhang, Mao, Li, and Mao (2011). How to Count Thumb-Ups and Thumb-Downs: User-Rating Based Ranking of Items from an Axiomatic Perspective. ICTIR 2011. [doi:10.1007/978-3-642-23318-0_22](https://doi.org/10.1007/978-3-642-23318-0_22)
- <a id="ref-cummins2011"></a>Cummins and O'Riordan (2011). Analysing Ranking Functions in Information Retrieval Using Constraints. Information Extraction from the Internet.
- <a id="ref-lv2012"></a>Lv and Zhai (2012). A Log-Logistic Model-Based Interpretation of TF Normalization of BM25. ECIR 2012. [doi:10.1007/978-3-642-28997-2_21](https://doi.org/10.1007/978-3-642-28997-2_21)
- <a id="ref-wu2012"></a>Wu and Fang (2012). Relation Based Term Weighting Regularization. ECIR 2012. [doi:10.1007/978-3-642-28997-2_10](https://doi.org/10.1007/978-3-642-28997-2_10)
- <a id="ref-li2012"></a>Li and Gaussier (2012). An Information-Based Cross-Language Information Retrieval Model. ECIR 2012. [doi:10.1007/978-3-642-28997-2_24](https://doi.org/10.1007/978-3-642-28997-2_24)
- <a id="ref-gerani2012"></a>Gerani, Zhai, and Crestani (2012). Score Transformation in Linear Combination for Multi-criteria Relevance Ranking. ECIR 2012. [doi:10.1007/978-3-642-28997-2_22](https://doi.org/10.1007/978-3-642-28997-2_22)
- <a id="ref-karimzadehgan2012"></a>Karimzadehgan and Zhai (2012). Axiomatic Analysis of Translation Language Model for Information Retrieval. ECIR 2012. [doi:10.1007/978-3-642-28997-2_23](https://doi.org/10.1007/978-3-642-28997-2_23)
- <a id="ref-cummins2012"></a>Cummins and O'Riordan (2012). A Constraint to Automatically Regulate Document-Length Normalisation. CIKM 2012. [doi:10.1145/2396761.2398662](https://doi.org/10.1145/2396761.2398662)
- <a id="ref-clinchant2013"></a>Clinchant and Gaussier (2013). A Theoretical Analysis of Pseudo-Relevance Feedback Models. ICTIR 2013. [doi:10.1145/2499178.2499179](https://doi.org/10.1145/2499178.2499179)
- <a id="ref-wang2014"></a>Wang, Liu, and Fang (2014). A Study of Concept-based Weighting Regularization for Medical Records Search. ACL 2014. [doi:10.3115/v1/P14-1057](https://doi.org/10.3115/v1/P14-1057)
- <a id="ref-rahimi2014"></a>Rahimi, Shakery, and King (2014). Axiomatic Analysis of Cross-Language Information Retrieval. CIKM 2014. [doi:10.1145/2661829.2661915](https://doi.org/10.1145/2661829.2661915)
- <a id="ref-hazimeh2015"></a>Hazimeh and Zhai (2015). Axiomatic Analysis of Smoothing Methods in Language Models for Pseudo-Relevance Feedback. ICTIR 2015. [doi:10.1145/2808194.2809471](https://doi.org/10.1145/2808194.2809471)
- <a id="ref-lv2015"></a>Lv (2015). A Study of Query Length Heuristics in Information Retrieval. CIKM 2015. [doi:10.1145/2806416.2806592](https://doi.org/10.1145/2806416.2806592)
- <a id="ref-goswami2015"></a>Goswami, Amini, and Gaussier (2015). Study of Heuristic IR Constraints Under Function Discovery Framework. ICTIR 2015. [doi:10.1145/2808194.2809479](https://doi.org/10.1145/2808194.2809479)
- <a id="ref-makarenkov2015"></a>Makarenkov, Shapira, and Rokach (2015). Theoretical Categorization of Query Performance Predictors. ICTIR 2015. [doi:10.1145/2808194.2809475](https://doi.org/10.1145/2808194.2809475)
- <a id="ref-hagen2016"></a>:material-new-box: Hagen, Völske, Göring, and Stein (2016). Axiomatic Result Re-Ranking. CIKM 2016. [doi:10.1145/2983323.2983704](https://doi.org/10.1145/2983323.2983704)
- <a id="ref-bondarenko2018"></a>:material-new-box: Bondarenko et al. (2018). Webis at TREC 2018: Common Core Track. TREC 2018. [PDF](https://trec.nist.gov/pubs/trec27/papers/Webis-CC.pdf)
- <a id="ref-bondarenko2019"></a>:material-new-box: Bondarenko et al. (2019). Webis at TREC 2019: Decision Track. TREC 2019. [PDF](https://trec.nist.gov/pubs/trec28/papers/Webis.D.pdf)
- <a id="ref-rennings2019"></a>:material-new-box: Rennings, Moraes, and Hauff (2019). An Axiomatic Approach to Diagnosing Neural IR Models. ECIR 2019. [doi:10.1007/978-3-030-15712-8_32](https://doi.org/10.1007/978-3-030-15712-8_32)
- <a id="ref-rosset2019"></a>:material-new-box: Rosset et al. (2019). An Axiomatic Approach to Regularizing Neural Ranking Models. SIGIR 2019. [doi:10.1145/3331184.3331296](https://doi.org/10.1145/3331184.3331296)
- <a id="ref-camara2020"></a>:material-new-box: Câmara and Hauff (2020). Diagnosing BERT with Retrieval Heuristics. ECIR 2020. [doi:10.1007/978-3-030-45439-5_40](https://doi.org/10.1007/978-3-030-45439-5_40)
- <a id="ref-volske2021"></a>:material-new-box: Völske et al. (2021). Towards Axiomatic Explanations for Neural Ranking Models. ICTIR 2021. [doi:10.1145/3471158.3472256](https://doi.org/10.1145/3471158.3472256)
- <a id="ref-macavaney2022"></a>:material-new-box: MacAvaney et al. (2022). ABNIRML: Analyzing the Behavior of Neural IR Models. TACL. [doi:10.1162/tacl_a_00457](https://doi.org/10.1162/tacl_a_00457)
- <a id="ref-bondarenko2022"></a>:material-new-box: Bondarenko et al. (2022). Axiomatic Retrieval Experimentation with `ir_axioms`. SIGIR 2022. [doi:10.1145/3477495.3531743](https://doi.org/10.1145/3477495.3531743)
- <a id="ref-rosset2023"></a>:material-new-box: Rosset et al. (2023). Axiomatic Preference Modeling for Longform Question Answering. EMNLP 2023. [doi:10.18653/v1/2023.emnlp-main.702](https://doi.org/10.18653/v1/2023.emnlp-main.702)
- <a id="ref-merker2025"></a>:material-new-box: Merker et al. (2025). Axioms for Retrieval-Augmented Generation. ICTIR 2025. [doi:10.1145/3731120.3744601](https://doi.org/10.1145/3731120.3744601)

### Evaluation and Similarity

- <a id="ref-tversky1977"></a>Tversky (1977). Features of Similarity. Psychological Review 84(4). [doi:10.1037/0033-295X.84.4.327](https://doi.org/10.1037/0033-295X.84.4.327)
- <a id="ref-shepard1987"></a>Shepard (1987). Toward a Universal Law of Generalization for Psychological Science. Science 237. [doi:10.1126/science.3629243](https://doi.org/10.1126/science.3629243)
- Dom (2001). An Information-Theoretic External Cluster-Validity Measure. IBM Research Report.
- <a id="ref-meila2003"></a>Meilă (2003). Comparing Clusterings by the Variation of Information. COLT 2003. [doi:10.1007/978-3-540-45167-9_14](https://doi.org/10.1007/978-3-540-45167-9_14)
- <a id="ref-sokolova2006"></a>Sokolova (2006). Assessing Invariance Properties of Evaluation Measures. NIPS 2006 Workshop on Testing Deployable Learning and Decision Systems.
- <a id="ref-rosenberg2007"></a>Rosenberg and Hirschberg (2007). V-Measure: A Conditional Entropy-Based External Cluster Evaluation Measure. EMNLP-CoNLL 2007. [aclanthology:D07-1043](https://aclanthology.org/D07-1043/)
- <a id="ref-amigo2009"></a>Amigó, Gonzalo, Artiles, and Verdejo (2009). A Comparison of Extrinsic Clustering Evaluation Metrics Based on Formal Constraints. Information Retrieval 12(4). [doi:10.1007/s10791-008-9066-8](https://doi.org/10.1007/s10791-008-9066-8)
- <a id="ref-amigo2013"></a>Amigó, Gonzalo, and Verdejo (2013). A General Evaluation Measure for Document Organization Tasks. SIGIR 2013. [doi:10.1145/2484028.2484081](https://doi.org/10.1145/2484028.2484081)
- <a id="ref-busin2013"></a>Busin and Mizzaro (2013). Axiometrics: An Axiomatic Approach to Information Retrieval Effectiveness Metrics. ICTIR 2013. [doi:10.1145/2499178.2499182](https://doi.org/10.1145/2499178.2499182)
- <a id="ref-moffat2013"></a>Moffat (2013). Seven Numeric Properties of Effectiveness Metrics. AIRS 2013. [doi:10.1007/978-3-642-45068-6_1](https://doi.org/10.1007/978-3-642-45068-6_1)
- <a id="ref-ferrante2015"></a>Ferrante, Ferro, and Maistro (2015). Towards a Formal Framework for Utility-oriented Measurements of Retrieval Effectiveness. ICTIR 2015. [doi:10.1145/2808194.2809452](https://doi.org/10.1145/2808194.2809452)
- <a id="ref-sebastiani2015"></a>Sebastiani (2015). An Axiomatically Derived Measure for the Evaluation of Classification Algorithms. ICTIR 2015. [doi:10.1145/2808194.2809449](https://doi.org/10.1145/2808194.2809449)
- Amigó, Gonzalo, Giner, and Verdejo (2017). An Axiomatic Account of Similarity. ATIR Workshop at SIGIR 2017.
- <a id="ref-amigo2020"></a>:material-new-box: Amigó, Giner, Gonzalo, and Verdejo (2020). On the Foundations of Similarity in Information Access. Information Retrieval Journal. [doi:10.1007/s10791-020-09375-z](https://doi.org/10.1007/s10791-020-09375-z)
