# Axiom Details

Below, we provide an overview and mapping of various [utility dimensions](#utility-dimensions) described in related work to the [implemented axioms](#axioms-for-utility-dimensions).

## Utility Dimensions

Several frameworks describe utility dimensions of retrieval-augmented generation. The following list puts each into context with [the overall categorization](https://doi.org/10.1145/3626772.3657849) we used for our axioms.

- [Gienapp et al.](https://doi.org/10.1145/3626772.3657849):
  - coherence
  - coverage
  - consistency
  - correctness
  - clarity
  - (overall quality)
- [ES et al.](https://aclanthology.org/2024.eacl-demo.16/) ([RAGAs](https://github.com/explodinggradients/ragas)):
  - [context precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/) (retrieval-focused)
  - [context recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/) (retrieval-focused)
  - [context entities recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_entities_recall/) (retrieval-focused)
  - [noise sensitivity](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/noise_sensitivity/) (≃ factual correctness)
  - [answer relevance / response relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/) (≃ deep coverage)
  - [faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) (≃ external consistency)
  - context relevance (≃ broad coverage)
  - [multi-modal faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/multi_modal_faithfulness/) (only textual context considered yet; submit a [pull request](#TODO))
  - [multi-modal relevance](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/multi_modal_relevance/) (only textual context considered yet; submit a [pull request](#TODO))
- [Hashemi et al.](https://aclanthology.org/2024.acl-long.745v2) (LLM Rubric):
  - satisfaction
  - naturalness (≃ language clarity / content clarity)
  - grounding sources (retrieval-focused)
  - citation presence (≃ factual correctness)
  - citation suitability (≃ factual correctness)
  - citation optimality (≃ factual correctness / external consistency)
  - redundancy (≃ broad coverage)
  - conciseness (≃ content clarity)
  - efficiency (architecture-specific)
- [Wallat et al.](https://arxiv.org/abs/2412.18004)
  - citation correctness (≃ factual correctness)
  - citation faithfulness (≃ factual correctness)
- [Rosset et al.](https://doi.org/10.18653/v1/2023.emnlp-main.702)
  - usefulness (≃ overall quality)
  - relevance (≃ deep coverage / topical correctness)
  - groundedness (≃ factual correctness)
  - truthfulness (≃ external consistency)
  - relevant/irrelevant grounding (≃ external consistency / factual correctness)
  - thoroughness (≃ broad coverage)

## Axioms for Utility Dimensions

The following list matches our new proposed axioms (and existing axioms) to the utility aspects they address as per the categorization by Gienapp et al. (see above):

- [COH1](/axioms/axiom/generation/coherence.py) "Prefer less variance in avg. word length across sents." (coherence)
- [COH2](/axioms/axiom/generation/coherence.py) "Prefer response with subject–verb pairs closer togehter." (coherence)
- [COV1](/axioms/axiom/generation/coverage.py) "Prefer response containing more extracted aspects." (coverage)
- [COV2](/axioms/axiom/generation/coverage.py) "Prefer response with less redundant extracted aspects." (coverage)
- [COV3](/axioms/axiom/generation/coverage.py) "Prefer if more sentences cover aspects from the query." (coverage)
- [CONS1](/axioms/axiom/generation/consistency.py) "Prefer if more sentences cover aspects from the context." (consistency)
- [CONS2](/axioms/axiom/generation/consistency.py) "Prefer response with higher textual overlap with contexts." (consistency)
- [CONS3](/axioms/axiom/generation/consistency.py) "Penalize entities mentioned in contradictory phrases." (consistency)
- [CORR1](/axioms/axiom/generation/correctness.py) "Prefer response with more sentences containing citations." (correctness)
- [CLAR1](/axioms/axiom/generation/clarity.py) "Prefer lower text proportion covered by grammar errors." (clarity)
- [CLAR2](/axioms/axiom/generation/clarity.py) "Prefer the more readable reponse." (clarity)
