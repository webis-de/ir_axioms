# Correctness: factual verifiability and topical alignment
# Factual correctness:
# - [ ] Verifyaiity/faithfulness
#   - Intuition: Prefer text that is factually more consistent with the context/sources.
#   - Implementation: Extract sentences containing aspects from input, then find contradictions of the retrieved context as evidence.
#   - Implementation: Extract sentences containing aspects from input, then do claim verification, using the retrieved context as evidence.
# Topical correctness:
# - [ ] Topical relevance/alignment (-> use retrieval axioms?)
# TODO: Propose axioms for correctness.
