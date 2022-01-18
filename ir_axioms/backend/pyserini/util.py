from ir_axioms.backend.pyserini.safe import autoclass

_similarities = "org.apache.lucene.search.similarities"
Similarity = autoclass(f"{_similarities}.Similarity")
ClassicSimilarity = autoclass(f"{_similarities}.ClassicSimilarity")
BM25Similarity = autoclass(f"{_similarities}.BM25Similarity")
DFRSimilarity = autoclass(f"{_similarities}.DFRSimilarity")
BasicModelIn = autoclass(f"{_similarities}.BasicModelIn")
AfterEffectL = autoclass(f"{_similarities}.AfterEffectL")
NormalizationH2 = autoclass(f"{_similarities}.NormalizationH2")
LMDirichletSimilarity = autoclass(
    f"{_similarities}.LMDirichletSimilarity"
)
