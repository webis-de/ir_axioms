from ir_axioms.backend import require_pyserini_backend

with require_pyserini_backend():
    from pyserini.pyclass import autoclass

_similarities = "org.apache.lucene.search.similarities"
JSimilarity = autoclass(f"{_similarities}.Similarity")
JClassicSimilarity = autoclass(f"{_similarities}.ClassicSimilarity")
JBM25Similarity = autoclass(f"{_similarities}.BM25Similarity")
JDFRSimilarity = autoclass(f"{_similarities}.DFRSimilarity")
JBasicModelIn = autoclass(f"{_similarities}.BasicModelIn")
JAfterEffectL = autoclass(f"{_similarities}.AfterEffectL")
JNormalizationH2 = autoclass(f"{_similarities}.NormalizationH2")
JLMDirichletSimilarity = autoclass(
    f"{_similarities}.LMDirichletSimilarity"
)
