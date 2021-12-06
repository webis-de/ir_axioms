from ir_axioms.backend import PyseriniBackendContext

with PyseriniBackendContext():
    from jnius import java_method, autoclass, PythonJavaClass

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


    class TfSimilarity(PythonJavaClass, ClassicSimilarity):
        @java_method('(JJ)F')
        def idf(self, _document_frequency: int, _document_count: int) -> float:
            return 1

        # noinspection PyPep8Naming
        @java_method('(I)F')
        def lengthNorm(self, _num_terms: int) -> float:
            return 1

        @java_method('(F)F')
        def tf(self, freq: float) -> float:
            return freq
