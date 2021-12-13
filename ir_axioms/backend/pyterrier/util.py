from typing import Union

from ir_axioms.backend import PyTerrierBackendContext

with PyTerrierBackendContext():
    from jnius import autoclass, cast, JavaException

    RequestContextMatching = autoclass(
        "org.terrier.python.RequestContextMatching"
    )
    ApplicationSetup = autoclass("org.terrier.utility.ApplicationSetup")
    StringReader = autoclass("java.io.StringReader")
    Index = autoclass("org.terrier.structures.Index")
    PropertiesIndex = autoclass("org.terrier.structures.PropertiesIndex")
    Lexicon = autoclass("org.terrier.structures.Lexicon")
    CollectionStatistics = autoclass(
        "org.terrier.structures.CollectionStatistics"
    )
    Tokeniser = autoclass("org.terrier.indexing.tokenisation.Tokeniser")
    EnglishTokeniser = autoclass(
        "org.terrier.indexing.tokenisation.EnglishTokeniser"
    )
    WeightingModel = autoclass("org.terrier.matching.models.WeightingModel")
    TfModel = autoclass("org.terrier.matching.models.Tf")
    TfIdfModel = autoclass("org.terrier.matching.models.TF_IDF")
    BM25Model = autoclass("org.terrier.matching.models.BM25")
    PL2Model = autoclass("org.terrier.matching.models.PL2")
    DirichletLMModel = autoclass("org.terrier.matching.models.DirichletLM")
    TermPipelineAccessor = autoclass("org.terrier.terms.TermPipelineAccessor")
    BaseTermPipelineAccessor = autoclass(
        "org.terrier.terms.BaseTermPipelineAccessor"
    )
    SearchRequest = autoclass('org.terrier.querying.SearchRequest')
    ScoredDoc = autoclass('org.terrier.querying.ScoredDoc')
    ScoredDocList = autoclass('org.terrier.querying.ScoredDocList')
    Manager = autoclass('org.terrier.querying.Manager')
    ManagerFactory = autoclass('org.terrier.querying.ManagerFactory')


    def with_properties(index: Index) -> Union[PropertiesIndex, Index]:
        try:
            return cast("org.terrier.structures.PropertiesIndex", index)
        except JavaException:
            return index

