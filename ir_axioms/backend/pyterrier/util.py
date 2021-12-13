from ir_axioms.backend import PyTerrierBackendContext

with PyTerrierBackendContext():
    from jnius import autoclass

    StringReader = autoclass("java.io.StringReader")
    Index = autoclass("org.terrier.structures.Index")
    PostingIndex = autoclass("org.terrier.structures.PostingIndex")
    DocumentIndex = autoclass("org.terrier.structures.DocumentIndex")
    MetaIndex = autoclass("org.terrier.structures.MetaIndex")
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
