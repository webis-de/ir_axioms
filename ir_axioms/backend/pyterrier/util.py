from ir_axioms.backend.pyterrier.safe import autoclass

StringReader = autoclass("java.io.StringReader")
RequestContextMatching = autoclass("org.terrier.python.RequestContextMatching")
Index = autoclass("org.terrier.structures.Index")
IndexRef = autoclass('org.terrier.querying.IndexRef')
IndexFactory = autoclass('org.terrier.structures.IndexFactory')
PropertiesIndex = autoclass("org.terrier.structures.PropertiesIndex")
MetaIndex = autoclass("org.terrier.structures.MetaIndex")
Lexicon = autoclass("org.terrier.structures.Lexicon")
CollectionStatistics = autoclass("org.terrier.structures.CollectionStatistics")
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
ApplicationSetup = autoclass('org.terrier.utility.ApplicationSetup')
