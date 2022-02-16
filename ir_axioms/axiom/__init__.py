from ir_axioms.axiom import (
    argumentative, arithmetic, base, cache, conversion, length_norm,
    lower_bound, proximity, query_aspects, retrieval_score, term_frequency,
    term_similarity, simple, trec, estimator, preconditions
)

# Re-export from child modules.
ArgumentativeUnitsCountAxiom = argumentative.ArgumentativeUnitsCountAxiom
QueryTermOccurrenceInArgumentativeUnitsAxiom = (
    argumentative.QueryTermOccurrenceInArgumentativeUnitsAxiom
)
QueryTermPositionInArgumentativeUnitsAxiom = (
    argumentative.QueryTermPositionInArgumentativeUnitsAxiom
)
AverageSentenceLengthAxiom = argumentative.AverageSentenceLengthAxiom
ArgUC = argumentative.ArgUC
QTArg = argumentative.QTArg
QTPArg = argumentative.QTPArg
aSL = argumentative.aSL

UniformAxiom = arithmetic.UniformAxiom
SumAxiom = arithmetic.SumAxiom
ProductAxiom = arithmetic.ProductAxiom
MultiplicativeInverseAxiom = arithmetic.MultiplicativeInverseAxiom
AndAxiom = arithmetic.AndAxiom
VoteAxiom = arithmetic.VoteAxiom
NormalizedAxiom = arithmetic.NormalizedAxiom

Axiom = base.Axiom
AxiomLike = base.AxiomLike

CachedAxiom = cache.CachedAxiom

to_axiom = conversion.to_axiom
to_axioms = conversion.to_axioms
AutoAxiom = conversion.AutoAxiom

EstimatorAxiom = estimator.EstimatorAxiom

LNC1 = length_norm.LNC1
TF_LNC = length_norm.TF_LNC

LEN = preconditions.LEN
LEN_Mixin = preconditions.LEN_Mixin

LB1 = lower_bound.LB1

PROX1 = proximity.PROX1
PROX2 = proximity.PROX2
PROX3 = proximity.PROX3
PROX4 = proximity.PROX4
PROX5 = proximity.PROX5

REG = query_aspects.REG
REG_fastText = query_aspects.REG_fastText
REG_f = query_aspects.REG_f
ANTI_REG = query_aspects.ANTI_REG
ANTI_REG_fastText = query_aspects.ANTI_REG_fastText
ANTI_REG_f = query_aspects.ANTI_REG_f
ASPECT_REG = query_aspects.ASPECT_REG
ASPECT_REG_fastText = query_aspects.ASPECT_REG_fastText
ASPECT_REG_f = query_aspects.ASPECT_REG_f
AND = query_aspects.AND
LEN_AND = query_aspects.LEN_AND
M_AND = query_aspects.M_AND
LEN_M_AND = query_aspects.LEN_M_AND
DIV = query_aspects.DIV
LEN_DIV = query_aspects.LEN_DIV

RetrievalScoreAxiom = retrieval_score.RetrievalScoreAxiom
RS_TF = retrieval_score.RS_TF
RS_TF_IDF = retrieval_score.RS_TF_IDF
RS_BM25 = retrieval_score.RS_BM25
RS_PL2 = retrieval_score.RS_PL2
RS_QL = retrieval_score.RS_QL
RS = retrieval_score.RS

NopAxiom = simple.NopAxiom
OriginalAxiom = simple.OriginalAxiom
OracleAxiom = simple.OracleAxiom
RandomAxiom = simple.RandomAxiom
NOP = simple.NOP
ORIG = simple.ORIG
ORACLE = simple.ORACLE
RANDOM = simple.RANDOM

TFC1 = term_frequency.TFC1
TFC3 = term_frequency.TFC3
M_TDC = term_frequency.M_TDC
LEN_M_TDC = term_frequency.LEN_M_TDC

TrecOracleAxiom = trec.TrecOracleAxiom
TREC = trec.TREC

STMC1 = term_similarity.STMC1
STMC1_fastText = term_similarity.STMC1_fastText
STMC1_f = term_similarity.STMC1_f
STMC2 = term_similarity.STMC2
STMC2_fastText = term_similarity.STMC2_fastText
STMC2_f = term_similarity.STMC2_f
