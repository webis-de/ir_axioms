from ir_axioms.axiom import base, arithmetic, cache, conversion, argumentative

# Re-export from child modules.
Axiom = base.Axiom
NormalizedAxiom = base.NormalizedAxiom
WeightedAxiom = arithmetic.WeightedAxiom
AggregatedAxiom = arithmetic.AggregatedAxiom
CachedAxiom = cache.CachedAxiom
AxiomLike = conversion.AxiomLike
to_axiom = conversion.to_axiom

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
