# Quickstart

<p class="subtitle">Your first axiomatic preferences</p>

This quickstart computes axiomatic preferences for a handful of documents without any retrieval index.
All you need is a [basic installation](installation.md).

## Queries and Documents

Axioms compare two _outputs_ (e.g., documents) for the same _input_ (e.g., a query).
For retrieval axioms, inputs are [`Query`](../api/model.md) and outputs are [`Document`](../api/model.md) objects:

```python
from ir_axioms.model import Query, Document

query = Query(id="q1", text="axiomatic information retrieval")
documents = [
    Document(
        id="d1",
        text="Retrieval axioms formalize constraints for information retrieval models.",
        rank=1,
        score=12.3,
    ),
    Document(
        id="d2",
        text="Axiomatic thinking in information retrieval: axiomatic constraints guide retrieval model design.",
        rank=2,
        score=11.0,
    ),
    Document(
        id="d3",
        text="A recipe for apple pie with cinnamon.",
        rank=3,
        score=4.2,
    ),
]
```

The optional `rank` and `score` fields are used by the [ORIG](../axioms/utility.md#orig) axiom, and the optional `relevance` field by the [ORACLE](../axioms/utility.md#oracle) axiom.

## Pairwise Preferences

Each axiom is a class, and most axioms are also available under a short alias like `TFC1` (see the [axiom catalog](../axioms/index.md)).
Calling the alias creates an axiom instance with all its dependencies (e.g., tokenizers) resolved automatically.

```python
from ir_axioms.axiom import TFC1

axiom = TFC1()
axiom.preference(input=query, output1=documents[0], output2=documents[1])
# -1.0: TFC1 prefers d2, which contains more query term occurrences.
```

A preference is **positive** if the first output should be ranked higher, **negative** if the second output should be ranked higher, and **zero** if the axiom has no preference (e.g., because a precondition is not met).

To compare all outputs at once, compute the preference matrix:

```python
axiom.preferences(input=query, outputs=documents)
# array([[ 0., -1.,  1.],
#        [ 1.,  0.,  1.],
#        [-1., -1.,  0.]])
```

The entry in row \(i\) and column \(j\) is the preference between the \(i\)-th and \(j\)-th output.

## Combining Axioms

Axioms can be combined with Python operators.
For example, require that two axioms agree (`&`), and fall back to the original ranking if they do not (`|`):

```python
from ir_axioms.axiom import TFC1, PROX1, ORIG

axiom = (TFC1() & PROX1()) | ORIG()
```

Or let multiple axioms vote and only keep preferences with an absolute majority:

```python
from ir_axioms.axiom import LNC1, PROX4, MajorityVoteAxiom

axiom = MajorityVoteAxiom(
    axioms=[TFC1(), LNC1(), PROX1(), PROX4()],
    minimum_votes=0.5,
) | ORIG()
```

All operators are listed in the [core concepts](concepts.md#combining-axioms).

## Re-Ranking

Based on an axiom's preferences, re-rank the documents with the KwikSort algorithm [@ailon2008aggregating]:

```python
reranked = axiom.rerank_kwiksort(input=query, ranking=documents)
[document.id for document in reranked]
```

## Axioms for Generated Responses

RAG axioms compare two generated responses ([`GenerationResponse`](../api/model.md)) for the same request ([`GenerationRequest`](../api/model.md)), which optionally includes the retrieved context:

```python
from ir_axioms.axiom import CLAR2, COV1, GEN_TFC1
from ir_axioms.model import GenerationRequest, GenerationResponse

request = GenerationRequest(
    id="q1",
    text="What are retrieval axioms?",
    context=["Retrieval axioms are formal constraints that good retrieval models should satisfy."],
)
responses = [
    GenerationResponse(
        id="a",
        text="Retrieval axioms are formal constraints on ranking functions [1]. They help to analyze retrieval models.",
    ),
    GenerationResponse(id="b", text="Axioms. Stuff."),
]

COV1().preferences(input=request, outputs=responses)  # Coverage
CLAR2().preferences(input=request, outputs=responses)  # Clarity (readability)
GEN_TFC1().preferences(input=request, outputs=responses)  # Adapted retrieval axiom
```

## Next Steps

- Learn the [core concepts](concepts.md) behind axioms, preferences, and tools.
- Use axioms with a real retrieval index via the [PyTerrier integration](integrations.md).
- Follow the [hands-on tutorial](../walkthroughs/hands-on.md) on TREC Deep Learning data.
