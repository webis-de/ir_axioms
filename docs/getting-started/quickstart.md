# Quickstart

<p class="subtitle">Compute axiomatic preferences</p>

This quickstart computes axiomatic preferences for a handful of documents without any retrieval index.
All you need is a [basic installation](installation.md).
You can either run these commands in the [Python REPL](https://docs.python.org/3/tutorial/appendix.html#tut-interac), or create a [Jupyter notebook](https://docs.jupyter.org/en/latest/) (`pip install jupyter`).

## Queries and Documents

Axiomatic constraints in general compare two "outputs" (e.g., documents) for the same "input" (e.g., a query).
For retrieval axioms, inputs are [`Query`][ir_axioms.model.retrieval.Query] and outputs are [`Document`][ir_axioms.model.retrieval.Document] objects. Define some dummy data to play around with:

```python
from ir_axioms.model import Query, Document

query = Query(
    id="q1",
    text="axiomatic information retrieval",
)
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

In `ir_axioms`, each axiom is implemented as a class, and most axioms are available under a short alias like `TFC1` (see the [axiom catalog](../axioms/index.md)).
Calling the alias creates an axiom instance with all its dependencies (e.g., tokenizers) resolved automatically.

```python
from ir_axioms.axiom import TFC1

axiom = TFC1()
```

Let's see which of the given documents, the TFC1 axiom prefers [@fang2004formal]:

```python
axiom.preference(
    input=query,
    output1=documents[0],
    output2=documents[1],
)
# Output: -1
```

The pairwise preference is either (learn more on the concept of [preferences](concepts.md#axioms-and-preferences)):

- _positive_ if the first output should be ranked higher,
- _negative_ if the second output should be ranked higher, or
- _zero_ if the axiom has no preference, e.g., because a precondition is not met
  
So, TFC1 has preferred the second document because it contains more query term occurrences.

To compare all outputs at once, compute the preference matrix:

```python
axiom.preferences(input=query, outputs=documents)
# Output: array([[ 0., -1.,  1.],
#                [ 1.,  0.,  1.],
#                [-1., -1.,  0.]])
```

The entry in row \(i\) and column \(j\) is the preference between the \(i\)-th and \(j\)-th output.

## Combining Axioms

Axioms can be combined with Python operators.
For example, to require that two given axioms agree (`&`), and to fall back to the original ranking if they do not (`|`):

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

Based on an axiom's preferences, re-rank the documents with the KwikSort algorithm [@ailon2008aggregating,@hagen2016axiomatic]:

```python
axiom = TFC1()
reranked = axiom.rerank_kwiksort(input=query, ranking=documents)
[document.id for document in reranked]
# Output: ['d2', 'd1', 'd3']
```

Because TFC1 prefers the second document, it is pushed to the top of the ranking.

## Axioms for Generated Responses

Axiomatic constraints can similarly be applied to retrieval-augmented generation (RAG), to compare the generated responses ([`GenerationResponse`][ir_axioms.model.generation.GenerationResponse]) from two RAG systems for the same request ([`GenerationRequest`][ir_axioms.model.generation.GenerationRequest], optionally includes the retrieved context):

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

In this case, all axioms would prefer the first response.

## Next Steps

- Learn the [core concepts](concepts.md) behind axioms, preferences, and tools.
- Use axioms with a real retrieval index via the [PyTerrier integration](integrations.md).
- Follow the [hands-on tutorial](../walkthroughs/hands-on.md) on TREC Deep Learning data.
