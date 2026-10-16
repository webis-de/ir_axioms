# Evaluating RAG Responses

<p class="subtitle">Compare RAG axioms to human judgments</p>

This walkthrough shows how to apply the [RAG axioms](../axioms/generation.md) to generated responses and how to evaluate the axioms against human judgments, as in our ICTIR 2025 paper [@merker2025axioms].
We measure how often an axiom expresses a preference (_decisiveness_) and how often its preferences agree with the human judgments (_consistency_).

Based on the notebook [`ictir2025.ipynb`](https://github.com/webis-de/ir_axioms/blob/main/experiments/ictir2025.ipynb).

## Setup

Install `ir_axioms` with the extras used in the paper, and download the spaCy model:

```shell
pip install "ir_axioms[pyterrier,keybert,sbert,fasttext]"
python -m spacy download en_core_web_sm
```

Then, clone the [repository](https://github.com/webis-de/ir_axioms) to get the data files in the [`data/`](https://github.com/webis-de/ir_axioms/tree/main/data) directory, and run the following code from the repository root.

```python
from pathlib import Path

data_dir = Path("data/")
cache_dir = Path("cache/")
crowd_responses_path = data_dir / "crowd" / "responses.jsonl.gz"
crowd_ratings_path = data_dir / "crowd" / "ratings.jsonl.gz"
```

## Load Requests and Responses

We use a corpus of RAG responses written by humans and LLMs in different styles, together with crowdsourced pairwise utility judgments.
Each request (query) has one response per kind and style.
We convert the data to [`GenerationRequest`][ir_axioms.model.generation.GenerationRequest] and [`GenerationResponse`][ir_axioms.model.generation.GenerationResponse] objects, so that all requests have the same responses in the same order:

```python
from collections.abc import Sequence
from pandas import DataFrame, read_json
from ir_axioms.model import GenerationRequest, GenerationResponse

responses = read_json(crowd_responses_path, lines=True)
ratings = read_json(crowd_ratings_path, lines=True)
responses = responses[responses["topic"].isin(ratings["query_id"])]
responses["name"] = responses["kind"] + "-" + responses["style"]


def load_requests_responses(
    df: DataFrame,
) -> list[tuple[GenerationRequest, list[GenerationResponse]]]:
    names = sorted(df["name"].unique())
    data = []
    for (query_id, query), df_query in df.groupby(["topic", "query"], sort=False):
        context = df_query["references_texts"].iloc[0]
        request = GenerationRequest(
            id=query_id,
            text=query,
            context=list(context) if isinstance(context, Sequence) else None,
        )
        responses_by_name = {
            row["name"]: GenerationResponse(id=row["name"], text=row["raw_text"])
            for _, row in df_query.iterrows()
        }
        data.append((
            request,
            [responses_by_name.get(name, GenerationResponse(id=name, text="")) for name in names],
        ))
    return data


data = load_requests_responses(responses)
```

## Oracle Preferences

The [`TrecRagCrowdAxiom`][ir_axioms.axiom.generation.crowd.TrecRagCrowdAxiom] derives preferences from the crowdsourced judgments for a given utility dimension (e.g., `"overall"`, `"coherence"`, `"broad coverage"`).
It serves as the ground truth ("ORACLE") for evaluating the axioms:

```python
from ir_axioms.axiom import TrecRagCrowdAxiom

oracle = TrecRagCrowdAxiom(
    utility_type="overall",
    responses_path=crowd_responses_path,
    ratings_path=crowd_ratings_path,
)
```

!!! tip "Nugget-based oracle"

    To use the nugget assignments of the TREC 2024 RAG track [@pradeep2024initial] instead, load the corresponding responses and use `TrecRagNuggetAxiom(assignments_path=data_dir / "nugget_assignment.20241218.jsonl", score_type="vital")`.

## Axioms

We compare three groups of axioms:
[adapted retrieval axioms](../axioms/generation.md#adapted-retrieval-axioms),
[generation-specific axioms](../axioms/generation.md#coherence), and
majority votes of all axioms.
The margins below are the ones used in the paper; the aspect-based axioms use spaCy entities as aspects.
Because some axioms are costly (e.g., grammar checking with LanguageTool), we cache all axioms on disk:

```python
from ir_axioms.axiom import (
    GEN_TFC1, GEN_LNC1, GEN_TF_LNC, GEN_REG, GEN_AND, GEN_DIV, GEN_STMC1, GEN_STMC2,
    GEN_PROX1, GEN_PROX2, GEN_PROX3, GEN_PROX4, GEN_PROX5, GEN_aSL,
    COH1, COH2, COV1, COV2, COV3, CONS1, CONS2, CONS3, CORR1, CLAR1, CLAR2,
    MajorityVoteAxiom,
)
from ir_axioms.tools import SpacyEntitiesAspectExtraction

entities = SpacyEntitiesAspectExtraction()

retrieval_axioms = {
    "GEN-TFC1": GEN_TFC1(), "GEN-LNC1": GEN_LNC1(), "GEN-TF-LNC": GEN_TF_LNC(),
    "GEN-REG": GEN_REG(), "GEN-AND": GEN_AND(), "GEN-DIV": GEN_DIV(),
    "GEN-STMC1": GEN_STMC1(), "GEN-STMC2": GEN_STMC2(),
    "GEN-PROX1": GEN_PROX1(), "GEN-PROX2": GEN_PROX2(), "GEN-PROX3": GEN_PROX3(),
    "GEN-PROX4": GEN_PROX4(), "GEN-PROX5": GEN_PROX5(), "GEN-aSL": GEN_aSL(),
}
generation_axioms = {
    "COH1": COH1(margin_fraction=0.75),
    "COH2": COH2(margin_fraction=0.75),
    "COV1": COV1(aspect_extraction=entities, margin_fraction=0.5),
    "COV2": COV2(aspect_extraction=entities, margin_fraction=0.5),
    "COV3": COV3(aspect_extraction=entities, margin_fraction=0.5),
    "CONS1": CONS1(aspect_extraction=entities, margin_fraction=0.5),
    "CONS2": CONS2(margin_fraction=0.5),
    "CONS3": CONS3(margin_fraction=0.5),
    "CORR1": CORR1(margin_fraction=0.75),
    "CLAR1": CLAR1(margin_fraction=0.5),
    "CLAR2": CLAR2(margin_fraction=0.5),
}
vote_axioms = {
    "VOTE-all": MajorityVoteAxiom(
        axioms=[*retrieval_axioms.values(), *generation_axioms.values()],
        minimum_votes=0.3,
    ),
}

axioms = {
    name: axiom.cached(cache_dir / "axioms" / f"{name}.cache")
    for name, axiom in {**retrieval_axioms, **generation_axioms, **vote_axioms}.items()
}
```

## Compute Preferences

For each request, we compute the preference matrices of the oracle and of all axioms, and normalize the preferences to −1, 0, or 1:

```python
from numpy import sign, stack
from tqdm.auto import tqdm

oracle_preferences = sign(stack([
    oracle.preferences(input=request, outputs=outputs)
    for request, outputs in tqdm(data, desc="ORACLE")
]))
axiom_preferences = {
    name: sign(stack([
        axiom.preferences(input=request, outputs=outputs)
        for request, outputs in data
    ]))
    for name, axiom in tqdm(axioms.items(), desc="Axioms")
}
```

## Decisiveness and Consistency

We evaluate each axiom by two measures, ignoring the trivial comparisons of each response with itself:

- **Decisiveness:** the proportion of response pairs for which the axiom expresses a preference.
- **Consistency:** the proportion of the axiom's preferences that do not contradict the oracle's preference.

The harmonic mean of both measures summarizes the axiom's effectiveness:

```python
from numpy import eye

num_outputs = oracle_preferences.shape[-1]
off_diagonal = ~eye(num_outputs, dtype=bool)

results = []
for name, preferences in axiom_preferences.items():
    decisive = (preferences != 0) & off_diagonal
    decisiveness = decisive.sum() / (off_diagonal.sum() * len(preferences))
    consistent = decisive & (abs(preferences - oracle_preferences) <= 1)
    consistency = consistent.sum() / decisive.sum() if decisive.sum() > 0 else 0
    effectiveness = (
        2 * decisiveness * consistency / (decisiveness + consistency)
        if decisiveness + consistency > 0 else 0
    )
    results.append({
        "axiom": name,
        "decisiveness": decisiveness,
        "consistency": consistency,
        "effectiveness": effectiveness,
    })

DataFrame(results).sort_values("effectiveness", ascending=False).round(2)
```

The consistency above is _relaxed_: an axiom preference does not contradict a zero oracle preference.
For a _strict_ consistency, count only exact matches, i.e., `preferences == oracle_preferences`.

## Learn More

- The [ICTIR 2025 notebook](https://github.com/webis-de/ir_axioms/blob/main/experiments/ictir2025.ipynb) also evaluates on the TREC 2024 RAG nugget assignments, compares to LLM-based ORIG axioms ([`TrecRagLlmOrigAxiom`][ir_axioms.axiom.generation.llm.TrecRagLlmOrigAxiom]), and contains the code for all figures and tables of the paper.
- The [RAG axioms](../axioms/generation.md) page maps the axioms to the utility dimensions of related evaluation frameworks.
