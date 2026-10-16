# Parallel Preferences

<p class="subtitle">Compute preference matrices for many TREC runs in parallel</p>

This walkthrough shows how `ir_axioms` computes axiomatic preferences for all runs submitted to the passage ranking task of the TREC 2020 Deep Learning track, distributing the computation across multiple cores.

Based on the notebook [`pyterrier_preferences_parallel.ipynb`](https://github.com/webis-de/ir_axioms/blob/main/experiments/pyterrier_preferences_parallel.ipynb)
[![Open in Colab](https://img.shields.io/badge/open%20in-colab-informational?style=flat-square)](https://colab.research.google.com/github/webis-de/ir_axioms/blob/main/experiments/pyterrier_preferences_parallel.ipynb).

## Setup

Follow the [common setup](index.md#common-setup) to load the datasets, build the index, and connect `ir_axioms` to the index.
This walkthrough does not need a baseline retrieval model, as we analyze existing runs.

## Load the Submitted Runs

Download the run files submitted to TREC 2020 Deep Learning (passage ranking) from the [TREC results archive](https://trec.nist.gov/results.html) (access requires a password from NIST) and store them in a directory of your choice.
Then, load all runs into a single data frame:

```python
from pathlib import Path
from pandas import concat
from pyterrier.io import read_results
from tqdm.auto import tqdm

run_dir = Path("runs/trec29/deep.passages/")

run_files = list(run_dir.iterdir())
all_results = concat([
    read_results(str(run_file))
    for run_file in tqdm(run_files, desc="Load runs", unit="run")
])
```

## Define the Axioms

We compute the preferences of a broad set of axioms:

```python
from ir_axioms.axiom import (
    LNC1, TF_LNC, LB1,
    PROX1, PROX2, PROX3, PROX4, PROX5,
    REG, ANTI_REG, ASPECT_REG,
    AND, LEN_AND, M_AND, LEN_M_AND, DIV, LEN_DIV,
    TFC1, TFC3, M_TDC, LEN_M_TDC,
    STMC1, STMC2,
)

axioms = {
    "LNC1": LNC1(), "TF_LNC": TF_LNC(), "LB1": LB1(),
    "PROX1": PROX1(), "PROX2": PROX2(), "PROX3": PROX3(), "PROX4": PROX4(), "PROX5": PROX5(),
    "REG": REG(), "ANTI_REG": ANTI_REG(), "ASPECT_REG": ASPECT_REG(),
    "AND": AND(), "LEN_AND": LEN_AND(), "M_AND": M_AND(), "LEN_M_AND": LEN_M_AND(),
    "DIV": DIV(), "LEN_DIV": LEN_DIV(),
    "TFC1": TFC1(), "TFC3": TFC3(), "M_TDC": M_TDC(), "LEN_M_TDC": LEN_M_TDC(),
    "STMC1": STMC1(), "STMC2": STMC2(),
}
```

## Compute Preferences in Parallel

We create a PyTerrier pipeline that takes the top-10 results of each run and computes all pairwise preferences with the `AxiomaticPreferences` transformer:

```python
from pyterrier import Transformer
from ir_axioms.integrations import AxiomaticPreferences

compute_preferences = Transformer.from_df(all_results) % 10 >> AxiomaticPreferences(
    axioms=list(axioms.values()),
    axiom_names=list(axioms.keys()),
    verbose=True,
)
```

To speed up the computation, we distribute it across 4 cores with PyTerrier's `parallel()`, and measure the elapsed time:

```python
from time import perf_counter

compute_preferences = compute_preferences.parallel(4)

start = perf_counter()
preferences = compute_preferences.transform(dataset_test.get_topics())
elapsed_seconds = perf_counter() - start

print(f"Elapsed time: {elapsed_seconds:.2f}s")
print(f"Preferences per second: {len(preferences) / elapsed_seconds:.2f}")
```

The resulting data frame contains one row per run, query, and document pair, and one `<axiom>_preference` column per axiom.

!!! tip "Parallelizing a single axiom"

    Independent of PyTerrier, you can also parallelize the preference matrix computation of a single axiom with `axiom.parallel(n_jobs=4)`.

## Parallelization Backends

The parallelization is implemented in [PyTerrier](https://pyterrier.readthedocs.io/en/latest/ext/pyterrier-alpha/parallel.html).
By default, it uses the [Joblib](https://joblib.readthedocs.io/) backend, which splits the computation per query and runs it on multiple cores of the same machine.

You can also use the [Ray](https://www.ray.io/) backend to connect to remote clusters and distribute the workload across multiple machines (e.g., on [Kubernetes](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) or [Slurm](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html)).
Please refer to the [Ray documentation](https://docs.ray.io/en/latest/) for instructions on how to connect to your cluster.
