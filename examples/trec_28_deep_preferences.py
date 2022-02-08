from pathlib import Path

# experiment_dir = Path(__file__).parent
experiment_dir = Path(
    "/mnt/ceph/storage/data-in-progress/data-research/web-search/ir-axioms"
)
pyterrier_dir = experiment_dir / "pyterrier"
cache_dir = experiment_dir / "cache"
base_results_dir = experiment_dir / "trec-system-runs"

# Start/initialize PyTerrier.
from pyterrier import started, init

if not started():
    init(home_dir=str(pyterrier_dir.absolute()), tqdm="auto")

from examples.trec_util import TrecTrack

track = TrecTrack(
    28, "deep", "msmarco-passage/trec-dl-2019", cache_dir, base_results_dir
)
retrieval_systems = [result % 10 for result in track.results]

from ir_axioms.axiom import (
    ArgUC, QTArg, QTPArg, aSL, PROX1, PROX2, PROX3, PROX4, PROX5, TFC1, TFC3,
    RS_TF, RS_TF_IDF, RS_BM25, RS_PL2, RS_QL, LNC1, TF_LNC, LB1, STMC1,
    STMC1_f, STMC2, STMC2_f, AND, LEN_AND, M_AND, LEN_M_AND, DIV, LEN_DIV, REG,
    ANTI_REG, M_TDC, LEN_M_TDC
)

axioms = [
    # ArgUC(),
    # QTArg(),
    # QTPArg(),
    aSL(),
    # LNC1(),
    # TF_LNC(),
    # LB1(),
    # PROX1(),
    # PROX2(),
    # PROX3(),
    # PROX4(),
    # PROX5(),
    # REG(), # Tie
    # ANTI_REG(), # Tie
    # AND(),
    # LEN_AND(),
    # M_AND(),
    # LEN_M_AND(),
    # DIV(),
    # LEN_DIV(),
    # RS_TF(), # Memory Overflow
    # RS_TF_IDF(), # Memory Overflow
    # RS_BM25(), # Memory Overflow
    # RS_PL2(), # Memory Overflow
    # RS_QL(), # Memory Overflow
    # TFC1(),
    # TFC3(), # Memory Overflow
    # M_TDC(),
    # LEN_M_TDC(),
    # STMC1(), # Slow
    # STMC1_f(), # Slow
    # STMC2(), # Slow
    # STMC2_f(), # Slow
]
from ir_axioms.backend.pyterrier.experiment import AxiomaticExperiment

axiomatic_experiment = AxiomaticExperiment(
    retrieval_systems=retrieval_systems,
    topics=track.dataset.get_topics(),
    qrels=track.dataset.get_qrels(),
    index=track.index,
    dataset=track.ir_dataset,
    axioms=axioms,
    filter_by_qrels=True,
    filter_by_topics=True,
    verbose=True,
    cache_dir=cache_dir,
)
print(axiomatic_experiment.preferences)
