from dataclasses import dataclass
from pathlib import Path

from pyterrier import started, init
from tqdm import tqdm

from ir_axioms.axiom import (
    ArgUC, QTArg, QTPArg, aSL, LNC1, TF_LNC, LB1, PROX1, PROX2, PROX3, PROX4,
    PROX5, REG, REG_f, ANTI_REG, ANTI_REG_fastText, ASPECT_REG,
    ASPECT_REG_fastText, AND, LEN_AND, M_AND, LEN_M_AND, DIV, LEN_DIV, RS_TF,
    RS_TF_IDF, RS_BM25, RS_PL2, RS_QL, TFC1, TFC3, M_TDC, LEN_M_TDC, STMC1,
    STMC1_fastText, STMC2, STMC2_fastText
)

if not started():
    init(tqdm="auto")

from pyterrier import Transformer
from pyterrier.datasets import get_dataset
from pyterrier.index import IterDictIndexer
from pyterrier.io import read_results
from ir_axioms.backend.pyterrier.experiment import AxiomaticExperiment


@dataclass(frozen=True)
class Track:
    edition: int
    track: str
    dataset: str
    contents_field: str


tracks = [
    # Track(
    #     edition=28,
    #     track="deep.documents",
    #     dataset="msmarco-document/trec-dl-2019/judged",
    #     contents_field="body",
    # ),
    # Track(
    #     edition=28,
    #     track="deep.passages",
    #     dataset="msmarco-passage/trec-dl-2019/judged",
    #     contents_field="text",
    # ),
    Track(
        edition=29,
        track="deep.documents",
        dataset="msmarco-document/trec-dl-2020/judged",
        contents_field="body",
    ),
    # Track(
    #     edition=29,
    #     track="deep.passages",
    #     dataset="msmarco-passage/trec-dl-2020/judged",
    #     contents_field="text",
    # ),
]
depths = [
    10,
    20,
    # 50,
]
configurations = [
    (track, depth)
    for depth in depths
    for track in tracks
]

axioms = [
    ArgUC(),
    QTArg(),
    QTPArg(),
    aSL(),
    LNC1(),
    TF_LNC(),
    LB1(),
    PROX1(),
    PROX2(),
    PROX3(),
    PROX4(),
    PROX5(),
    REG(),
    REG_f(),
    ANTI_REG(),
    ANTI_REG_fastText(),
    ASPECT_REG(),
    ASPECT_REG_fastText(),
    AND(),
    LEN_AND(),
    M_AND(),
    LEN_M_AND(),
    DIV(),
    LEN_DIV(),
    RS_TF(),
    RS_TF_IDF(),
    RS_BM25(),
    RS_PL2(),
    RS_QL(),
    TFC1(),
    TFC3(),
    M_TDC(),
    LEN_M_TDC(),
    STMC1(),
    STMC1_fastText(),
    STMC2(),
    STMC2_fastText(),
]

results_dir = Path(__file__).parent
cache_dir = Path(__file__).parent / "cache"
indices_dir = cache_dir / "indices"
runs_base_dir = Path(
    "/mnt/ceph/storage/data-in-progress/data-research/"
    "web-search/web-search-trec/trec-system-runs"
)
cache_dir.mkdir(exist_ok=True)
indices_dir.mkdir(exist_ok=True)

print(f"Reading runs from {runs_base_dir.absolute()}")
print(f"Storing results in {results_dir.absolute()}")
print(f"Storing cache in {cache_dir.absolute()}")
print(f"Storing indices in {indices_dir.absolute()}")

for track, depth in configurations:
    dataset = get_dataset(f"irds:{track.dataset}")

    index_dir = indices_dir / track.dataset.split("/")[0]
    runs_dir = runs_base_dir / f"trec{track.edition}" / track.track
    run_files = list(runs_dir.iterdir())

    if not index_dir.exists():
        indexer = IterDictIndexer(str(index_dir.absolute()))
        indexer.index(
            dataset.get_corpus_iter(),
            fields=[track.contents_field]
        )

    run = [
        Transformer.from_df(read_results(result_file))
        for result_file in tqdm(run_files, desc="Load runs")
    ]
    run_name = [
        result_file.stem.replace("input.", "")
        for result_file in run_files
    ]

    axioms_cached = [~axiom for axiom in axioms]
    axiom_names = [axiom.name for axiom in axioms]

    experiment = AxiomaticExperiment(
        retrieval_systems=run,
        topics=dataset.get_topics(),
        qrels=dataset.get_qrels(),
        index=index_dir,
        dataset=track.dataset,
        contents_accessor=track.contents_field,
        axioms=axioms,
        axiom_names=axiom_names,
        depth=depth,
        filter_by_qrels=False,
        filter_by_topics=False,
        verbose=True,
        cache_dir=cache_dir,
    )
    preferences = experiment.preferences
    result_file = results_dir / (
        f"trec-{track.edition}-{track.track}-preferences-"
        f"all-axioms-depth-{depth}.csv"
    )
    preferences.to_csv(result_file)
