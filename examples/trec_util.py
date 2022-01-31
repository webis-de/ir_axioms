from dataclasses import dataclass
from functools import cached_property
from itertools import islice
from pathlib import Path
from typing import List

from ir_datasets import load
from pyterrier import Transformer, IndexRef
from pyterrier.datasets import IRDSDataset, get_dataset, Dataset
from pyterrier.index import IterDictIndexer
from pyterrier.io import read_results
from tqdm import tqdm


@dataclass(frozen=True, unsafe_hash=True)
class TrecTrack:
    edition: int
    track: str
    dataset_name: str
    base_results_dir: Path = Path(
        "/mnt/ceph/storage/data-in-progress/data-research/"
        "web-search/web-search-trec/trec-system-runs"
    )

    @cached_property
    def dataset(self) -> IRDSDataset:
        return get_dataset(f"irds:{self.dataset_name}")

    @cached_property
    def ir_dataset(self) -> Dataset:
        return load(self.dataset_name)

    @cached_property
    def index(self) -> IndexRef:
        # Load documents and build index.
        index_dir = Path(f"./data/indices/{self.dataset_name}")
        index_ref: IndexRef
        if index_dir.exists():
            index_ref = IndexRef.of(str(index_dir.absolute()))
        else:
            # Don't forget to include the 'text' field in the meta index.
            indexer = IterDictIndexer(str(index_dir))
            index_ref = indexer.index(
                self.dataset.get_corpus_iter(),
                fields=["text"]
            )
        return index_ref

    @cached_property
    def result_dir(self) -> Path:
        return self.base_results_dir / f"trec{self.edition}" / self.track

    @cached_property
    def results(self) -> List[Transformer]:
        dataset = self.dataset
        num_files = sum(1 for _ in self.result_dir.iterdir())
        files = tqdm(
            self.result_dir.iterdir(),
            desc="Read results",
            unit="run",
            total=num_files,
        )
        files = islice(files, 2)
        return [
            Transformer.from_df(
                read_results(
                    str(path.absolute()),
                    dataset=dataset,
                )
            )
            for path in files
        ]
