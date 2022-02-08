from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import List

from ir_datasets import load
from pandas import DataFrame
from pyterrier import Transformer
from pyterrier.datasets import IRDSDataset, get_dataset, Dataset
from pyterrier.index import IterDictIndexer
from pyterrier.io import read_results
from pyterrier.transformer import TransformerBase
from tqdm.auto import tqdm


@dataclass(frozen=True, unsafe_hash=True)
class TrecTrack:
    edition: int
    track: str
    dataset_name: str
    cache_dir: Path
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
    def index(self) -> Path:
        # Load documents and build index.
        index_dir = self.cache_dir / "indices" / self.dataset_name
        if not index_dir.exists():
            # Don't forget to include the 'text' field in the meta index.
            indexer = IterDictIndexer(str(index_dir))
            indexer.index(
                self.dataset.get_corpus_iter(),
                fields=["text"]
            )
        return index_dir

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
        return [
            _TrecTrackSourceTransformer(
                str(path.absolute()),
                dataset,
            )
            for path in files
        ]


@dataclass(frozen=True)
class _TrecTrackSourceTransformer(TransformerBase):
    filename: str
    dataset: IRDSDataset

    @cached_property
    def _results(self) -> DataFrame:
        return read_results(
            self.filename,
            dataset=self.dataset,
        )

    def transform(self, topics):
        assert "qid" in topics.columns
        results = self._results

        columns = ["qid"]
        results_contain_query = "query" in results.columns
        topics_contains_query = "query" in topics.columns
        if not results_contain_query and topics_contains_query:
            columns.append("query")
        retrieved = topics[columns].merge(results, on="qid")
        return retrieved
