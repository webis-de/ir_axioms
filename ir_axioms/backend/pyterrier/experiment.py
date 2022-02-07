from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Sequence, Optional, Union, Callable, NamedTuple

from ir_datasets import Dataset
from pandas import DataFrame, concat
from tqdm import tqdm

from ir_axioms.axiom import Axiom, OriginalAxiom
from ir_axioms.backend.pyterrier import ContentsAccessor
from ir_axioms.backend.pyterrier.axiom import OracleAxiom
from ir_axioms.backend.pyterrier.safe import Transformer, generic
from ir_axioms.backend.pyterrier.transformers import AxiomaticPreferences
from ir_axioms.backend.pyterrier.util import IndexRef, Index, Tokeniser


@dataclass(frozen=True)
class AxiomaticExperiment:
    retrieval_systems: Sequence[Transformer]
    topics: DataFrame
    qrels: DataFrame
    index: Union[Path, IndexRef, Index]
    axioms: Sequence[Axiom]
    filter_by_topics: bool = True
    filter_by_qrels: bool = True
    dataset: Optional[Union[Dataset, str]] = None
    contents_accessor: Optional[ContentsAccessor] = "text"
    tokeniser: Optional[Tokeniser] = None
    cache_dir: Optional[Path] = None
    verbose: bool = False

    @cached_property
    def _axioms(self) -> Sequence[Axiom]:
        return [
            OriginalAxiom(),
            OracleAxiom(self.topics, self.qrels),
            *self.axioms,
        ]

    @cached_property
    def _preferences_transformer(self) -> Transformer:
        return AxiomaticPreferences(
            axioms=self._axioms,
            index=self.index,
            dataset=self.dataset,
            contents_accessor=self.contents_accessor,
            tokeniser=self.tokeniser,
            cache_dir=self.cache_dir,
            verbose=False,
        )

    def _filter(self, ranking: DataFrame):
        if self.filter_by_topics:
            # Retain only queries that are contained in the topics.
            ranking = ranking[ranking["qid"].isin(self.topics["qid"])]
        if self.filter_by_qrels:
            # Retain only query-document pairs that are contained in the qrels.
            ranking = ranking[
                ranking["qid"].isin(self.qrels["qid"]) &
                ranking["docno"].isin(self.qrels["docno"])
                ]
        return ranking

    @cached_property
    def preferences(self) -> DataFrame:
        """
        Return a dataframe with each axiom's preference for each
        document pair, query and retrieval system.
        The returned dataframe has the following columns:
        - name: Retrieval system name
        - qid: Query ID
        - query: Query text
        - docno_a: Document A ID
        - score_a: Document A score
        - rank_a: Document A rank
        - label_a: Document A relevance label (if qrels were given)
        - docno_b: Document B ID
        - score_b: Document B score
        - rank_b: Document B rank
        - rank_b: Document B relevance label (if qrels were given)
        - original_preference: Preference from original retrieved ranking
        - oracle_preference: Preference from qrels (NaN if qrels are missing)
        - <axiom>_preference: Preference from each axiom <axiom>
        """
        systems = self.retrieval_systems
        if self.verbose:
            systems = tqdm(
                systems,
                desc="AxiomaticExperiment",
                unit="system",
            )
        return concat([
            (
                    system >>
                    generic(self._filter) >>
                    self._preferences_transformer
            ).transform(self.topics)
            for system in systems
        ])
