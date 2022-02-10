from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Sequence, Optional, Union

from ir_datasets import Dataset
from pandas import DataFrame, concat
from tqdm.auto import tqdm

from ir_axioms.axiom import Axiom, OriginalAxiom
from ir_axioms.backend.pyterrier import ContentsAccessor
from ir_axioms.backend.pyterrier.axiom import OracleAxiom
from ir_axioms.backend.pyterrier.safe import (
    Transformer, IdentityTransformer
)
from ir_axioms.backend.pyterrier.transformer_utils import (
    FilterTopicsTransformer, FilterQrelsTransformer
)
from ir_axioms.backend.pyterrier.transformers import AxiomaticPreferences
from ir_axioms.backend.pyterrier.util import IndexRef, Index, Tokeniser


@dataclass(frozen=True)
class AxiomaticExperiment:
    retrieval_systems: Sequence[Transformer]
    topics: DataFrame
    qrels: DataFrame
    index: Union[Path, IndexRef, Index]
    axioms: Sequence[Axiom]
    filter_by_qrels: bool = True
    filter_by_topics: bool = False
    dataset: Optional[Union[Dataset, str]] = None
    contents_accessor: Optional[ContentsAccessor] = "text"
    tokeniser: Optional[Tokeniser] = None
    cache_dir: Optional[Path] = None
    verbose: bool = False

    @cached_property
    def _axioms(self) -> Sequence[Axiom]:
        axioms = [
            OriginalAxiom(),
            OracleAxiom(self.topics, self.qrels),
            *self.axioms,
        ]
        return [axiom for axiom in axioms]

    @cached_property
    def _filter_transformer(self) -> Transformer:
        pipeline = IdentityTransformer()
        if self.filter_by_topics:
            pipeline = pipeline >> FilterTopicsTransformer(self.topics)
        if self.filter_by_qrels:
            pipeline = pipeline >> FilterQrelsTransformer(self.qrels)
        return pipeline

    @cached_property
    def _preferences_transformer(self) -> Transformer:
        return AxiomaticPreferences(
            axioms=self._axioms,
            index=self.index,
            dataset=self.dataset,
            contents_accessor=self.contents_accessor,
            tokeniser=self.tokeniser,
            cache_dir=self.cache_dir,
            verbose=self.verbose,
        )

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
        pipelines = [
            ~(
                    system >>
                    self._filter_transformer >>
                    self._preferences_transformer
            )
            for system in systems
        ]
        if self.verbose:
            pipelines = tqdm(
                pipelines,
                desc="Computing system axiomatic preferences",
                unit="system",
            )
        return concat([
            pipeline.transform(self.topics)
            for pipeline in pipelines
        ])
