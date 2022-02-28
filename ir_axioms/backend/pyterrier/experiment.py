from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Sequence, Optional, Union, Callable

from ir_datasets import Dataset
from pandas import DataFrame, concat
from tqdm.auto import tqdm

from ir_axioms.axiom import Axiom, OriginalAxiom, OracleAxiom
from ir_axioms.backend.pyterrier import ContentsAccessor
from ir_axioms.backend.pyterrier.safe import Transformer
from ir_axioms.backend.pyterrier.transformer_utils import (
    FilterTopicsTransformer, FilterQrelsTransformer, JoinQrelsTransformer
)
from ir_axioms.backend.pyterrier.transformers import AxiomaticPreferences
from ir_axioms.backend.pyterrier.util import IndexRef, Index, Tokeniser
from ir_axioms.model import JudgedRankedDocument


@dataclass(frozen=True)
class AxiomaticExperiment:
    retrieval_systems: Sequence[Transformer]
    topics: DataFrame
    qrels: DataFrame
    index: Union[Path, IndexRef, Index]
    axioms: Sequence[Axiom]
    axiom_names: Optional[Sequence[str]] = None
    depth: Optional[int] = 10
    filter_by_qrels: bool = True
    filter_by_topics: bool = False
    filter_pairs: Optional[Callable[
        [JudgedRankedDocument, JudgedRankedDocument],
        bool
    ]] = None
    dataset: Optional[Union[Dataset, str]] = None
    contents_accessor: Optional[ContentsAccessor] = "text"
    tokeniser: Optional[Tokeniser] = None
    cache_dir: Optional[Path] = None
    verbose: bool = False

    @cached_property
    def _axioms(self) -> Sequence[Axiom]:
        return [
            OriginalAxiom(),
            OracleAxiom(),
            *self.axioms,
        ]

    @cached_property
    def _axiom_names(self) -> Sequence[str]:
        names: Sequence[str]
        if (
                self.axiom_names is not None and
                len(self.axiom_names) == len(self.axioms)
        ):
            names = self.axiom_names
        else:
            names = [str(axiom) for axiom in self.axioms]
        return [
            OriginalAxiom.name,
            OracleAxiom.name,
            *names,
        ]

    @cached_property
    def _filter_topics(self) -> Transformer:
        return FilterTopicsTransformer(self.topics)

    @cached_property
    def _filter_qrels(self) -> Transformer:
        return FilterQrelsTransformer(self.qrels)

    @cached_property
    def _join_qrels(self) -> Transformer:
        return JoinQrelsTransformer(self.qrels)

    def _pipeline(self, system: Transformer) -> Transformer:
        pipeline = system
        if self.depth is not None:
            # noinspection PyTypeChecker
            pipeline = pipeline % self.depth
        if self.filter_by_topics:
            pipeline = pipeline >> self._filter_topics
        if self.filter_by_qrels:
            pipeline = pipeline >> self._filter_qrels
        pipeline = pipeline >> self._join_qrels
        return pipeline

    @cached_property
    def _preferences_transformer(self) -> Transformer:
        return AxiomaticPreferences(
            axioms=self._axioms,
            axiom_names=self._axiom_names,
            index=self.index,
            dataset=self.dataset,
            contents_accessor=self.contents_accessor,
            filter_pairs=self.filter_pairs,
            tokeniser=self.tokeniser,
            cache_dir=self.cache_dir,
            verbose=self.verbose,
        )

    def _preferences_pipeline(self, system: Transformer) -> Transformer:
        pipeline = self._pipeline(system)
        pipeline = pipeline >> self._preferences_transformer
        return pipeline

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
        - ORIG_preference: Preference from original retrieved ranking
        - ORACLE_preference: Preference from qrels (NaN if qrels are missing)
        - <axiom>_preference: Preference from each axiom <axiom>
        """
        systems = self.retrieval_systems
        # noinspection PyTypeChecker
        pipelines = [
            self._preferences_pipeline(system)
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
