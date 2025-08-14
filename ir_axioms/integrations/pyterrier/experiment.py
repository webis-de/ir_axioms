from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed() or TYPE_CHECKING:
    from dataclasses import dataclass
    from functools import reduce, cached_property
    from math import nan
    from operator import or_
    from typing import Sequence, Optional, Literal, Iterable

    from pandas import DataFrame, concat
    from pyterrier import Transformer
    from tqdm.auto import tqdm

    from ir_axioms.axiom.base import Axiom
    from ir_axioms.axiom.retrieval.simple import ORIG, ORACLE
    from ir_axioms.integrations.pyterrier.transformers import AxiomaticPreferences
    from ir_axioms.integrations.pyterrier.utils import (
        FilterTopicsTransformer,
        FilterQrelsTransformer,
        JoinQrelsTransformer,
        AddNameTransformer,
    )

    @dataclass(frozen=True, kw_only=True)
    class AxiomaticExperiment:
        retrieval_systems: Sequence[Transformer]
        names: Optional[Sequence[str]] = None
        topics: DataFrame
        qrels: DataFrame
        axioms: Sequence[Axiom]
        axiom_names: Optional[Sequence[str]] = None
        depth: Optional[int] = 10
        filter_by_qrels: bool = True
        filter_by_topics: bool = False
        text_field: Optional[str] = "text"
        parallel_jobs: int = 1
        parallel_backend: Literal["joblib", "ray"] = "joblib"
        verbose: bool = False

        @cached_property
        def _additional_axioms(self) -> Sequence[Axiom]:
            return [
                ORIG(),
                ORACLE(),
            ]

        @cached_property
        def _axiom_names(self) -> Sequence[str]:
            names: Sequence[str]
            if self.axiom_names is not None:
                if len(self.axiom_names) != len(self.axioms):
                    raise ValueError("Number of axioms and names must match.")
                names = self.axiom_names
            else:
                names = [str(axiom) for axiom in self.axioms]
            return names

        @cached_property
        def _additional_axiom_names(self) -> Sequence[str]:
            return ["ORIG", "ORACLE"]

        @cached_property
        def _retrieval_system_names(self) -> Sequence[str]:
            names: Sequence[str]
            if self.names is not None:
                if len(self.names) != len(self.retrieval_systems):
                    raise ValueError("Number of axioms and names must match.")
                names = self.names
            else:
                names = [str(system) for system in self.retrieval_systems]
            return names

        def _preferences_pipeline(
            self,
            system: Transformer,
            name: str,
        ) -> Transformer:
            # Load original retrieval system.
            pipeline = system
            # Cutoff at rank k
            if self.depth is not None:
                # noinspection PyTypeChecker
                pipeline = pipeline % self.depth
            # Remove results with unknown topics.
            if self.filter_by_topics:
                pipeline = pipeline >> FilterTopicsTransformer(self.topics)
            # Remove results without judgments.
            if self.filter_by_qrels:
                pipeline = pipeline >> FilterQrelsTransformer(self.qrels)
            # Add relevance labels.
            pipeline = pipeline >> JoinQrelsTransformer(self.qrels)
            # Compute preferences
            pipeline = pipeline >> AxiomaticPreferences(
                axioms=[
                    *self._additional_axioms,
                    *self.axioms,
                ],
                axiom_names=[
                    *self._additional_axiom_names,
                    *self._axiom_names,
                ],
                text_field=self.text_field,
                verbose=self.verbose,
            )
            # Parallelize computation.
            if self.parallel_jobs != 1:
                pipeline = pipeline.parallel(
                    self.parallel_jobs,
                    self.parallel_backend,
                )
            # Add system name.
            pipeline = pipeline >> AddNameTransformer(name)
            return pipeline

        @cached_property
        def preferences(self) -> DataFrame:
            """
            Return a dataframe with each axiom's preference for each
            document pair, query and retrieval system.

            The returned dataframe has the following columns:
            - name: Retrieval system name (if found in source)
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
            names = self._retrieval_system_names
            pipelines: Iterable[Transformer] = [
                self._preferences_pipeline(system, name)
                for system, name in zip(systems, names)
            ]
            if self.verbose:
                pipelines = tqdm(
                    pipelines,
                    desc="Computing system axiomatic preferences",
                    unit="system",
                )
            return concat([pipeline.transform(self.topics) for pipeline in pipelines])

        @cached_property
        def preference_distribution(self) -> DataFrame:
            """
            Return a dataframe with each axiom's preference distribution
            across all documents, queries and retrieval systems.

            The returned dataframe has the following columns:
            - axiom: Axiom name
            - axiom == 0: Absolute number of preferences equal to 0.
            - axiom == ORIG: Absolute number of preferences equal to
                the original ranking preference.
            - axiom != ORIG: Absolute number of preferences opposing
                the original ranking preference.
            """

            pref = self.preferences.copy()
            pref = pref[pref["ORIG_preference"] > 0]
            distributions = [
                {
                    "axiom": axiom_name,
                    "axiom == 0": len(pref[pref[f"{axiom_name}_preference"] == 0]),
                    "axiom == ORIG": len(pref[pref[f"{axiom_name}_preference"] > 0]),
                    "axiom != ORIG": len(pref[pref[f"{axiom_name}_preference"] < 0]),
                }
                for axiom_name in [
                    *self._additional_axiom_names,
                    *self._axiom_names,
                ]
            ]
            return DataFrame(distributions)

        def _oracle_consistency(
            self,
            axiom_name: str,
        ) -> float:
            preferences = self.preferences.copy()
            # Non-zero axiom preferences.
            preferences = preferences[preferences[f"{axiom_name}_preference"] > 0]
            if len(preferences) == 0:
                # Can't compute consistency if we compare against an empty set.
                return nan
            # Preferences are 'consistent' in the following cases:
            # > (same as axiom) and == (order doesn't matter)
            consistent = preferences[preferences["ORACLE_preference"] >= 0]
            return len(consistent) / len(preferences)

        def _orig_consistency(
            self,
            axiom_name: str,
            system: Optional[str] = None,
        ) -> float:
            preferences = self.preferences.copy()
            # Non-zero axiom preferences.
            preferences = preferences[preferences[f"{axiom_name}_preference"] > 0]
            # Only preferences of a specific system.
            if system is not None:
                preferences = preferences[preferences["name"] == system]
            # Preferences are 'consistent' in the following cases:
            # > (same as axiom) and == (order doesn't matter)
            consistent = preferences[preferences["ORIG_preference"] >= 0]
            if len(preferences) == 0:
                # Can't compute consistency if we compare against an empty set.
                return nan
            return len(consistent) / len(preferences)

        @cached_property
        def preference_consistency(self) -> DataFrame:
            """
            Return a dataframe with each axiom's consistency
            with the original ranking preferences and oracle preferences
            across all documents, queries and retrieval systems.

            The returned dataframe has the following columns:
            - axiom: Axiom name
            - ORIG_consistency: Relative consistency of non-zero preferences
                with ORIG preferences.
            - ORACLE_consistency: Relative consistency of non-zero preferences
                with ORACLE preferences.
            - <system>_consistency: Relative consistency of non-zero preferences
                with ORIG preferences from system <system> (if found in source)
            """

            if "name" in self.preferences.columns:
                systems = self.preferences["name"].unique().tolist()
            else:
                systems = []
            distributions = [
                {
                    **{
                        "axiom": axiom_name,
                        "ORIG_consistency": self._orig_consistency(axiom_name),
                        "ORACLE_consistency": self._oracle_consistency(axiom_name),
                    },
                    **{
                        f"{system}_consistency": self._orig_consistency(
                            axiom_name=axiom_name, system=system
                        )
                        for system in systems
                    },
                }
                for axiom_name in [
                    *self._additional_axiom_names,
                    *self._axiom_names,
                ]
            ]
            return DataFrame(distributions)

        @cached_property
        def inconsistent_pairs(self) -> DataFrame:
            pref = self.preferences.copy()
            # Preferences that are ranked wrong.
            ranked_wrong = (pref["ORACLE_preference"] > 0) & (
                pref["ORIG_preference"] < 0
            )
            pref = pref[ranked_wrong]
            # Preferences that have at least one correct axiom preference.
            axiom_hint = [
                pref[f"{axiom_name}_preference"] > 0
                for axiom_name in [
                    *self._additional_axiom_names,
                    *self._axiom_names,
                ]
            ]
            axiom_hinted = reduce(or_, axiom_hint)
            pref = pref[axiom_hinted]

            return pref

else:
    AxiomaticExperiment = NotImplemented
