#!/usr/bin/env python3
from pathlib import Path
from typing import (
    Any, Dict, 
    # List,
    Optional, Sequence, Type
)

from autojudge_base import (
    Report,
    LeaderboardSpec,
    LeaderboardBuilder,
    LeaderboardVerification,
    MeasureSpec,
    AutoJudge,
    auto_judge_to_click_command,
    Leaderboard,
    Qrels,
    Request,
    # LlmConfigProtocol,
    NuggetBanks,
    NuggetBanksProtocol,
)
from collections import defaultdict
from time import perf_counter
from tqdm import tqdm
from ir_axioms.model import GenerationInput, GenerationOutput
from ir_axioms.tools import SpacyEntitiesAspectExtraction
spacy_ent = SpacyEntitiesAspectExtraction()


from ir_axioms.axiom import (  # noqa: E402
    Axiom,
    # Adapted axioms
    GEN_TFC1,
    GEN_LNC1,
    GEN_TF_LNC,
    GEN_AND,
    GEN_DIV,
    GEN_STMC1,
    GEN_STMC2,
    GEN_PROX1,
    GEN_PROX2,
    GEN_PROX3,
    GEN_PROX4,
    GEN_PROX5,
    GEN_aSL,
    # Generation-specific axioms
    # CLAR1,
    # CLAR2,
    CONS3,
    CONS2,
    CONS1,
    # CORR1,
    COV1,
    COV2,
    COV3,
    COH1,
    COH2,
)


axioms: dict[str, Axiom[GenerationInput, GenerationOutput]] = {
    #
    # Generative Axioms
    "GEN-TFC1": GEN_TFC1(),
    "GEN-LNC1": GEN_LNC1(),
    "GEN-AND": GEN_AND(),
    "GEN-DIV": GEN_DIV(),
    "GEN-STMC1": GEN_STMC1(),
    "GEN-STMC2": GEN_STMC2(),
    "GEN-PROX1": GEN_PROX1(),
    "GEN-PROX2": GEN_PROX2(),
    "GEN-PROX3": GEN_PROX3(),
    "GEN-PROX4": GEN_PROX4(),
    "GEN-PROX5": GEN_PROX5(),
    "GEN-aSL": GEN_aSL(),
    "GEN-TF-LNC": GEN_TF_LNC(),
    #
    # Coherence axioms
    "COH1-0.75": COH1(margin_fraction=0.75),
    "COH2-0.75": COH2(margin_fraction=0.75),
    #
    # Coverage axioms
    "COV1-SE-0.5": COV1(aspect_extraction=spacy_ent, margin_fraction=0.5),
    "COV2-SE-0.5": COV2(aspect_extraction=spacy_ent, margin_fraction=0.5),
    "COV3-SE-0.5": COV3(aspect_extraction=spacy_ent, margin_fraction=0.5),
    #
    # Consistency axioms
    "CONS1-SE-0.5": CONS1(aspect_extraction=spacy_ent, margin_fraction=0.5),
    "CONS2-0.5": CONS2(margin_fraction=0.5),
    "CONS3-0.5": CONS3(margin_fraction=0.5),
    #
    # Correctness axioms
    #"CORR1-0.75": CORR1(margin_fraction=0.75),
    #
    # Clarity axioms
    #"CLAR1-0.5": CLAR1(margin_fraction=0.5),
    #"CLAR2-0.5": CLAR2(margin_fraction=0.5),
}

def group_by_topic_id(rag_responses: Sequence[Report]) -> Dict[str, Dict[str, str]]:
    """Group RAG responses by topic_id, then by run_id."""
    ret: Dict[str, Dict[str, str]] = defaultdict(dict)
    for rag_response in rag_responses:
        run_id: str = rag_response.metadata.run_id
        topic_id: str = rag_response.metadata.topic_id
        ret[topic_id][run_id] = rag_response.get_report_text()
    return ret


LEADERBOARD_SPEC = LeaderboardSpec(measures=list(MeasureSpec(i) for i in axioms.keys()))

# Accumulates total wall-clock time (seconds) spent per axiom, across all topics.
axiom_timings: Dict[str, float] = defaultdict(float)


def _axiom_timings_lines() -> list[str]:
    lines = ["Axiom timing summary (total seconds, descending):"]
    for name, seconds in sorted(
        axiom_timings.items(), key=lambda kv: kv[1], reverse=True
    ):
        lines.append(f"  {name:20s} {seconds:10.3f}s")
    return lines


def print_axiom_timings() -> None:
    if not axiom_timings:
        return
    print("\n" + "\n".join(_axiom_timings_lines()))


def write_axiom_timings(out_dir: Optional[Path]) -> None:
    if not axiom_timings:
        return
    out_dir = Path(out_dir) if out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    timings_path = out_dir / "axiom_timings.txt"
    timings_path.write_text("\n".join(_axiom_timings_lines()) + "\n")
    print(f"\nAxiom timings written to {timings_path}")


class IrAxiomJudge(AutoJudge):
    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def create_nuggets(self, **kwargs: Any) -> Optional[NuggetBanksProtocol]:
        return None

    def create_qrels(self, **kwargs: Any) -> Optional[Qrels]:
        return None

    def judge_for_topic(self, inp: GenerationInput, outp: Sequence[GenerationOutput]):
        ret = {i.id: {} for i in outp}
        for axiom_name, axiom in axioms.items():
            start = perf_counter()
            preds = axiom.preferences(inp, outp)
            axiom_timings[axiom_name] += perf_counter() - start
            for v, run in zip(preds, outp):
                ret[run.id][axiom_name] = sum(v)/len(outp)
        return ret


    def judge(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        **kwargs: Any,
    ) -> Leaderboard:
        topic_id_to_title: Dict[str, str] = {
            i.request_id: i.title for i in rag_topics
        }
        topic_id_to_responses: Dict[str, Dict[str, str]] = group_by_topic_id(
            rag_responses
        )

        builder: LeaderboardBuilder = LeaderboardBuilder(LEADERBOARD_SPEC)
        

        for topic in tqdm(topic_id_to_responses.keys(), "Process Topics"):
            inp = GenerationInput(id=topic, text=topic_id_to_title[topic])
            outp = [GenerationOutput(id=k, text=v) for k, v in topic_id_to_responses[topic].items()]            
            system_to_axiom_to_score = self.judge_for_topic(inp, outp)


            for system, values in system_to_axiom_to_score.items():
                builder.add(
                    run_id=system,
                    topic_id=topic,
                    values=values,
                )

        leaderboard: Leaderboard = builder.build()
        LeaderboardVerification(leaderboard, on_missing="fix_aggregate", warn=True).all()
        print_axiom_timings()
        try:
            write_axiom_timings(Path(kwargs.get("filebase")).parent)
        except:
            pass
        return leaderboard


if __name__ == '__main__':
    auto_judge_to_click_command(IrAxiomJudge(), "axiom-judge")()
