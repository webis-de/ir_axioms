from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Final, Literal

from pandas import DataFrame, read_json, concat, Series
from typing_extensions import TypeAlias  # type: ignore

from axioms.axiom.base import Axiom
from axioms.model import GenerationInput, GenerationOutput
from axioms.axiom.utils import strictly_greater

TrecRagNuggetScoreType: TypeAlias = Literal[
    "all",
    "vital",
    "weighted",
]


@dataclass(frozen=True, kw_only=True)
class TrecRagNuggetAxiom(Axiom[GenerationInput, GenerationOutput]):
    assignments_path: Path = Path("data/nugget_assignment.20241108.jl")
    score_type: TrecRagNuggetScoreType = "all"
    strict: bool = False

    @cached_property
    def _assignments(self) -> DataFrame:
        df = read_json(
            self.assignments_path,
            lines=True,
        )
        df["query_hash"] = df.pop("query").map(hash)
        df["answer_text_hash"] = df.pop("answer_text").map(hash)
        df = df.drop(
            columns=[
                "qid",
                "run_id",
                "response_length",
            ]
        )
        df = df.explode(column="nuggets")
        df = concat(
            [df, df.pop("nuggets").apply(Series)],
            axis=1,
        )
        df = df.drop(
            columns=[
                "text",
            ]
        )
        df["assignment_score"] = (
            df["assignment"]
            .map(
                {
                    "support": 1.0,
                    "partial_support": 0.0 if self.strict else 0.5,
                }
            )
            .fillna(0.0)
        )
        return df

    def preference(
        self,
        input: GenerationInput,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> float:
        df = self._assignments
        df = df[df["query_hash"] == hash(input.text)]
        df1 = df[df["answer_text_hash"] == hash(output1.text)]
        df2 = df[df["answer_text_hash"] == hash(output2.text)]
        if len(df1) == 0 or len(df2) == 0:
            return 0

        if self.score_type == "all":
            score1 = df1["assignment_score"].sum() / len(df1)
            score2 = df2["assignment_score"].sum() / len(df2)
        elif self.score_type == "vital":
            df1 = df1[df1["importance"] == "vital"]
            df2 = df2[df2["importance"] == "vital"]
            score1 = df1["assignment_score"].sum() / len(df1)
            score2 = df2["assignment_score"].sum() / len(df2)
        elif self.score_type == "weighted":
            df1_vital = df1[df1["importance"] == "vital"]
            df2_vital = df2[df2["importance"] == "vital"]
            df1_okay = df1[df1["importance"] != "vital"]
            df2_okay = df2[df2["importance"] != "vital"]
            score1 = (
                df1_vital["assignment_score"].sum()
                + 0.5 * df1_okay["assignment_score"].sum()
            ) / (len(df1_vital) + 0.5 * len(df1_okay))
            score2 = (
                df2_vital["assignment_score"].sum()
                + 0.5 * df2_okay["assignment_score"].sum()
            ) / (len(df2_vital) + 0.5 * len(df2_okay))

        return strictly_greater(score1, score2)


TREC_RAG_NUGGET_ALL: Final = TrecRagNuggetAxiom()


TrecRagNuggetAxiom(
    score_type="weighted",
    strict=True,
).preference(
    input=GenerationInput(text="what have teen done to stop animal testing"),
    output1=GenerationOutput(
        text="Once a company starts selling in China, all leverage to end animal testing in the country is lost. PETA and the Humane Society, which have provided financial support to IIVS through sponsored trainings and research, have not profited from their efforts, demonstrating that there are alternative ways to combat animal testing in the region. The most effective thing for a company to do is not sell products that have been tested on animals, says Guillermo. Once a company starts selling in China, as Guillermo notes, all leverage to end animal testing in the country is lost. Some major brands, such as The Body Shop and Kat Von D, that have chosen not to sell in China have all leverage to end animal testing in the country. PETA and the Humane Society, who have also chosen not to sell in China, have provided financial support to IIVS, but neither made a profit from it, showing there are other ways to put a stop to animal testing in the region. Note: These organizations, PETA and the Humane Society, have demonstrated that it is possible to address animal testing in the region without seeking financial gain, specifically in the context of selling in China. Congress is revising the Toxic Substances Control Act, which regulates industrial chemicals, to better align with international standards and ensure that companies comply with animal testing regulations when selling in China. Sign the Petition to Tell the Labs Behind the Industrial Chemicals Regulated by Congress to Stop Testing on Animals! Donate to nonprofits like Leaping Bunny, PETA, and Physicians Committee for Responsible Medicine, which work tirelessly to end animal testing, and consider supporting their efforts by making a donation. Note: These organizations, such as Leaping Bunny, PETA, and Physicians Committee for Responsible Medicine, provide information on how to take action to stop animal testing, such as signing petitions and donating to organizations that work to end animal testing. 6. You can also take action by buying cruelty-free products and stopping the use of products that have been tested on animals. 7. By supporting organizations like People for Ethical Treatment of Animals or PETA, you can help prevent animal testing. By choosing to use only cruelty-free products, individuals can help organizations like People for Ethical Treatment of Animals or PETA in their efforts to prevent animal testing, ultimately saving millions of innocent animals."
    ),
    output2=GenerationOutput(
        text="Teen activists have taken several actions to stop animal testing including boycotting companies that conduct animal testing and encouraging friends and family to do the same Boycott them instead and ask friends and family to boycott them as well. They have shared information on social media about the negative impacts of animal testing and have distributed outreach and educational materials to raise awareness about the issue Handout Brochures Outreach and Education Materials. Some teens have adopted a vegan lifestyle as a way to help reduce the need for testing on animals by avoiding products that result in animal testing such as animal-derived foods linked to diseases that are commonly tested on animals Go Vegan!.\n- Teen activists have also acted as whistleblowers by reporting concerns about animal testing to organizations like PETA documenting evidence with photographs or videos and providing details to organizations to investigate . Be A Whistleblower - By supporting organizations like PETA teenagers contribute to campaigns that aim to expose and end the use of animals in experiments fund humane non-animal research and advocate for alternatives to animal testing .With the help of our members and supporters PETA campaigns globally to expose and end the use of animals in experiments. Teen activists have encouraged others to join them in purchasing cruelty-free products sharing information on companies that do not test on animals and urging businesses to use only cruelty-free products thereby promoting ethical consumer choices 49:885383423#What You Can Do; #Never Abandon Your Pet to an Animal Shelter & Rescue From an Animal Shelter.\n- Additionally some teenagers have taken steps to educate themselves and others on the alternatives to animal testing such as artificial skin models human cell cultures and computer simulations to promote more ethical and effective testing methods. Here are some ideas to get you started; In this article we are going to look at ways that can be used to stop animal testing; #Thus  because animals are subjected to agonizing pain suffering and death when they are used in laboratory and cosmetics testing animal research must be stopped to prevent more waste of animal life. Please check the provided context documents for more detailed information on the actions teens have taken to stop animal testing."
    ),
)
