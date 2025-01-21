# ray job submit --runtime-env experiments/ray-env.yml -- python experiments/generate_llm_responses.py generate-llm-responses --model-name HuggingFaceTB/SmolLM-360M-Instruct

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Literal, Mapping, TYPE_CHECKING
from typing_extensions import TypeAlias  # type: ignore
from uuid import UUID, uuid5, NAMESPACE_URL

from cached_property import cached_property
from cyclopts import App
from cyclopts.types import ResolvedExistingFile, ResolvedDirectory
from dotenv import load_dotenv, find_dotenv
from pandas import DataFrame, Series, concat
from ray import init
from ray.data import read_json
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from torch import manual_seed
from torch.cuda import is_available as cuda_is_available

if TYPE_CHECKING:
    cached_property = property

Style: TypeAlias = Literal["bullet", "essay", "news"]

_NAMESPACE_UUID: UUID = uuid5(NAMESPACE_URL, "axioms")

_BASE_PROMPT: str = """You are an agent for Retrieval-Augmented Generation. 
You are given a search request ('topic') and a list of reference documents, just like a search engine would return them.
Compose a response text that answers the search request, explains the topic, and cites the respective references.
Cite claims you take from the references. If many references make very similar claims, cite them all. Not all references need be cited. 
Claims without reference can be added without citation. Quotations "..." are allowed.
Add the reference ID in brackets: [0]. Add citations at the end of a sentence before the full-stop. Combine multiple citations within the same brackets separated by commas.
Adopt a formal tone. Avoid first or second person ("I", "you").
The response should be short, around 250 words. Do not use markdown, only emit plain text.
"""

_STYLE_PROMPTS: Mapping[Style, str] = {
    "essay": _BASE_PROMPT
    + "The response should be written in the style of an essay: Start with a clear thesis, provide arguments, finish with a conclusion.",
    "bullet": _BASE_PROMPT
    + "The response should be written in the style of a bullet point list: List all the relevant points to the answer in a single-level bullet point list.",
    "news": _BASE_PROMPT
    + "The response should be written in the style of a news article following the inverted pyramid scheme: Start with the lead, then provide the important details, lastly add background information.",
}

_QUERY_PROMPT: str = "Query: {query}\n"
_DOCUMENT_PROMPT: str = "[{ref_number}] {ref_text}\n"
_REFERENCE_PROMPT: str = "References:\n"


def _system_prompt(style: Style) -> str:
    return _STYLE_PROMPTS[style] + "\n\n"


def _user_prompt(query: str, references: Sequence[str]) -> str:
    references = list(map(lambda ref: ref.replace("\n", " "), references))
    return (
        _QUERY_PROMPT.format(query=query)
        + "\n"
        + _REFERENCE_PROMPT
        + "".join(
            [
                _DOCUMENT_PROMPT.format(ref_number=i, ref_text=text)
                for i, text in enumerate(references)
            ]
        )
    )


@dataclass(frozen=True)
class _Generator:
    model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct"
    num_return_sequences: int = 5
    max_new_tokens: int = 1000
    references_top_k: int = 5

    @cached_property
    def _device(self) -> str:
        return "cuda" if cuda_is_available() else "cpu"

    @cached_property
    def _model(self) -> PreTrainedModel:
        if find_dotenv():
            load_dotenv()
        return AutoModelForCausalLM.from_pretrained(self.model_name).to(self._device)

    @cached_property
    def _tokenizer(self) -> PreTrainedTokenizer:
        if find_dotenv():
            load_dotenv()
        return AutoTokenizer.from_pretrained(self.model_name)

    def _generate(self, input: Series) -> DataFrame:
        # Construct the prompt.
        messages = [
            {
                "role": "system",
                "content": _system_prompt(input["style"]),
            },
            {
                "role": "user",
                "content": _user_prompt(
                    input["query"], input["references_texts"][: self.references_top_k]
                ),
            },
        ]

        # Tokenize the prompt.
        inputs = self._tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._model.device)
        input_length = inputs.shape[1]

        # Specify the generation configuration (i.e., use beam search).
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_return_sequences,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            num_return_sequences=self.num_return_sequences,
        )

        # Generate the responses.
        manual_seed(0)
        outputs = self._model.generate(
            inputs=inputs,
            generation_config=generation_config,
        )

        # Cut off the input tokens.
        outputs = outputs[:, input_length:]

        # Decode the responses.
        texts = self._tokenizer.batch_decode(
            sequences=outputs,
            skip_special_tokens=True,
        )

        # Cut off the text at the end-of-sequence token.
        texts = [text[: text.find(self._tokenizer.eos_token)] for text in texts]

        # Print the query and the first response for debugging.
        print(
            f"=============================\n\n"
            f"Query: {input['query']}\n"
            f"Response 0:\n{texts[0]}\n\n"
        )

        return DataFrame(
            [
                {
                    "topic": input["topic"],
                    "query": input["query"],
                    "references_ids": input["references_ids"][: self.references_top_k],
                    "references_texts": input["references_texts"][
                        : self.references_top_k
                    ],
                    "style": input["style"],
                    "model": self.model_name,
                    "response": str(
                        uuid5(
                            _NAMESPACE_UUID,
                            input["topic"] + input["style"] + self.model_name,
                        )
                    ),
                    "rank": rank,
                    "text": text,
                }
                for rank, text in enumerate(texts)
            ]
        )

    def generate(self, batch: DataFrame) -> DataFrame:
        return concat([self._generate(row) for _, row in batch.iterrows()])


cli = App()


@cli.command
def generate_llm_responses(
    model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct",
    num_return_sequences: int = 5,
    max_new_tokens: int = 1000,
    references_top_k: int = 5,
    input_path: ResolvedExistingFile = Path("/mnt/ceph/storage/data-in-progress/data-research/web-search/axioms/crowd/study1_retrieval.jsonl.gz"),
    output_path: ResolvedDirectory = Path(
        "/mnt/ceph/storage/data-in-progress/data-research/web-search/axioms/llm-responses"
    ),
) -> None:
    init()

    dataset = read_json(str(input_path))
    dataset = dataset.flat_map(
        lambda row: [
            {
                **row,
                "style": style,
            }
            for style in (
                "bullet",
                "essay",
                "news",
            )
        ],
        num_cpus=0.1,
        concurrency=4,
    )

    generator = _Generator(
        model_name=model_name,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        references_top_k=references_top_k,
    )

    dataset = dataset.map_batches(
        generator.generate,
        zero_copy_batch=True,
        batch_size=1,
        batch_format="pandas",
        num_cpus=1,
        num_gpus=0 if "1.7B" in model_name else 0.5,
        accelerator_type="A100-20GB" if "1.7B" in model_name else "GeForce-GTX-1080",
        # memory=20 * 1024 * 1024 * 1024,
        concurrency=10,
        max_retries=10,
    )

    dataset.write_json(str(output_path / model_name))


if __name__ == "__main__":
    cli()
