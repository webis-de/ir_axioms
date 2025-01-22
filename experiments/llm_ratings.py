from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Literal, Mapping, TYPE_CHECKING
from typing_extensions import TypeAlias  # type: ignore
from uuid import UUID, uuid5, NAMESPACE_URL

from cached_property import cached_property
from cappr import Example
from cappr.huggingface.classify import predict_proba_examples
from cyclopts import App
from cyclopts.types import ResolvedExistingFile, ResolvedDirectory
from dotenv import load_dotenv, find_dotenv
from pandas import DataFrame
from ray import init
from ray.data import read_json, Dataset, DataContext
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
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
class _Rater:
    model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct"
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

    def rate(self, batch: DataFrame) -> DataFrame:
        # Construct the chat prompts.
        messages = [
            [
                {
                    "role": "system",
                    "content": _system_prompt(
                        style=row["style"],
                    ),
                },
                {
                    "role": "user",
                    "content": _user_prompt(
                        query=row["query"],
                        references=row["references_texts"][: self.references_top_k],
                    ),
                },
            ]
            for _, row in batch.iterrows()
        ]

        # Template the chat prompts.
        inputs: list[str] = self._tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Prepare input-output examples.
        examples = [
            Example(
                prompt=input,
                completions=[row["raw_text"]],
                end_of_prompt="",
                normalize=False,
            )
            for input, (_, row) in zip(inputs, batch.iterrows())
        ]

        # Predict the completion probabilities.
        probabilities = predict_proba_examples(
            examples=examples,
            model_and_tokenizer=(self._model, self._tokenizer),
            show_progress_bar=False,
            batch_size=2,
            batch_size_completions=2,
        )[:, 0]

        # Construct the output DataFrame.
        batch["probability"] = probabilities

        return batch[
            [
                "topic",
                "query",
                "response",
                "raw_text",
                "probability",
            ]
        ]


cli = App()


@cli.command
def rate_llm_responses(
    model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct",
    references_top_k: int = 5,
    input_path: ResolvedExistingFile = Path(
        "/mnt/ceph/storage/data-in-progress/data-research/web-search/axioms/crowd/responses.jsonl.gz"
    ),
    output_path: ResolvedDirectory = Path(
        "/mnt/ceph/storage/data-in-progress/data-research/web-search/axioms/llm-ratings"
    ),
) -> None:
    init()

    data_context = DataContext.get_current()
    data_context.enable_tensor_extension_casting = False

    dataset: Dataset = read_json(
        paths=str(input_path),
        concurrency=10,
    )
    dataset = dataset.select_columns(
        cols=["topic", "query", "response", "style", "references_texts", "raw_text"],
        concurrency=10,
    )

    rater = _Rater(
        model_name=model_name,
        references_top_k=references_top_k,
    )

    is_large = (
        "1.7B" in model_name
        or "1B" in model_name
        or "Phi-3-mini" in model_name
        or "B-Instruct" in model_name.casefold()
    )

    dataset = dataset.map_batches(
        rater.rate,
        zero_copy_batch=True,
        batch_size=32,
        batch_format="pandas",
        num_gpus=1 if is_large else 0.5,
        accelerator_type="A100-10GB" if False else "GeForce-GTX-1080",
        # memory=20 * 1024 * 1024 * 1024,
        concurrency=10,
        max_retries=10,
    )

    dataset.write_json(str(output_path / model_name))


if __name__ == "__main__":
    cli()
