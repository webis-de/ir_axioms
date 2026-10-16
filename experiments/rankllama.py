"""
PyTerrier wrapper for the RankLlama re-ranker (e.g., castorini/rankllama-v1-7b-lora-passage).
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Hashable, Mapping

from pandas import DataFrame, concat
from peft import PeftConfig, PeftModel
from pyterrier import Transformer
from pyterrier.model import add_ranks
from torch import no_grad
from torch.cuda import is_available as cuda_is_available
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


@dataclass(frozen=True, kw_only=True)
class RankLlamaReranker(Transformer):
    """
    Reranker PyTerrier basé sur RankLlama.
    A appliquer sur le top-k d'un premier retrieval (ex. `bm25 % 20 >> RankLlamaReranker()`).
    """

    base_model_name: str = "meta-llama/Llama-2-7b-hf"
    peft_model_name: str = "castorini/rankllama-v1-7b-lora-passage"
    text_field: str = "text"
    batch_size: int = 8
    max_length: int = 256
    verbose: bool = False

    @cached_property
    def _device(self) -> str:
        return "cuda" if cuda_is_available() else "cpu"

    @cached_property
    def _tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        tokenizer.pad_token = (
            tokenizer.eos_token
        )  # FIXME: Temporary workaround for missing pad_token in LLaMA tokenizer.
        return tokenizer

    @cached_property
    def _model(self) -> PeftModel:
        config = PeftConfig.from_pretrained(self.peft_model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path, num_labels=1
        )
        model = PeftModel.from_pretrained(base_model, self.peft_model_name)
        model = model.merge_and_unload()  # type: ignore
        model.eval()
        return model

    def _score_batch(self, queries: list[str], texts: list[str]) -> list[float]:
        inputs = self._tokenizer(
            [f"query: {q}" for q in queries],
            [f"document: {t}" for t in texts],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self._device)
        with no_grad():
            logits = self._model(**inputs).logits
        return logits[:, 0].detach().cpu().tolist()

    def _transform_group(
        self, group_keys: Mapping[Hashable, Any], res: DataFrame
    ) -> DataFrame:
        query = group_keys["query"]
        scores: list[float] = []
        rows = res.to_dict("records")
        for start in range(0, len(rows), self.batch_size):
            batch = rows[start : start + self.batch_size]
            scores.extend(
                self._score_batch(
                    queries=[query] * len(batch),
                    texts=[row[self.text_field] for row in batch],
                )
            )
        res = res.copy()
        res["score"] = scores
        return add_ranks(res, single_query=True)

    def transform(self, inp: DataFrame) -> DataFrame:
        query_rankings = inp.groupby(by=["qid", "query"], group_keys=True, sort=False)
        if len(query_rankings) == 0:
            return inp
        groups = tqdm(
            query_rankings,
            desc="RankLlama re-rank",
            unit="query",
            disable=not self.verbose,
        )
        return concat(
            [
                self._transform_group(
                    group_keys={"qid": qid, "query": query}, res=ranking
                )
                for (qid, query), ranking in groups
            ]
        )
