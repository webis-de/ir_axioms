from dataclasses import dataclass
from enum import Enum
from hashlib import md5
from json import loads
from math import nan
from pathlib import Path
from typing import Dict, List
from typing import Optional, Set

from requests import Response, post

DEFAULT_TARGER_API_URL = "https://demo.webis.de/targer-api/apidocs/"
DEFAULT_TARGER_MODELS = {"tag-ibm-fasttext"}


class TargerArgumentLabel(Enum):
    C_B = "C-B"
    C_I = "C-I"
    MC_B = "MC-B"
    MC_I = "MC-I"
    P_B = "P-B"
    P_I = "P-I"
    MP_B = "MP-B"
    MP_I = "MP-I"
    O = "O"  # noqa: E741
    B_B = "B-B"
    B_I = "B-I"
    X = "-X-"

    @classmethod
    def from_json(cls, json):
        return cls(str(json))


@dataclass
class TargerArgumentTag:
    label: TargerArgumentLabel
    probability: float
    token: str

    @classmethod
    def from_json(cls, json):
        return cls(
            TargerArgumentLabel.from_json(json["label"]),
            float(json["prob"]) if "prob" in json else nan,
            str(json["token"])
        )


class TargerArgumentSentence(List[TargerArgumentTag]):
    @classmethod
    def from_json(cls, json):
        return cls(
            TargerArgumentTag.from_json(tag)
            for tag in json
        )


class TargerArgumentSentences(List[TargerArgumentSentence]):
    @classmethod
    def from_json(cls, json) -> "TargerArgumentSentences":
        return cls(
            TargerArgumentSentence.from_json(sentence)
            for sentence in json
        )


def fetch_arguments(
        text: str,
        models: Set[str] = None,
        api_url: str = DEFAULT_TARGER_API_URL,
        cache_dir: Optional[Path] = None,
) -> Dict[str, TargerArgumentSentences]:
    if models is None:
        models = DEFAULT_TARGER_MODELS

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    arguments: Dict[str, TargerArgumentSentences] = {
        model: _fetch_sentences(api_url, model, text, cache_dir)
        for model in models
    }
    return arguments


def _fetch_sentences(
        text: str,
        model: str,
        api_url: str = DEFAULT_TARGER_API_URL,
        cache_dir: Optional[Path] = None,
) -> TargerArgumentSentences:
    content_hash: str = md5(text.encode()).hexdigest()
    cache_file = cache_dir / model / f"{content_hash}.json" \
        if cache_dir is not None \
        else None

    # Check if the API response is found in the cache.
    if cache_file is not None and cache_file.exists() and cache_file.is_file():
        with cache_file.open("r") as file:
            json = loads(file.read())
            return TargerArgumentSentences.from_json(json)

    headers = {
        "Accept": "application/json",
        "Content-Type": "text/plain",
    }
    res: Response = post(
        api_url + model,
        headers=headers,
        data=text.encode("utf-8")
    )
    json = res.json()

    # Cache the API response.
    if cache_file is not None:
        cache_file.parent.mkdir(exist_ok=True)
        with cache_file.open("wb") as file:
            file.write(res.content)

    return TargerArgumentSentences.from_json(json)
