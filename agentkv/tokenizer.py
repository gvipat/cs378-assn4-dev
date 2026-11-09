"""Tokenize trace segments with the pinned Qwen3 tokenizer.

Each segment is tokenized separately, then concatenated to form the prompt.
Vidur uses these token counts and prefix hashes to simulate serving.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

TOKENIZER_NAME = "qwen3-0.6b@c1899de2"
TOKENIZER_PATH = Path(__file__).parent / "data" / "qwen3_tokenizer.json"


@lru_cache(maxsize=1)
def _tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(TOKENIZER_PATH))


def single_threaded() -> None:
    """Use one tokenizer thread per process.

    Pass this function as a process pool initializer to prevent each worker
    from starting its own pool of tokenizer threads.
    """
    os.environ["RAYON_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def encode(text: str) -> list[int]:
    return _tokenizer().encode(text, add_special_tokens=False).ids


def encode_batch(texts: list[str]) -> list[list[int]]:
    if not texts:
        return []
    return [e.ids for e in _tokenizer().encode_batch(texts, add_special_tokens=False)]


def count(text: str) -> int:
    return len(encode(text))


def decode(ids: list[int]) -> str:
    return _tokenizer().decode(ids, skip_special_tokens=False)
