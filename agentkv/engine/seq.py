"""A sequence in the engine. Field names match nano-vLLM's `Sequence` so the
GPU adapter can hand these objects straight to its ModelRunner."""

from __future__ import annotations

from array import array
from enum import Enum, auto
from itertools import count


class SeqStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Seq:
    block_size = 256
    _ids = count()

    def __init__(
        self,
        num_prompt_tokens: int,
        max_tokens: int,
        hashes: tuple[int, ...],
        tokens: array | None = None,
        owner=None,
    ):
        assert num_prompt_tokens >= 1 and max_tokens >= 1
        self.seq_id = next(Seq._ids)
        self.status = SeqStatus.WAITING
        self.num_prompt_tokens = num_prompt_tokens
        self.num_tokens = num_prompt_tokens
        self.max_tokens = max_tokens
        # Chained hashes of every full block of prompt + frozen output.
        self.hashes = hashes
        # Prompt + frozen output token ids (GPU replay and verified lookups only).
        self.tokens = tokens
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.block_table: list[int] = []
        self.is_prefill = True
        self.temperature = 1.0
        self.ignore_eos = True
        self.owner = owner
        self.admissions = 0
        self.preemptions = 0
        self.first_cached_tokens: int | None = None

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, key):
        assert self.tokens is not None
        if isinstance(key, slice):
            start, stop, step = key.indices(self.num_tokens)
            return self.tokens[start:stop:step].tolist()
        if key < 0:
            key += self.num_tokens
        assert 0 <= key < self.num_tokens
        return self.tokens[key]

    @property
    def last_token(self) -> int:
        assert self.tokens is not None
        return self.tokens[self.num_tokens - 1]

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def is_finished(self) -> bool:
        return self.status == SeqStatus.FINISHED

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i: int) -> tuple[int, ...] | None:
        if self.tokens is None:
            return None
        bs = self.block_size
        return tuple(self.tokens[i * bs : min((i + 1) * bs, self.num_tokens)])

    def append_token(self) -> None:
        """Advance by one generated token. Outputs are frozen, so the token id is
        already in `tokens`; the GPU adapter forces the sampler to produce it."""
        self.num_tokens += 1
