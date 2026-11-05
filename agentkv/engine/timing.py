"""Step-time model for the CPU simulator.

    prefill step = p_base + p_token * new_tokens + p_attn * attention_pairs
    decode step  = d_base + d_seq * batch + d_ctx * sum(context lengths)

attention_pairs is the number of (query, key) pairs the step computes, which
captures how a cache hit shortens prefill: hit tokens are keys but not queries.

The defaults are first-principles estimates for Qwen3-8B (bf16, 36 layers,
32 query and 8 KV heads of dim 128, so 144 KiB of KV per token and 16.4 GB of
weights) on one NVIDIA A100 (about 1.55 TB/s and an effective 170 TFLOP/s).
A smaller model makes prefill too cheap for caching to matter (the staff
pilot measured prefill at 5-14% of busy time for Qwen3-0.6B on an L4). This
model only drives the teaching engine; scores come from Vidur's measured profile.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

SIM_VERSION = "1"
TIMING_PATH = Path(__file__).resolve().parent.parent / "data" / "timing.json"


@dataclass(frozen=True)
class TimingModel:
    p_base: float = 0.010  # s per prefill step (launch + weight read)
    p_token: float = 9.6e-5  # s per new token (dense layers: ~16 GFLOP/token)
    p_attn: float = 3.5e-9  # s per attention (q, k) pair over all layers (~0.6 MFLOP)
    d_base: float = 0.0106  # s per decode step (16.4 GB of weights read once per step)
    d_seq: float = 1.0e-4  # s per sequence in the batch
    d_ctx: float = 9.7e-8  # s per context token read (144 KiB at 1.55 TB/s)
    name: str = "qwen3-8b-a100-estimate"

    def prefill(self, chunks: list[tuple[int, int]]) -> float:
        """Estimate prefill time from (cached tokens, new tokens) pairs, one per sequence."""
        new = sum(n for _, n in chunks)
        pairs = sum(n * s + n * (n + 1) // 2 for s, n in chunks)
        return self.p_base + self.p_token * new + self.p_attn * pairs

    def decode(self, batch: int, context: int) -> float:
        return self.d_base + self.d_seq * batch + self.d_ctx * context

    def decode_run(self, batch: int, context: int, k: int) -> float:
        """Estimate the total time for k decode steps as each sequence's context grows."""
        return k * self.decode(batch, context) + self.d_ctx * batch * k * (k - 1) / 2

    def fingerprint(self) -> str:
        blob = json.dumps({"v": SIM_VERSION, **asdict(self)}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "TimingModel":
        path = Path(path) if path else TIMING_PATH
        if path.exists():
            return cls(**json.loads(path.read_text()))
        return cls()

    def save(self, path: str | Path | None = None) -> None:
        Path(path or TIMING_PATH).write_text(json.dumps(asdict(self), indent=2) + "\n")
