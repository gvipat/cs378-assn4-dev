import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agentkv.trace import (
    HEADER,
    TraceBuilder,
    chat_segment,
    load,
    output_text,
)  # noqa: E402

LRU = str(
    REPO / "policies" / "reference.py"
)  # the Part 2 solution (a stub in the starter)
FIFO = str(REPO / "tests" / "policies" / "fifo.py")  # infrastructure tests use this one


def build_chat_trace(
    path,
    *,
    jobs=16,
    turns=3,
    num_blocks=120,
    block_size=16,
    seed=1,
    gap=0.5,
    think=(0.5, 3.0),
    fanout=0,
):
    """A small synthetic trace: chat sessions sharing a system prompt, with
    optional parallel fan-out calls per turn. Fast to simulate."""
    rng = random.Random(seed)
    b = TraceBuilder(
        Path(path).stem, "W1", "test", num_blocks=num_blocks, block_size=block_size
    )
    system = chat_segment(
        "system", "You are a helpful assistant for systems questions. " * 12
    )
    for j in range(jobs):
        job = b.job(arrival=j * gap)
        hist, prev = [system], None
        for t in range(turns):
            hist = hist + [
                chat_segment("user", f"job {j} turn {t} " * rng.randint(3, 15)),
                HEADER,
            ]
            out = output_text(f"answer {j}.{t} " * rng.randint(4, 30))
            deps = [prev] if prev else []
            n = job.llm("chat", hist, out, deps=deps)
            branch = [
                job.llm(
                    "side",
                    hist[:2] + [chat_segment("user", f"side {k}"), HEADER],
                    output_text(f"side {j} {k} " * rng.randint(2, 8)),
                    deps=deps,
                )
                for k in range(fanout)
            ]
            hist = hist + [out]
            prev = job.tool("think", rng.uniform(*think), deps=[n, *branch])
    b.save(path, require_jobs=jobs if jobs == 16 else None)
    return load(path)


@pytest.fixture
def chat_trace(tmp_path):
    return build_chat_trace(tmp_path / "chat.json")


@pytest.fixture
def pressure_trace(tmp_path):
    """Tight memory and fan-out: forces evictions and preemptions."""
    return build_chat_trace(
        tmp_path / "pressure.json", num_blocks=100, fanout=2, gap=0.1, think=(0.0, 0.3)
    )


def write_policy(tmp_path, source: str, name: str = "cand.py") -> str:
    p = tmp_path / name
    p.write_text(source)
    return str(p)
