"""Trace format `agentkv-trace/1`: loading, validation, block hashing, building.

A trace is 16 jobs with arrival times. Each job is a DAG of LLM calls (prompt
segments plus a frozen output) and fixed tool or think-time delays. See
docs/trace_format.md for the full contract.
"""

from __future__ import annotations

import hashlib
import json
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from . import tokenizer

SCHEMA = "agentkv-trace/1"
GRADED_JOBS = 16
HEADER = "<|im_start|>assistant\n"
END = "<|im_end|>\n"


# ---------------------------------------------------------------- rendering


def segment_id(text: str) -> str:
    return "s" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def chat_segment(role: str, content: str) -> str:
    """Format one chat message with the Qwen3 role and end-of-turn tokens."""
    if role == "tool":
        return f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response>{END}"
    return f"<|im_start|>{role}\n{content}{END}"


def output_text(content: str) -> str:
    """Add the end-of-turn token to the recorded assistant reply."""
    return content + END


# ---------------------------------------------------------------- hashing


def chain_hashes(tokens: array, block_size: int) -> list[int]:
    """Compute a chained hash for each full token block.

    Each hash includes the previous block's hash, so a block's identity
    depends on its own tokens and the entire preceding prefix.
    """
    out: list[int] = []
    prev = b""
    raw = memoryview(tokens).cast("B")
    width = tokens.itemsize * block_size
    for i in range(len(tokens) // block_size):
        h = hashlib.blake2b(prev, digest_size=16)
        h.update(raw[i * width : (i + 1) * width])
        prev = h.digest()
        out.append(int.from_bytes(prev, "little"))
    return out


# ---------------------------------------------------------------- model


@dataclass(frozen=True)
class Memory:
    num_blocks: int
    block_size: int = 256


@dataclass(frozen=True)
class EngineConfig:
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 8192


@dataclass
class LLMNode:
    id: str
    stage: str
    deps: tuple[str, ...]
    prompt: tuple[str, ...]
    output: str
    n_prompt: int = 0
    n_output: int = 0
    # Chained hashes of every full block of prompt + output, root first.
    hashes: tuple[int, ...] = ()
    type: str = "llm"

    @property
    def n_total(self) -> int:
        return self.n_prompt + self.n_output


@dataclass
class ToolNode:
    id: str
    stage: str
    deps: tuple[str, ...]
    delay: float
    type: str = "tool"


@dataclass
class Job:
    id: str
    arrival: float
    nodes: list
    meta: dict = field(default_factory=dict)


@dataclass
class Trace:
    trace_id: str
    family: str
    split: str
    description: str
    memory: Memory
    engine: EngineConfig
    jobs: list[Job]
    segments: dict[str, str]
    seg_tokens: dict[str, array]
    sha256: str
    path: str | None = None
    generation: dict = field(default_factory=dict)

    def tokens(self, node: LLMNode) -> array:
        """Return the token ids of a call's prompt followed by its recorded output."""
        out = array("I")
        for sid in node.prompt:
            out.extend(self.seg_tokens[sid])
        out.extend(self.seg_tokens[node.output])
        return out

    def llm_nodes(self) -> Iterable[tuple[Job, LLMNode]]:
        for job in self.jobs:
            for node in job.nodes:
                if node.type == "llm":
                    yield job, node

    def stats(self) -> dict:
        prompt = sum(n.n_prompt for _, n in self.llm_nodes())
        output = sum(n.n_output for _, n in self.llm_nodes())
        calls = sum(1 for _ in self.llm_nodes())
        unique = sum(len(t) for t in self.seg_tokens.values())
        return {
            "jobs": len(self.jobs),
            "llm_calls": calls,
            "tool_nodes": sum(
                1 for j in self.jobs for n in j.nodes if n.type == "tool"
            ),
            "prompt_tokens": prompt,
            "output_tokens": output,
            "unique_segment_tokens": unique,
            "num_blocks": self.memory.num_blocks,
            "memory_tokens": self.memory.num_blocks * self.memory.block_size,
        }


class TraceError(ValueError):
    pass


# ---------------------------------------------------------------- validation


def validate(
    d: dict,
    *,
    require_jobs: int | None = None,
    seg_counts: dict[str, int] | None = None,
) -> list[str]:
    """Check the trace structure and return a list of problems, empty if valid.

    When seg_counts is provided, also verify the stored token counts and
    check that each call fits within the allowed memory capacity.
    """
    errs: list[str] = []
    if d.get("schema") != SCHEMA:
        errs.append(f"schema must be {SCHEMA!r}")
    for key in ("trace_id", "family", "memory", "segments", "jobs"):
        if key not in d:
            errs.append(f"missing {key!r}")
    if errs:
        return errs
    mem = d["memory"]
    bs = mem.get("block_size", 256)
    nb = mem.get("num_blocks", 0)
    if not isinstance(nb, int) or nb < 2:
        errs.append("memory.num_blocks must be an int >= 2")
    if not isinstance(bs, int) or bs < 1:
        errs.append("memory.block_size must be a positive int")
    segs = d["segments"]
    for sid, seg in segs.items():
        if not isinstance(seg, dict) or not isinstance(seg.get("text"), str):
            errs.append(f"segment {sid}: needs text")
            continue
        if seg_counts is not None and seg.get("n_tokens") != seg_counts.get(sid):
            errs.append(
                f"segment {sid}: n_tokens {seg.get('n_tokens')} != tokenizer {seg_counts.get(sid)}"
            )
    jobs = d["jobs"]
    if require_jobs is not None and len(jobs) != require_jobs:
        errs.append(f"expected {require_jobs} jobs, found {len(jobs)}")
    job_ids = set()
    last_arrival = float("-inf")
    for job in jobs:
        jid = job.get("id")
        if jid in job_ids:
            errs.append(f"duplicate job id {jid}")
        job_ids.add(jid)
        arr = job.get("arrival")
        if not isinstance(arr, (int, float)) or arr < 0:
            errs.append(f"job {jid}: arrival must be a number >= 0")
        elif arr < last_arrival:
            errs.append(f"job {jid}: jobs must be sorted by arrival")
        else:
            last_arrival = arr
        nodes = job.get("nodes") or []
        if not nodes:
            errs.append(f"job {jid}: no nodes")
        ids = [n.get("id") for n in nodes]
        if len(set(ids)) != len(ids):
            errs.append(f"job {jid}: duplicate node ids")
        idset = set(ids)
        for n in nodes:
            where = f"job {jid} node {n.get('id')}"
            for dep in n.get("deps", []):
                if dep not in idset:
                    errs.append(f"{where}: unknown dep {dep}")
            t = n.get("type")
            if t == "llm":
                prompt = n.get("prompt") or []
                if not prompt:
                    errs.append(f"{where}: empty prompt")
                for sid in prompt + [n.get("output")]:
                    if sid not in segs:
                        errs.append(f"{where}: unknown segment {sid}")
                if seg_counts is not None and not any(
                    e.startswith(where) for e in errs
                ):
                    n_out = seg_counts.get(n["output"], 0)
                    total = sum(seg_counts.get(s, 0) for s in prompt) + n_out
                    if n_out < 1:
                        errs.append(f"{where}: output must be at least one token")
                    if isinstance(nb, int) and -(-total // bs) > nb // 2:
                        errs.append(
                            f"{where}: {total} tokens do not fit in half of memory"
                        )
            elif t == "tool":
                delay = n.get("delay")
                if not isinstance(delay, (int, float)) or delay < 0:
                    errs.append(f"{where}: delay must be a number >= 0")
            else:
                errs.append(f"{where}: type must be llm or tool")
        if _has_cycle(nodes):
            errs.append(f"job {jid}: dependency cycle")
    return errs


def _has_cycle(nodes: list[dict]) -> bool:
    deps = {n.get("id"): list(n.get("deps", [])) for n in nodes}
    state: dict = {}

    def visit(u) -> bool:
        if state.get(u) == 1:
            return True
        if state.get(u) == 2:
            return False
        state[u] = 1
        for v in deps.get(u, []):
            if v in deps and visit(v):
                return True
        state[u] = 2
        return False

    return any(visit(u) for u in deps)


# ---------------------------------------------------------------- loading


def load(path: str | Path, *, require_jobs: int | None = None) -> Trace:
    raw = Path(path).read_bytes()
    t = from_dict(
        json.loads(raw),
        require_jobs=require_jobs,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    t.path = str(path)
    return t


def from_dict(
    d: dict, *, require_jobs: int | None = None, sha256: str | None = None
) -> Trace:
    errs = validate(d, require_jobs=require_jobs)
    if errs:
        raise TraceError("; ".join(errs[:10]))
    sids = list(d["segments"])
    toks = tokenizer.encode_batch([d["segments"][s]["text"] for s in sids])
    seg_tokens = {sid: array("I", ids) for sid, ids in zip(sids, toks)}
    counts = {sid: len(a) for sid, a in seg_tokens.items()}
    errs = validate(d, require_jobs=require_jobs, seg_counts=counts)
    if errs:
        raise TraceError("; ".join(errs[:10]))
    mem = Memory(**d["memory"])
    eng = EngineConfig(**d.get("engine", {}))
    jobs = []
    for jd in d["jobs"]:
        nodes = []
        for nd in jd["nodes"]:
            deps = tuple(nd.get("deps", []))
            if nd["type"] == "llm":
                node = LLMNode(
                    id=nd["id"],
                    stage=nd.get("stage", ""),
                    deps=deps,
                    prompt=tuple(nd["prompt"]),
                    output=nd["output"],
                )
                node.n_prompt = sum(counts[s] for s in node.prompt)
                node.n_output = counts[node.output]
                nodes.append(node)
            else:
                nodes.append(
                    ToolNode(
                        id=nd["id"],
                        stage=nd.get("stage", ""),
                        deps=deps,
                        delay=float(nd["delay"]),
                    )
                )
        jobs.append(
            Job(
                id=jd["id"],
                arrival=float(jd["arrival"]),
                nodes=nodes,
                meta=jd.get("meta", {}),
            )
        )
    if sha256 is None:
        sha256 = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    trace = Trace(
        trace_id=d["trace_id"],
        family=d["family"],
        split=d.get("split", ""),
        description=d.get("description", ""),
        memory=mem,
        engine=eng,
        jobs=jobs,
        segments={s: d["segments"][s]["text"] for s in sids},
        seg_tokens=seg_tokens,
        sha256=sha256,
        generation=d.get("generation", {}),
    )
    for _, node in trace.llm_nodes():
        node.hashes = tuple(chain_hashes(trace.tokens(node), mem.block_size))
    return trace


# ---------------------------------------------------------------- building


class TraceBuilder:
    """Helper for generators. Segments are deduplicated by content hash.

    b = TraceBuilder("dev-W1-0", "W1", "public_dev", num_blocks=192)
    job = b.job(arrival=0.0)
    n0 = job.llm("chat", prompt=[system_text, user_text, HEADER], output=output_text("hi"))
    job.tool("think", 4.2, deps=[n0])
    b.save("data/traces/public_dev/dev-W1-0.json")
    """

    def __init__(
        self,
        trace_id: str,
        family: str,
        split: str,
        *,
        num_blocks: int,
        block_size: int = 256,
        description: str = "",
        engine: dict | None = None,
        generation: dict | None = None,
    ):
        self.d = {
            "schema": SCHEMA,
            "trace_id": trace_id,
            "family": family,
            "split": split,
            "description": description,
            "tokenizer": tokenizer.TOKENIZER_NAME,
            "memory": {"num_blocks": num_blocks, "block_size": block_size},
            "engine": engine or {"max_num_seqs": 16, "max_num_batched_tokens": 8192},
            "generation": generation or {},
            "segments": {},
            "jobs": [],
        }

    def seg(self, text: str) -> str:
        sid = segment_id(text)
        if sid not in self.d["segments"]:
            self.d["segments"][sid] = {"text": text, "n_tokens": None}
        return sid

    def job(
        self, arrival: float, job_id: str | None = None, meta: dict | None = None
    ) -> "JobBuilder":
        jid = job_id or f"J{len(self.d['jobs']):02d}"
        jd = {
            "id": jid,
            "arrival": round(float(arrival), 6),
            "meta": meta or {},
            "nodes": [],
        }
        self.d["jobs"].append(jd)
        return JobBuilder(self, jd)

    def to_dict(self) -> dict:
        self.d["jobs"].sort(key=lambda j: j["arrival"])
        segs = self.d["segments"]
        sids = list(segs)
        for sid, ids in zip(
            sids, tokenizer.encode_batch([segs[s]["text"] for s in sids])
        ):
            segs[sid]["n_tokens"] = len(ids)
        return self.d

    def save(self, path: str | Path, *, require_jobs: int | None = None) -> dict:
        d = self.to_dict()
        counts = {s: v["n_tokens"] for s, v in d["segments"].items()}
        errs = validate(d, require_jobs=require_jobs, seg_counts=counts)
        if errs:
            raise TraceError("; ".join(errs[:10]))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
        return d


class JobBuilder:
    def __init__(self, b: TraceBuilder, jd: dict):
        self.b, self.jd = b, jd

    @property
    def id(self) -> str:
        return self.jd["id"]

    def _nid(self) -> str:
        return f"n{len(self.jd['nodes'])}"

    def llm(
        self, stage: str, prompt: list[str], output: str, deps: Iterable[str] = ()
    ) -> str:
        """`prompt` is a list of segment texts; `output` is the frozen output text
        (use trace.output_text). Returns the node id."""
        nid = self._nid()
        self.jd["nodes"].append(
            {
                "id": nid,
                "type": "llm",
                "stage": stage,
                "deps": list(deps),
                "prompt": [self.b.seg(t) for t in prompt],
                "output": self.b.seg(output),
            }
        )
        return nid

    def tool(self, stage: str, delay: float, deps: Iterable[str] = ()) -> str:
        nid = self._nid()
        self.jd["nodes"].append(
            {
                "id": nid,
                "type": "tool",
                "stage": stage,
                "deps": list(deps),
                "delay": round(float(delay), 6),
            }
        )
        return nid
