"""Workload analysis of a recorded Part 1 run (task 1.3).

    from research_app.analysis import load_run
    run = load_run("runs/c2")
    critical_path(run, "R03")              # the chain of operations that set R03's finish time
    prefix_hit_rate(run.llm_calls())   # fraction of prompt tokens in previously seen blocks
    reuse_intervals(run.llm_calls())            # how long each reused block had to stay cached
    kv_footprint(run.llm_calls())          # KV bytes held by running calls over time

Times are seconds since the run started, as recorded in trace.jsonl.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

BLOCK_TOKENS = 256  # block_hashes cover full 256-token blocks
KV_BYTES_PER_TOKEN = (
    128 * 1024
)  # Llama-3-8B, bf16: 32 layers x 8 KV heads x 128 dims x 2 (K, V) x 2 bytes


@dataclass
class Op:
    op_id: str
    report: str
    kind: str  # llm | search | fetch | validate
    stage: str
    deps: list[str]
    start: float  # the operation was issued (its dependencies had finished)
    end: float
    status: str  # ok | failed | cancelled
    attempts: int = 1
    slot_requested: float | None = None  # LLM calls: asked for a slot under the cap
    slot_acquired: float | None = None  # LLM calls: got the slot; service starts here
    prompt_tokens: int = 0
    output_tokens: int = 0
    block_hashes: list[str] = field(default_factory=list)

    @property
    def slot_wait(self) -> float:
        if self.slot_requested is None or self.slot_acquired is None:
            return 0.0
        return self.slot_acquired - self.slot_requested


@dataclass
class Run:
    cap: int
    backend: str
    arrivals: dict[str, float]  # report id -> start time of the report
    ops: dict[str, Op]

    def report_ops(self, report: str) -> list[Op]:
        return sorted(
            (o for o in self.ops.values() if o.report == report),
            key=lambda o: (o.start, o.op_id),
        )

    def llm_calls(self) -> list[Op]:
        """Completed LLM calls of every report, in start order."""
        return sorted(
            (o for o in self.ops.values() if o.kind == "llm" and o.status == "ok"),
            key=lambda o: (o.start, o.op_id),
        )


def load_run(path) -> Run:
    """Read runs/<name>/trace.jsonl (or the file itself) into finished operations."""
    path = Path(path)
    if path.is_dir():
        path = path / "trace.jsonl"
    cap, backend, arrivals, started, ops = 0, "", {}, {}, {}
    for line in path.read_text().splitlines():
        ev = json.loads(line)
        kind = ev["event"]
        if kind == "run_start":
            cap, backend = ev.get("cap", 0), ev.get("backend", "")
        elif kind == "report_start":
            arrivals[ev["report_id"]] = ev["t"]
        elif kind == "op_start":
            started[ev["op_id"]] = ev
        elif kind == "op_end":
            s = started[ev["op_id"]]
            ops[ev["op_id"]] = Op(
                ev["op_id"],
                s["report_id"],
                s["kind"],
                s["stage"],
                list(s["deps"]),
                s["t"],
                ev["t"],
                ev["status"],
                ev.get("attempts", 1),
                ev.get("slot_requested"),
                ev.get("slot_acquired"),
                ev.get("prompt_tokens", 0),
                ev.get("output_tokens", 0),
                list(ev.get("block_hashes") or []),
            )
    return Run(cap, backend, arrivals, ops)


@dataclass
class Step:
    """One operation on a critical path. gap + wait + service spans from the
    previous step's end (or the report's start) to this operation's end."""

    op_id: str
    kind: str
    stage: str
    gap: float  # from the blocking dependency's end (or the report start) to this operation's start
    wait: float  # waiting for an LLM slot under the cap (0 for tools)
    service: float  # running, including retries and their backoff


def critical_path(run: Run, report: str) -> list[Step]:
    """Return the operations that determined when a report finished, first to last."""
    # [Part 1] TODO: Walk back from the report's last-finishing operation.
    # Start from the report's operation with the latest end (break ties by
    # op_id). Its blocking dependency is the dependency that finished last;
    # repeat until an operation has no dependencies in the run. Return the
    # steps in time order. For each step, gap is the time from the blocking
    # dependency's end (or the report's arrival for the first step) to the
    # operation's start; wait is its slot wait; service is the rest of its
    # duration. The steps' gap + wait + service must add up to the report's
    # end minus its arrival.
    raise NotImplementedError("Part 1: fill critical_path")
    # END OF YOUR CODE


def prefix_hit_rate(calls: list[Op], block_size: int = BLOCK_TOKENS) -> float:
    """Return the fraction of prompt tokens that are in prefix blocks seen by earlier calls."""
    # [Part 1] TODO: calls are in start order. A full block counts as shared
    # only if its block hash appeared in an earlier call; add this call's hashes to
    # the history after counting. Each shared block contributes block_size
    # tokens. Divide by the total prompt_tokens (which include partial final
    # blocks). Return 0.0 when there are no prompt tokens.
    raise NotImplementedError("Part 1: fill prefix_hit_rate")
    # END OF YOUR CODE


@dataclass
class Reuse:
    block: str
    stage: str  # stage of the call that reused the block
    source: str  # stage of the earlier call that held it most recently
    interval: float  # seconds the block sat unused before this reuse (0 if a call still held it)


def reuse_intervals(calls: list[Op]) -> list[Reuse]:
    """For every reuse of a prefix block, how long it had to stay cached."""
    # [Part 1] TODO: Process calls in start order. A block is held from a
    # call's start to its end. When a call's block hash appeared in an earlier
    # call, record one Reuse: interval is this call's start minus the latest end of
    # the earlier calls that held the block (0 if one of them is still running),
    # and source is the stage of that latest-ending call. Count a block once
    # per call.
    raise NotImplementedError("Part 1: fill reuse_intervals")
    # END OF YOUR CODE


@dataclass
class Footprint:
    t: float
    calls: int  # LLM calls holding a slot
    private: float  # bytes if every call stores its own prompt and output
    shared: float  # bytes if calls on one GPU share identical full prefix blocks


def kv_footprint(
    calls: list[Op],
    bytes_per_token: float = KV_BYTES_PER_TOKEN,
    block_size: int = BLOCK_TOKENS,
) -> list[Footprint]:
    """KV-cache bytes held by the LLM calls that hold a slot, at every time that changes."""
    # [Part 1] TODO: A call holds KV memory from slot_acquired to end, for its
    # prompt_tokens + output_tokens. Evaluate the footprint at each distinct
    # slot_acquired and end time, in increasing order; at time t a call counts
    # if slot_acquired <= t < end. private sums every call's tokens. shared
    # stores each distinct full prefix block once and adds each call's
    # remaining tokens (partial final block and output) separately.
    raise NotImplementedError("Part 1: fill kv_footprint")
    # END OF YOUR CODE
