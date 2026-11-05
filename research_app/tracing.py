"""The tracing client (supplied): every LLM and tool call goes through it.

Each operation has an id identifying its position in the workflow, such as
`R03/f2/fetch1`, a list of dependencies, and one record per attempt. LLM calls hold one slot of
the shared concurrency cap for all of their attempts. The recorder writes
the run as a JSONL event log (the Part 1 trace) and `to_trace_job` turns a
report into one job of an agentkv replay trace.

Live mode waits for simulated local-tool delays and retry backoff. Record
mode (staff trace generation) records those assigned delays without waiting.
Network search backends are awaited and their service times are measured.
"""

from __future__ import annotations

import asyncio
import json
import inspect
import time
from array import array
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agentkv import tokenizer
from agentkv.trace import JobBuilder, chain_hashes, output_text, segment_id

from .citations import check_report
from .concurrency import ConcurrencyLimiter
from .corpus import PermanentToolError, TransientToolError
from .llm import LLM, LLMResponse, PermanentLLMError, TransientLLMError
from .prompts import Msg, render
from .search import DocumentStore, SearchBackend

SCHEMA = "research-trace/1"
BLOCK_SIZE = 256


@dataclass(frozen=True)
class Budgets:
    llm_calls: int = 8
    searches: int = 5
    fetches: int = 15
    attempts: int = 3


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Attempt:
    attempt_id: str
    start: float
    end: float
    duration: float  # service time: measured for APIs, assigned for local tools
    status: str  # ok | transient | permanent | cancelled
    error: str | None = None
    backoff: float = 0.0  # wait after this attempt, before the next


@dataclass
class Op:
    op_id: str
    kind: str  # llm | search | fetch | validate
    stage: str
    deps: list[str]
    order: tuple
    status: str = "pending"  # ok | failed | cancelled
    start: float = 0.0
    end: float = 0.0
    attempts: list[Attempt] = field(default_factory=list)
    prompt: list[str] = field(default_factory=list)  # segment ids (llm)
    output: str | None = None  # segment id of the frozen output (llm)
    input_tokens: int = 0  # as billed by the backend
    output_tokens: int = 0
    prompt_tokens: int = 0  # under the frozen local tokenizer
    block_hashes: list[str] = field(
        default_factory=list
    )  # chained hashes of full 256-token blocks
    model: str | None = None
    slot_requested: float | None = None
    slot_acquired: float | None = None
    args: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    validation: dict | None = None

    @property
    def retries(self) -> int:
        return max(0, len(self.attempts) - 1)

    @property
    def delay(self) -> float:
        return sum(a.duration + a.backoff for a in self.attempts)


@dataclass
class Recording:
    """Store one report's operations, model outputs, and final validation result."""

    report_id: str
    question_id: str
    question: str
    arrival: float
    snapshot: str
    ops: list[Op] = field(default_factory=list)
    segments: dict[str, str] = field(default_factory=dict)
    start: float = 0.0
    end: float = 0.0
    report: dict | None = None
    valid: bool | None = None
    errors: list[str] = field(default_factory=list)
    branches: list[dict] = field(default_factory=list)

    def op(self, op_id: str) -> Op:
        return next(o for o in self.ops if o.op_id == op_id)

    def counts(self) -> dict:
        kinds = [o.kind for o in self.ops]
        return {
            "llm_calls": kinds.count("llm"),
            "searches": kinds.count("search"),
            "fetches": kinds.count("fetch"),
        }


class Recorder:
    """Collect events and prompt segments for all reports in a run, using one clock."""

    def __init__(
        self,
        *,
        cap: ConcurrencyLimiter,
        live: bool = True,
        backend: str = "",
        meta: dict | None = None,
    ):
        self.cap = cap
        self.live = live
        self.backend = backend
        self.meta = meta or {}
        self.t0 = time.monotonic()
        self.events: list[dict] = []
        self.recordings: dict[str, Recording] = {}
        self.segments: dict[str, str] = {}
        self._seg_tokens: dict[str, array] = {}
        self.llm_in_flight = 0
        self.llm_peak = 0
        self.cap_violations = 0  # LLM calls started while in_flight > limit

    def now(self) -> float:
        return round(time.monotonic() - self.t0, 6)

    def emit(self, event: str, **fields) -> None:
        self.events.append({"event": event, "t": self.now(), **fields})

    def segment(self, text: str) -> str:
        sid = segment_id(text)
        if sid not in self.segments:
            self.segments[sid] = text
            self._seg_tokens[sid] = array("I", tokenizer.encode(text))
        return sid

    def tokens(self, sids: list[str]) -> array:
        out = array("I")
        for sid in sids:
            out.extend(self._seg_tokens[sid])
        return out

    def open_report(
        self,
        report_id: str,
        question: dict,
        *,
        arrival: float,
        snapshot: str,
        llm: LLM,
        tools: SearchBackend,
        budgets: Budgets = Budgets(),
        backoff: float = 0.5,
    ) -> "TracingClient":
        rec = Recording(
            report_id,
            question["id"],
            question["question"],
            arrival,
            snapshot,
            start=self.now(),
        )
        self.recordings[report_id] = rec
        self.emit(
            "report_start",
            report_id=report_id,
            question_id=rec.question_id,
            question=rec.question,
            arrival=arrival,
            snapshot=snapshot,
        )
        return TracingClient(self, rec, llm, tools, budgets=budgets, backoff=backoff)

    def close_report(self, rec: Recording) -> None:
        rec.end = self.now()
        rec.ops.sort(key=lambda o: o.order)
        rec.segments = {
            sid: self.segments[sid]
            for o in rec.ops
            for sid in (o.prompt + ([o.output] if o.output else []))
        }
        self.emit(
            "report_end",
            report_id=rec.report_id,
            valid=rec.valid,
            errors=rec.errors,
            **rec.counts(),
        )

    async def set_cap(self, limit: int) -> None:
        await self.cap.set_limit(limit)
        self.emit("cap_change", limit=limit)

    def export_jsonl(self, path: str | Path) -> None:
        """Write the trace header, prompt segments, and events to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            head = {
                "event": "run_start",
                "schema": SCHEMA,
                "backend": self.backend,
                "live": self.live,
                "cap": self.cap.limit,
                "llm_peak": self.llm_peak,
                "cap_violations": self.cap_violations,
                "tokenizer": tokenizer.TOKENIZER_NAME,
                "block_size": BLOCK_SIZE,
                **self.meta,
            }
            f.write(json.dumps(head) + "\n")
            for sid, text in self.segments.items():
                f.write(
                    json.dumps(
                        {
                            "event": "segment",
                            "sid": sid,
                            "n_tokens": len(self._seg_tokens[sid]),
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            for ev in self.events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")


class TracingClient:
    """Run and record the LLM and tool operations for one report.

    Calls return (result, op), where op records the operation and its attempts.
    The result is None if the operation cannot complete successfully.
    """

    def __init__(
        self,
        recorder: Recorder,
        rec: Recording,
        llm: LLM,
        tools: SearchBackend,
        *,
        budgets: Budgets = Budgets(),
        backoff: float = 0.5,
    ):
        self.r = recorder
        self.rec = rec
        self.llm = llm
        self.tools = tools
        self.budgets = budgets
        self.backoff = backoff

    @property
    def report_id(self) -> str:
        return self.rec.report_id

    @property
    def corpus(self) -> DocumentStore:
        return self.tools.corpus

    def _new_op(
        self, op_id: str, kind: str, stage: str, deps: list[str], order: tuple
    ) -> Op:
        if any(o.op_id == op_id for o in self.rec.ops):
            raise ValueError(f"duplicate op id {op_id}")
        limits = {
            "llm": self.budgets.llm_calls,
            "search": self.budgets.searches,
            "fetch": self.budgets.fetches,
        }
        if (
            kind in limits
            and self.rec.counts()[
                {"llm": "llm_calls", "search": "searches", "fetch": "fetches"}[kind]
            ]
            >= limits[kind]
        ):
            raise BudgetExceeded(
                f"{self.report_id}: {kind} budget of {limits[kind]} exhausted"
            )
        op = Op(op_id, kind, stage, list(deps), order, start=self.r.now())
        self.rec.ops.append(op)
        self.r.emit(
            "op_start",
            report_id=self.report_id,
            op_id=op_id,
            kind=kind,
            stage=stage,
            deps=op.deps,
        )
        return op

    def _end_op(self, op: Op, status: str) -> None:
        op.status = status
        op.end = self.r.now()
        extra = {}
        if op.kind == "llm":
            extra = {
                "prompt": op.prompt,
                "output": op.output,
                "input_tokens": op.input_tokens,
                "output_tokens": op.output_tokens,
                "prompt_tokens": op.prompt_tokens,
                "block_hashes": op.block_hashes,
                "model": op.model,
                "slot_requested": op.slot_requested,
                "slot_acquired": op.slot_acquired,
            }
        elif op.kind == "validate":
            extra = {"validation": op.validation}
        else:
            extra = {"args": op.args, "result": op.result}
        self.r.emit(
            "op_end",
            report_id=self.report_id,
            op_id=op.op_id,
            status=status,
            attempts=len(op.attempts),
            retries=op.retries,
            **extra,
        )

    def _attempt(
        self,
        op: Op,
        n: int,
        start: float,
        duration: float,
        status: str,
        error: str | None = None,
        backoff: float = 0.0,
    ) -> None:
        a = Attempt(
            f"{op.op_id}#a{n}",
            start,
            self.r.now(),
            round(duration, 6),
            status,
            error,
            round(backoff, 6),
        )
        op.attempts.append(a)
        self.r.emit("attempt", report_id=self.report_id, op_id=op.op_id, **asdict(a))

    # ------------------------------------------------------------ LLM

    async def llm_call(
        self,
        stage: str,
        messages: list[Msg],
        *,
        op_id: str,
        deps: list[str],
        order: tuple,
        max_tokens: int,
    ) -> tuple[LLMResponse | None, Op]:
        op = self._new_op(op_id, "llm", stage, deps, order)
        op.prompt = [self.r.segment(s) for s in render(messages)]
        tokens = self.r.tokens(op.prompt)
        op.prompt_tokens = len(tokens)
        op.block_hashes = [f"{h:032x}" for h in chain_hashes(tokens, BLOCK_SIZE)]
        resp = None
        status = "failed"
        op.slot_requested = self.r.now()
        async with self.r.cap.slot():
            op.slot_acquired = self.r.now()
            self.r.llm_in_flight += 1
            self.r.llm_peak = max(self.r.llm_peak, self.r.llm_in_flight)
            if self.r.llm_in_flight > self.r.cap.limit:
                self.r.cap_violations += 1
            try:
                for n in range(1, self.budgets.attempts + 1):
                    t, wall = self.r.now(), time.monotonic()
                    try:
                        resp = await self.llm.complete(
                            messages, max_tokens=max_tokens, stage=stage
                        )
                        self._attempt(op, n, t, time.monotonic() - wall, "ok")
                        status = "ok"
                        break
                    except TransientLLMError as e:
                        wait = (
                            self.backoff * 2 ** (n - 1)
                            if n < self.budgets.attempts
                            else 0.0
                        )
                        self._attempt(
                            op, n, t, time.monotonic() - wall, "transient", str(e), wait
                        )
                        if wait and self.r.live:
                            await asyncio.sleep(wait)  # the retry keeps this slot
                    except PermanentLLMError as e:
                        self._attempt(
                            op, n, t, time.monotonic() - wall, "permanent", str(e)
                        )
                        break
            except asyncio.CancelledError:
                self._attempt(op, len(op.attempts) + 1, op.start, 0.0, "cancelled")
                self._end_op(op, "cancelled")
                raise
            finally:
                self.r.llm_in_flight -= 1
        if resp is not None:
            op.output = self.r.segment(output_text(resp.text))
            op.input_tokens, op.output_tokens, op.model = (
                resp.input_tokens,
                resp.output_tokens,
                resp.model,
            )
        self._end_op(op, status)
        return resp, op

    # ------------------------------------------------------------ tools

    async def _tool(self, kind: str, op: Op, call):
        result = None
        status = "failed"
        try:
            for n in range(1, self.budgets.attempts + 1):
                t = self.r.now()
                d = self.tools.duration(kind, op.op_id, n)
                if d is not None and self.r.live:
                    await asyncio.sleep(d)
                wall = time.monotonic()
                try:
                    result = call(n)
                    if inspect.isawaitable(result):
                        result = await result
                    elapsed = d if d is not None else time.monotonic() - wall
                    self._attempt(op, n, t, elapsed, "ok")
                    status = "ok"
                    break
                except TransientToolError as e:
                    wait = 0.25 * 2 ** (n - 1) if n < self.budgets.attempts else 0.0
                    elapsed = d if d is not None else time.monotonic() - wall
                    self._attempt(op, n, t, elapsed, "transient", str(e), wait)
                    if wait and self.r.live:
                        await asyncio.sleep(wait)
                except PermanentToolError as e:
                    elapsed = d if d is not None else time.monotonic() - wall
                    self._attempt(op, n, t, elapsed, "permanent", str(e))
                    break
        except asyncio.CancelledError:
            self._attempt(op, len(op.attempts) + 1, t, self.r.now() - t, "cancelled")
            self._end_op(op, "cancelled")
            raise
        return result, status

    async def search(
        self, query: str, *, op_id: str, deps: list[str], order: tuple, k: int = 5
    ):
        op = self._new_op(op_id, "search", "search", deps, order)
        op.args = {"query": query, "k": k}
        hits, status = await self._tool(
            "search",
            op,
            lambda n: self.tools.search(query, op_id=op_id, attempt=n, k=k),
        )
        if hits is not None:
            op.result = {
                "hits": [
                    {"doc_id": h.doc_id, "snapshot": h.snapshot, "score": h.score}
                    for h in hits
                ]
            }
        self._end_op(op, status)
        return hits, op

    async def fetch(self, doc_id: str, *, op_id: str, deps: list[str], order: tuple):
        op = self._new_op(op_id, "fetch", "fetch", deps, order)
        op.args = {"doc_id": doc_id}
        doc, status = await self._tool(
            "fetch", op, lambda n: self.tools.fetch(doc_id, op_id=op_id, attempt=n)
        )
        if doc is not None:
            op.result = {
                "doc_id": doc.doc_id,
                "snapshot": doc.sha256,
                "uri": doc.uri,
                "chars": len(doc.text),
            }
        self._end_op(op, status)
        return doc, op

    def validate(
        self,
        report,
        *,
        op_id: str,
        deps: list[str],
        order: tuple,
        snapshot: str | None = None,
    ):
        """Run the citation checker as a traced (zero-latency) step."""
        op = self._new_op(op_id, "validate", "validate", deps, order)
        t = self.r.now()
        errors = check_report(report, self.corpus, snapshot=snapshot)
        self._attempt(
            op, 1, t, 0.0, "ok"
        )  # local and instant; recorded as zero so replays are deterministic
        op.validation = {"valid": not errors, "errors": [str(e) for e in errors]}
        self._end_op(op, "ok")
        return errors, op


# ---------------------------------------------------------------- export


def to_trace_job(rec: Recording, job: JobBuilder) -> dict[str, str]:
    """Append one report's DAG to an agentkv trace job; returns op_id -> node id.

    - A successful LLM op becomes an llm node: its prompt segments and its
      frozen output text. LLM ops that failed or were cancelled are dropped
      and their dependents inherit their dependencies.
    - Search and fetch ops become tool nodes whose delay is the sum of every
      attempt's duration plus backoff, failed or not.
    - The citation check becomes a zero-delay tool node (stage `validate`).
    """
    ops = sorted(rec.ops, key=lambda o: o.order)
    by_id = {o.op_id: o for o in ops}
    nodes: dict[str, str] = {}

    def resolve(deps: list[str]) -> list[str]:
        out: list[str] = []
        for d in deps:
            if d in nodes:
                out.append(nodes[d])
            elif d in by_id:
                out.extend(resolve(by_id[d].deps))
        return list(dict.fromkeys(out))

    for op in ops:
        deps = resolve(op.deps)
        if op.kind == "llm":
            if op.status != "ok" or op.output is None:
                continue
            nodes[op.op_id] = job.llm(
                op.stage,
                prompt=[rec.segments[s] for s in op.prompt],
                output=rec.segments[op.output],
                deps=deps,
            )
        else:
            nodes[op.op_id] = job.tool(op.stage, op.delay, deps=deps)
    return nodes
