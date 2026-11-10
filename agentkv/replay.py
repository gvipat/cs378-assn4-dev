"""Replay application DAGs and expose causal state to serving policies.

The driver tracks arrivals, call dependencies, tool delays, and dispatch.
Engines handle batching and KV memory. Policy hooks control concurrency,
dispatch order, eviction, admission, and preemption.

This module includes the legacy single-replica simulator. The assignment's
score() and viz.record() APIs use vidur_replay.VidurReplay, which reuses the
application driver and runs Vidur's scheduler on one or two replicas.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from .engine.kv import EngineError
from .engine.scheduler import SchedulingHooks
from .engine.seq import Seq
from .engine.sim import SimEngine
from .engine.timing import TimingModel
from .policy import MAX_CONCURRENCY
from .sandbox.runtime import PolicyError
from .trace import Job, LLMNode, ToolNode, Trace

_ARRIVAL, _TOOL_DONE = 0, 1


class Rekey:
    """Replace internal ids with opaque per-run ids while preserving equality and ancestry."""

    def __init__(self, salt: bytes):
        self.salt = salt
        self._blocks: dict[int, str] = {}
        self._unblock: dict[str, int] = {}

    def _h(self, prefix: str, data: bytes) -> str:
        return prefix + hashlib.blake2b(data, key=self.salt, digest_size=8).hexdigest()

    def block(self, h: int | None) -> str | None:
        if h is None:
            return None
        pid = self._blocks.get(h)
        if pid is None:
            pid = self._h("b", h.to_bytes(16, "little"))
            self._blocks[h] = pid
            self._unblock[pid] = h
        return pid

    def unblock(self, pid) -> int | None:
        return self._unblock.get(pid) if isinstance(pid, str) else None

    def stage(self, s: str) -> str:
        return self._h("s", b"stage:" + s.encode())

    def job(self, j: str) -> str:
        return self._h("j", b"job:" + j.encode())

    def call(self, j: str, n: str) -> str:
        return self._h("c", f"call:{j}/{n}".encode())


@dataclass
class _Call:
    key: str
    pid: str
    job: "_Job"
    node: LLMNode
    stage: str
    ready_at: float
    dispatched_at: float = 0.0
    admitted_at: float | None = None
    done_at: float | None = None


@dataclass
class _Job:
    job: Job
    pid: str
    pending: dict
    succ: dict
    left: int
    done_at: float | None = None


@dataclass
class RunResult:
    trace_id: str
    family: str
    flows: list[float]
    violation: str | None = None
    diag: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.violation is None

    @property
    def mean_flow(self) -> float:
        return sum(self.flows) / len(self.flows) if self.flows else math.inf


def run_policy(
    trace: Trace,
    policy_path: str,
    *,
    params: dict | None = None,
    seed: int = 0,
    isolate: bool = True,
    timing: TimingModel | None = None,
    engine=None,
) -> RunResult:
    """Start the policy (sandboxed unless isolate=False) and replay the trace.
    A policy that fails to load yields an invalid run, like any other violation."""
    from .sandbox.client import InProcessClient, WorkerClient

    kw = dict(
        params=params,
        seed=seed,
        block_size=trace.memory.block_size,
        total_blocks=trace.memory.num_blocks,
    )
    try:
        client = (
            WorkerClient(policy_path, **kw)
            if isolate
            else InProcessClient(policy_path, **kw)
        )
    except PolicyError as exc:
        return RunResult(
            trace.trace_id,
            trace.family,
            [math.inf] * len(trace.jobs),
            f"{exc.category}: {exc.detail}" if exc.detail else exc.category,
            {},
        )
    return Replay(trace, client, seed=seed, timing=timing, engine=engine).run()


class _PolicyHooks(SchedulingHooks):
    """Use the policy's scheduling hooks where overridden, and the engine defaults otherwise."""

    def __init__(self, replay: "Replay"):
        self.r = replay

    def _defines(self, hook: str) -> bool:
        return hook in getattr(self.r.client, "hooks", ())

    def admission_order(self, waiting: list[Seq]) -> list[Seq]:
        if not self._defines("admission_order"):
            return waiting
        by_pid = {s.owner.pid: s for s in waiting}
        order = self.r.client.admission_order(self.r._hdr(), list(by_pid))
        if sorted(order) != sorted(by_pid):
            raise PolicyError(
                "invalid",
                "admission_order must return every waiting call id exactly once",
            )
        return [by_pid[pid] for pid in order]

    def choose_preempted(self, running: list[Seq]) -> Seq:
        if not self._defines("choose_preempted"):
            return running[-1]
        by_pid = {s.owner.pid: s for s in running}
        pid = self.r.client.choose_preempted(self.r._hdr(), list(by_pid))
        if pid not in by_pid:
            raise PolicyError(
                "invalid",
                f"choose_preempted returned {pid!r}, which is not a running call",
            )
        return by_pid[pid]


class Replay:

    def __init__(
        self,
        trace: Trace,
        client,
        *,
        seed: int = 0,
        timing: TimingModel | None = None,
        engine=None,
        verify: bool = False,
        check_every: int = 0,
    ):
        self.trace = trace
        self.client = client
        self.timing = timing or TimingModel.load()
        self.rekey = Rekey(
            hashlib.blake2b(f"{trace.sha256}:{seed}".encode(), digest_size=16).digest()
        )
        self._residences: dict[int, str] = {}
        self._residence_hashes: dict[str, int] = {}
        self.check_every = check_every
        self.verify = verify
        hooks = _PolicyHooks(self)
        if engine is None:
            engine = SimEngine(
                trace.memory,
                trace.engine,
                timing=self.timing,
                choose_victim=self._victim,
                emit_block=self._emit_block,
                on_call=self._on_call,
                hooks=hooks,
                verify=verify,
            )
        else:
            engine.attach(
                choose_victim=self._victim,
                emit_block=self._emit_block,
                on_call=self._on_call,
                hooks=hooks,
            )
        self.engine = engine
        self.kv = engine.kv
        self.sched = engine.sched
        self.realtime = getattr(engine, "realtime", False)
        self.needs_tokens = verify or getattr(engine, "needs_tokens", False)
        self.now = 0.0
        self.t = 0.0  # timestamp for events being emitted
        self.heap: list = []
        self._tie = 0
        self.ready: OrderedDict[str, _Call] = OrderedDict()
        self.inflight: dict[int, _Call] = {}
        self.jobs: dict[str, _Job] = {}
        self.flows: dict[str, float] = {}
        self.cache_hit_tokens = 0
        self.tc_calls = 0
        self.cv_calls = 0
        self.steps = 0

    # ------------------------------------------------------------ main loop

    def run(self) -> RunResult:
        for job in self.trace.jobs:
            self._push(job.arrival, _ARRIVAL, job)
        violation = None
        t0 = time.perf_counter()
        try:
            self._loop()
            self.t = self.now
            stats = self.client.finish()
        except PolicyError as exc:
            violation = f"{exc.category}: {exc.detail}" if exc.detail else exc.category
            stats = {}
        finally:
            self.client.close()
        flows = [self.flows.get(j.id, math.inf) for j in self.trace.jobs]
        return RunResult(
            self.trace.trace_id,
            self.trace.family,
            flows,
            violation,
            self._diag(stats, time.perf_counter() - t0, flows),
        )

    def _loop(self) -> None:
        n_jobs = len(self.trace.jobs)
        while len(self.flows) < n_jobs:
            self._process_external(self.now, inclusive=True)
            if len(self.flows) == n_jobs:
                break
            if self.engine.has_work():
                horizon = self.heap[0][0] if self.heap else math.inf
                self.t = self.now
                step = self.engine.begin_step(self.now, horizon)
                end = step.end
                self._process_external(end, inclusive=False)
                self.now = self.t = end
                finished = self.engine.end_step(step)
                self.steps += 1
                for seq in finished:
                    self._complete(seq)
                self._dispatch()
                if self.check_every and self.steps % self.check_every == 0:
                    self.kv.check()
            elif self.heap:
                self.now = max(self.now, self.heap[0][0])
                if self.realtime:
                    self.engine.wait_until(self.now)
            elif self.ready:
                raise PolicyError(
                    "invalid",
                    "dispatch_order held back every ready call while nothing "
                    "else could make progress",
                )
            else:
                raise EngineError(
                    "replay stalled: no engine work and no pending events"
                )

    def _push(self, t: float, kind: int, payload) -> None:
        self._tie += 1
        heapq.heappush(self.heap, (t, self._tie, kind, payload))

    def _process_external(self, limit: float, *, inclusive: bool) -> None:
        while self.heap and (
            self.heap[0][0] <= limit if inclusive else self.heap[0][0] < limit
        ):
            t, _, kind, payload = heapq.heappop(self.heap)
            self.t = t
            if kind == _ARRIVAL:
                self._arrive(payload)
            else:
                jr, node = payload
                self.client.event(
                    ("tool_done", t, jr.pid, self.rekey.stage(node.stage))
                )
                self._node_done(jr, node.id)
            self._dispatch()

    # ------------------------------------------------------------ jobs

    def _arrive(self, job: Job) -> None:
        succ: dict[str, list[str]] = {n.id: [] for n in job.nodes}
        pending = {}
        for n in job.nodes:
            pending[n.id] = len(n.deps)
            for d in n.deps:
                succ[d].append(n.id)
        jr = _Job(job, self.rekey.job(job.id), pending, succ, len(job.nodes))
        self.jobs[job.id] = jr
        self.client.event(("arrival", self.t, jr.pid))
        for n in job.nodes:
            if not n.deps:
                self._start(jr, n)

    def _start(self, jr: _Job, node) -> None:
        stage = self.rekey.stage(node.stage)
        if isinstance(node, ToolNode):
            self.client.event(("tool_start", self.t, jr.pid, stage))
            self._push(self.t + node.delay, _TOOL_DONE, (jr, node))
            return
        bs = self.trace.memory.block_size
        call = _Call(
            f"{jr.job.id}/{node.id}",
            self.rekey.call(jr.job.id, node.id),
            jr,
            node,
            stage,
            self.t,
        )
        prefix = [self.rekey.block(h) for h in node.hashes[: node.n_prompt // bs]]
        self.client.event(
            ("ready", self.t, call.pid, jr.pid, stage, node.n_prompt, prefix)
        )
        self.ready[call.key] = call

    def _node_done(self, jr: _Job, node_id: str) -> None:
        jr.left -= 1
        nodes = {n.id: n for n in jr.job.nodes}
        for s in jr.succ[node_id]:
            jr.pending[s] -= 1
            if jr.pending[s] == 0:
                self._start(jr, nodes[s])
        if jr.left == 0:
            jr.done_at = self.t
            flow = self.t - jr.job.arrival
            self.flows[jr.job.id] = flow
            self.client.event(("job_done", self.t, jr.pid, flow))

    # ------------------------------------------------------------ dispatch

    def _hdr(self) -> list:
        return [
            self.t,
            len(self.kv.empty),
            self.kv.num_active,
            len(self.sched.waiting),
            len(self.sched.running),
        ]

    def _dispatch(self) -> None:
        if not self.ready or len(self.inflight) >= MAX_CONCURRENCY:
            return
        self.tc_calls += 1
        cap = self.client.target_concurrency(self._hdr())
        if (
            not isinstance(cap, int)
            or isinstance(cap, bool)
            or not 1 <= cap <= MAX_CONCURRENCY
        ):
            raise PolicyError(
                "invalid",
                f"target_concurrency returned {cap!r}; it must be an int in 1..{MAX_CONCURRENCY}",
            )
        if "dispatch_order" in getattr(self.client, "hooks", ()):
            by_pid = {c.pid: k for k, c in self.ready.items()}
            order = self.client.dispatch_order(self._hdr(), list(by_pid))
            if len(set(order)) != len(order) or not set(order) <= set(by_pid):
                raise PolicyError(
                    "invalid", "dispatch_order must return distinct ids of ready calls"
                )
            keys = [by_pid[pid] for pid in order]
        else:
            keys = list(self.ready)
        for key in keys:
            if len(self.inflight) >= cap:
                break
            call = self.ready.pop(key)
            call.dispatched_at = self.t
            self.client.event(("dispatch", self.t, call.pid, 0))
            node = call.node
            tokens = self.trace.tokens(node) if self.needs_tokens else None
            seq = Seq(
                node.n_prompt, node.n_output, node.hashes, tokens=tokens, owner=call
            )
            self.inflight[seq.seq_id] = call
            self.engine.add(seq)

    def _complete(self, seq: Seq) -> None:
        call = self.inflight.pop(seq.seq_id)
        call.done_at = self.t
        node = call.node
        queue = (
            call.admitted_at if call.admitted_at is not None else self.t
        ) - call.dispatched_at
        self.client.event(
            (
                "complete",
                self.t,
                call.pid,
                call.job.pid,
                call.stage,
                node.n_prompt,
                node.n_output,
                seq.first_cached_tokens or 0,
                queue,
                self.t - call.dispatched_at,
            )
        )
        self._node_done(call.job, node.id)

    # ------------------------------------------------------------ engine callbacks

    def _emit_block(self, kind: str, h: int, seq: Seq | None, **kw) -> None:
        rk = self.rekey.block
        if kind == "alloc":
            block = self.kv.blocks[self.kv.prefix_map[h]]
            pid = self.rekey._h("r", f"0:{block.id}:{block.gen}".encode())
            self._residences[h] = pid
            self._residence_hashes[pid] = h
        else:
            pid = self._residences[h]
        if kind == "hit":
            self.client.event(("hit", self.t, pid, seq.owner.pid))
        elif kind == "alloc":
            self.client.event(
                (
                    "alloc",
                    self.t,
                    pid,
                    rk(kw["parent"]),
                    kw["depth"],
                    seq.owner.pid,
                    rk(h),
                    0,
                )
            )
        elif kind == "release":
            self.client.event(("release", self.t, pid, seq.owner.pid, kw["refs"]))
        elif kind == "evict":
            self.client.event(("evict", self.t, pid))
            del self._residences[h]
            del self._residence_hashes[pid]

    def _on_call(self, kind: str, seq: Seq) -> None:
        call: _Call = seq.owner
        if kind == "admit":
            if call.admitted_at is None:
                call.admitted_at = self.t
            self.cache_hit_tokens += seq.num_cached_tokens
            self.client.event(("admit", self.t, call.pid, seq.num_cached_tokens))
        elif kind == "preempt":
            self.client.event(("preempt", self.t, call.pid))

    def _victim(self, cached: dict) -> int:
        self.cv_calls += 1
        pid = self.client.choose_victim(self._hdr())
        h = self._residence_hashes.get(pid) if isinstance(pid, str) else None
        bid = self.kv.prefix_map.get(h) if h is not None else None
        if bid is None or bid not in cached:
            raise PolicyError(
                "invalid",
                f"choose_victim returned {pid!r}, which is not one of the candidates",
            )
        return bid

    # ------------------------------------------------------------ diagnostics

    def _ideal_flow(self, job: Job) -> float:
        """Job latency alone on an idle engine with a cold cache: its critical path."""
        finish: dict[str, float] = {}
        nodes = {n.id: n for n in job.nodes}

        def f(nid: str) -> float:
            if nid not in finish:
                n = nodes[nid]
                start = max((f(d) for d in n.deps), default=0.0)
                if isinstance(n, ToolNode):
                    cost = n.delay
                else:
                    cost = self.timing.prefill(
                        [(0, n.n_prompt)]
                    ) + self.timing.decode_run(
                        1, n.n_prompt + 1, max(n.n_output - 1, 0)
                    )
                finish[nid] = start + cost
            return finish[nid]

        return max(f(n.id) for n in job.nodes)

    def _diag(self, stats: dict, wall: float, flows: list[float]) -> dict:
        done = sorted(x for x in flows if math.isfinite(x))
        ideal = [self._ideal_flow(j) for j in self.trace.jobs]
        prompt = sum(n.n_prompt for _, n in self.trace.llm_nodes())
        p95 = (
            done[min(len(done) - 1, math.ceil(0.95 * len(done)) - 1)]
            if done
            else math.inf
        )
        ends = [jr.done_at for jr in self.jobs.values() if jr.done_at is not None]
        return {
            "mean_flow": sum(done) / len(done) if len(done) == len(flows) else math.inf,
            "p95_flow": p95,
            "makespan": (
                (max(ends) - min(j.arrival for j in self.trace.jobs))
                if ends
                else math.inf
            ),
            "mean_ideal_flow": sum(ideal) / len(ideal),
            "prompt_tokens": prompt,
            "cache_hit_tokens": self.cache_hit_tokens,
            "prefill_tokens": self.engine.prefill_tokens,
            "recomputed_tokens": self.kv.recomputed_blocks
            * self.trace.memory.block_size,
            "preemptions": self.sched.preemptions,
            "evictions": self.kv.evictions,
            "prefill_steps": self.engine.prefill_steps,
            "decode_steps": self.engine.decode_steps,
            "busy_time": self.engine.busy_time,
            "prefill_time": getattr(self.engine, "prefill_time", 0.0),
            "policy_cpu_s": stats.get("cpu_s", getattr(self.client, "cpu", 0.0)),
            "max_hook_cpu_s": stats.get("max_hook_cpu_s", 0.0),
            "target_concurrency_calls": self.tc_calls,
            "choose_victim_calls": self.cv_calls,
            "events": stats.get("events", 0),
            "wall_s": wall,
        }
