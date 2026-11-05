"""Policy API for the Vidur assignment evaluator.

Implement target_concurrency and choose_victim; other methods have defaults.
The evaluator reveals observed events, current queues and cache contents,
and known prompt lengths. Output lengths, future arrivals, and tool durations
are hidden until those operations finish. Calls stay on the replica selected
at dispatch. Each replica has a separate cache.

A policy module defines one Policy subclass, or selects one with POLICY.
The sandbox allows no files, network, subprocesses, threads, or wall clock.
Use state.rng for randomness. Each hook has a 50 ms CPU limit; policy state
and instance attributes together must serialize to at most 4 MiB.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

MAX_CONCURRENCY = 16  # application-wide cap across all replicas
HOOK_CPU_LIMIT_S = 0.050
STATE_LIMIT_BYTES = 4 << 20
OPTIONAL_HOOKS = (
    "dispatch_order",
    "admission_order",
    "choose_preempted",
    "route_call",
    "batch_token_budget",
)


class Policy:
    """Subclass this policy and implement the required concurrency and eviction hooks."""

    def __init__(self, params: dict | None = None):
        self.params = dict(params or {})

    def on_event(self, event: "Event", state: "State") -> None:
        """Process an event in logical-time order, with a fixed order for simultaneous events."""

    def target_concurrency(self, view: "View", state: "State") -> int:
        """Return the maximum number of dispatched, unfinished calls, from 1 to MAX_CONCURRENCY."""
        raise NotImplementedError

    def choose_victim(
        self, view: "View", candidates: Sequence[str], state: "State"
    ) -> str:
        """Return the id of a cached block to evict from candidates.

        All candidates have zero active references and are listed by release time,
        oldest first.
        """
        raise NotImplementedError

    def dispatch_order(
        self, view: "View", ready: Sequence[str], state: "State"
    ) -> list[str]:
        """Return ready call ids in the order they should be dispatched.

        The default uses the given order, oldest first. You may return a subset;
        omitted calls stay ready. Dispatch stops when the concurrency cap is reached.
        """
        return list(ready)

    def admission_order(
        self, view: "View", waiting: Sequence[str], state: "State"
    ) -> list[str]:
        """Return a permutation of the waiting calls in admission order.

        The default is first come, first served. Calls whose chunked prefill has
        already started are excluded from waiting and finish their prefill first.
        """
        return list(waiting)

    def choose_preempted(
        self, view: "View", running: Sequence[str], state: "State"
    ) -> str:
        """Return the running call that should release its blocks when memory runs out.

        Running calls are listed in scheduling order. The default preempts the
        most recently scheduled call.
        """
        return running[-1]

    def route_call(self, view: "View", call: str, state: "State") -> int:
        """Choose a replica for a ready call. The default chooses the shortest queue.

        The assignment uses one replica in Part 2 and two in Part 3. Each has
        its own cache. A dispatched call stays on that replica until it finishes.
        """
        return min(view.replicas, key=lambda r: (view.replicas[r].in_flight, r))

    def batch_token_budget(self, view: "View", state: "State") -> int:
        """Choose 512, 1024, 2048, or 4096 tokens for the next Vidur batch.

        view.replica identifies the replica being scheduled. A batch may mix
        prompt and decode tokens; the default budget is 2048 tokens.
        """
        return 2048


def overridden_hooks(policy: Policy) -> tuple[str, ...]:
    """The optional hooks this policy defines itself. The engine applies the
    default rule for the others without calling the policy."""
    return tuple(
        h
        for h in OPTIONAL_HOOKS
        if getattr(getattr(policy, h), "__func__", None) is not getattr(Policy, h)
    )


class State(dict):
    """Store policy data for one trace; the evaluator creates fresh state for each trace."""

    rng: random.Random


# ------------------------------------------------------------------ events

#: Event kinds and their fields (besides `kind` and `t`):
EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    # a job arrived
    "arrival": ("job",),
    # an LLM call's dependencies finished; its prompt is known
    "ready": ("call", "job", "stage", "input_tokens", "block_hashes"),
    # the dispatcher sent a ready call to the engine
    "dispatch": ("call", "replica"),
    # the engine started prefilling the call; `cached_tokens` came from the cache
    "admit": ("call", "cached_tokens"),
    # a resident prefix block was matched by `call` (shared or reactivated)
    "hit": ("block", "call"),
    # a new prefix block became resident, computed by `call`
    "alloc": ("block", "parent_hash", "depth", "call", "block_hash", "replica"),
    # `call` dropped its reference; `ref_cnt` references remain
    "release": ("block", "call", "ref_cnt"),
    # a zero-reference block was overwritten (always the policy's choice)
    "evict": ("block",),
    # the engine preempted the call; it will be recomputed
    "preempt": ("call",),
    # an LLM call finished
    "complete": (
        "call",
        "job",
        "stage",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "queue_time",
        "latency",
    ),
    # a tool or think-time step started / finished (durations are not known in advance)
    "tool_start": ("job", "stage"),
    "tool_done": ("job", "stage"),
    # every node of the job finished
    "job_done": ("job", "flow_time"),
}


class Event:
    """Describe one engine event; kind selects the available fields listed in EVENT_FIELDS."""

    __slots__ = ("kind", "t", "_f")

    def __init__(self, kind: str, t: float, fields: dict):
        self.kind, self.t, self._f = kind, t, fields

    def __getattr__(self, name):
        try:
            return self._f[name]
        except KeyError:
            raise AttributeError(f"{self.kind} event has no field {name!r}") from None

    def get(self, name, default=None):
        return self._f.get(name, default)

    def __repr__(self):
        body = ", ".join(f"{k}={v!r}" for k, v in self._f.items())
        return f"Event({self.kind}, t={self.t:.3f}, {body})"


# ------------------------------------------------------------------ view


@dataclass(slots=True)
class Block:
    """Describe a resident block and its position in a prompt prefix.

    The id identifies one physical residence on one replica. `block_hash`
    identifies its contents and all preceding tokens (as in vLLM's prefix
    caching), so duplicate copies share a hash but have different ids.
    """

    id: str
    parent_hash: str | None  # block_hash of the preceding block (may not be resident)
    depth: int  # 0 for the first block of a prompt
    ref_cnt: int  # active references; 0 means it is a victim candidate
    hits: int = 0  # times matched since it became resident
    created: float = 0.0
    last_access: float = 0.0  # last alloc or hit (releases do not count)
    stage: str = ""  # stage of the call that computed it
    job: str = ""  # job of the call that computed it
    block_hash: str = ""  # content identity; duplicates can occupy different slots
    replica: int = 0


@dataclass(slots=True)
class Call:
    """An LLM call that is ready or in flight."""

    id: str
    job: str
    stage: str
    input_tokens: int
    block_hashes: tuple[
        str, ...
    ]  # block_hash of each full block of the prompt, root first
    ready_at: float
    dispatched_at: float | None = None
    admitted_at: float | None = None
    replica: int | None = None


@dataclass(frozen=True, slots=True)
class ReplicaView:
    """Current queue and memory counts for one GPU replica."""

    id: int
    total_blocks: int
    empty_blocks: int
    active_blocks: int
    cached_blocks: int
    waiting: int
    running: int
    in_flight: int
    busy: bool


@dataclass(slots=True)
class JobInfo:
    id: str
    arrival: float
    calls_done: int = 0
    calls_in_flight: int = 0
    calls_ready: int = 0
    tools_running: int = 0


@dataclass
class View:
    """A read-only snapshot passed to the decision hooks."""

    now: float
    block_size: int
    total_blocks: int
    empty_blocks: int  # blocks holding nothing reusable
    active_blocks: int  # blocks with references (includes partial blocks)
    cached_blocks: int  # resident zero-reference prefix blocks (== len(candidates))
    waiting: int  # not admitted, or returned to waiting after preemption
    running: int  # admitted calls, including unfinished chunked prefill
    ready: Mapping[str, Call]  # ready, not dispatched, oldest first
    calls: Mapping[str, Call]  # in flight (dispatched, not complete)
    blocks: Mapping[str, Block]  # resident prefix blocks
    jobs: Mapping[str, JobInfo]  # arrived, not done
    rng: random.Random = field(repr=False, default_factory=random.Random)
    replicas: Mapping[int, ReplicaView] = field(default_factory=dict)
    replica: int | None = None  # replica for an engine decision; None for dispatch

    @property
    def pressure(self) -> float:
        """Return the fraction of memory occupied by active or cached blocks."""
        return 1.0 - self.empty_blocks / self.total_blocks

    @property
    def in_flight(self) -> int:
        return len(self.calls)

    def num_cached_blocks(self, call: str, replica: int) -> int:
        """Return how many leading full blocks of a ready or in-flight call's prompt are resident on a replica.

        Matches `Call.block_hashes` against `Block.block_hash`; the keys of
        `blocks` are residence ids and never match a hash.
        """
        c = self.ready.get(call) or self.calls.get(call)
        if c is None:
            raise KeyError(f"{call!r} is neither ready nor in flight")
        resident = self.__dict__.get("_resident")
        if resident is None:
            resident = {}
            for b in self.blocks.values():
                resident.setdefault(b.replica, set()).add(b.block_hash)
            self.__dict__["_resident"] = resident
        have = resident.get(replica, ())
        n = 0
        for p in c.block_hashes:
            if p not in have:
                break
            n += 1
        return n


def readonly(d: dict) -> Mapping:
    return MappingProxyType(d)
