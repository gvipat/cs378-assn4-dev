"""Policy-side runtime: rebuilds the policy's View from the event stream and
calls the hooks. It runs inside the sandboxed worker (and in-process for unit
tests). It never sees anything but events and a small per-decision header,
so everything a policy can observe is causal by construction."""

from __future__ import annotations

import gc
import pickle
import random
import time

from ..policy import (
    EVENT_FIELDS,
    HOOK_CPU_LIMIT_S,
    STATE_LIMIT_BYTES,
    Block,
    Call,
    Event,
    JobInfo,
    Policy,
    ReplicaView,
    State,
    View,
    overridden_hooks,
    readonly,
)

_cpu = time.thread_time  # bound before the sandbox patches the time module

STATE_CHECK_EVERY = 256


class PolicyError(Exception):
    """Report a policy rule violation with a category that can be shown to students."""

    def __init__(self, category: str, detail: str = ""):
        super().__init__(f"{category}: {detail}" if detail else category)
        self.category = category
        self.detail = detail


class PolicyRuntime:

    def __init__(
        self, policy: Policy, *, block_size: int, total_blocks: int, seed: int
    ):
        self.policy = policy
        self.block_size = block_size
        self.total_blocks = total_blocks
        self.state = State()
        self.state.rng = random.Random(seed)
        self.blocks: dict[str, Block] = {}
        self.cand: dict[str, None] = {}
        self.ready: dict[str, Call] = {}
        self.calls: dict[str, Call] = {}
        self.jobs: dict[str, JobInfo] = {}
        self.cpu = 0.0
        self.max_cpu = 0.0
        self.n_events = 0
        self.n_decisions = 0
        self.hooks = overridden_hooks(
            policy
        )  # optional hooks the policy defines itself

    # ------------------------------------------------------------ events

    def apply(self, raw_events: list) -> None:
        on_event = self.policy.on_event
        for ev in raw_events:
            kind, t = ev[0], ev[1]
            fields = dict(zip(EVENT_FIELDS[kind], ev[2:]))
            self._mirror(kind, t, fields)
            e = Event(kind, t, fields)
            collecting = gc.isenabled()
            gc.disable()  # a collection triggered inside a hook would be charged to the policy
            c0 = _cpu()
            try:
                on_event(e, self.state)
            except Exception as exc:
                raise PolicyError(
                    "exception", f"on_event({kind}): {type(exc).__name__}: {exc}"
                ) from exc
            finally:
                dt = _cpu() - c0
                if collecting:
                    gc.enable()
            self._charge(dt, "on_event")
        self.n_events += len(raw_events)

    def _mirror(self, kind: str, t: float, f: dict) -> None:
        if kind == "hit":
            b = self.blocks[f["block"]]
            if b.ref_cnt == 0:
                del self.cand[b.id]
            b.ref_cnt += 1
            b.hits += 1
            b.last_access = t
        elif kind == "release":
            b = self.blocks[f["block"]]
            b.ref_cnt = f["ref_cnt"]
            if b.ref_cnt == 0:
                self.cand[b.id] = None
        elif kind == "alloc":
            c = self.calls.get(f["call"])
            self.blocks[f["block"]] = Block(
                f["block"],
                f["parent_hash"],
                f["depth"],
                1,
                0,
                t,
                t,
                c.stage if c else "",
                c.job if c else "",
                f.get("block_hash", f["block"]),
                f.get("replica", 0),
            )
        elif kind == "evict":
            del self.blocks[f["block"]]
            del self.cand[f["block"]]
        elif kind == "ready":
            self.ready[f["call"]] = Call(
                f["call"],
                f["job"],
                f["stage"],
                f["input_tokens"],
                tuple(f["block_hashes"]),
                t,
            )
            self.jobs[f["job"]].calls_ready += 1
        elif kind == "dispatch":
            c = self.ready.pop(f["call"])
            c.dispatched_at = t
            c.replica = f.get("replica", 0)
            self.calls[c.id] = c
            j = self.jobs[c.job]
            j.calls_ready -= 1
            j.calls_in_flight += 1
        elif kind == "admit":
            self.calls[f["call"]].admitted_at = t
        elif kind == "preempt":
            self.calls[f["call"]].admitted_at = None
        elif kind == "complete":
            c = self.calls.pop(f["call"])
            j = self.jobs[c.job]
            j.calls_in_flight -= 1
            j.calls_done += 1
        elif kind == "arrival":
            self.jobs[f["job"]] = JobInfo(f["job"], t)
        elif kind == "tool_start":
            self.jobs[f["job"]].tools_running += 1
        elif kind == "tool_done":
            self.jobs[f["job"]].tools_running -= 1
        elif kind == "job_done":
            del self.jobs[f["job"]]

    # ------------------------------------------------------------ decisions

    def view(self, hdr: list) -> View:
        now, empty, active, waiting, running = hdr[:5]
        replicas = {d["id"]: ReplicaView(**d) for d in hdr[5]} if len(hdr) > 5 else {}
        return View(
            now=now,
            block_size=self.block_size,
            total_blocks=self.total_blocks,
            empty_blocks=empty,
            active_blocks=active,
            cached_blocks=len(self.cand),
            waiting=waiting,
            running=running,
            ready=readonly(self.ready),
            calls=readonly(self.calls),
            blocks=readonly(self.blocks),
            jobs=readonly(self.jobs),
            rng=self.state.rng,
            replicas=readonly(replicas),
            replica=hdr[6] if len(hdr) > 6 else None,
        )

    def _decide(self, hook: str, hdr: list, *args):
        v = self.view(hdr)
        collecting = gc.isenabled()
        gc.disable()
        c0 = _cpu()
        try:
            out = getattr(self.policy, hook)(v, *args, self.state)
        except Exception as exc:
            raise PolicyError(
                "exception", f"{hook}: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            dt = _cpu() - c0
            if collecting:
                gc.enable()
        self._charge(dt, hook)
        self._after_decision()
        return out

    def target_concurrency(self, hdr: list):
        return self._decide("target_concurrency", hdr)

    def choose_victim(self, hdr: list, candidates=None):
        if candidates is None:
            candidates = self.cand
            if len(hdr) > 6 and hdr[6] is not None:
                candidates = [b for b in self.cand if self.blocks[b].replica == hdr[6]]
        return self._decide("choose_victim", hdr, tuple(candidates))

    def route_call(self, hdr: list, call: str):
        return self._decide("route_call", hdr, call)

    def batch_token_budget(self, hdr: list):
        return self._decide("batch_token_budget", hdr)

    def dispatch_order(self, hdr: list, ready: list):
        return _ids(self._decide("dispatch_order", hdr, tuple(ready)), "dispatch_order")

    def admission_order(self, hdr: list, waiting: list):
        return _ids(
            self._decide("admission_order", hdr, tuple(waiting)), "admission_order"
        )

    def choose_preempted(self, hdr: list, running: list):
        out = self._decide("choose_preempted", hdr, tuple(running))
        if not isinstance(out, str):
            raise PolicyError(
                "invalid", f"choose_preempted returned {type(out).__name__}, not str"
            )
        return out

    def _charge(self, dt: float, hook: str) -> None:
        self.cpu += dt
        if dt > self.max_cpu:
            self.max_cpu = dt
        if dt > HOOK_CPU_LIMIT_S:
            raise PolicyError(
                "timeout",
                f"{hook} used {dt * 1000:.1f} ms of CPU (limit {HOOK_CPU_LIMIT_S * 1000:.0f} ms)",
            )

    def _after_decision(self) -> None:
        self.n_decisions += 1
        if self.n_decisions % STATE_CHECK_EVERY == 0:
            self.check_state()

    def check_state(self) -> int:
        try:
            size = len(
                pickle.dumps((dict(self.state), self.policy.__dict__), protocol=4)
            )
        except Exception as exc:
            raise PolicyError(
                "state", f"state is not serializable: {type(exc).__name__}: {exc}"
            ) from exc
        if size > STATE_LIMIT_BYTES:
            raise PolicyError(
                "state", f"state is {size} bytes (limit {STATE_LIMIT_BYTES})"
            )
        return size

    def stats(self) -> dict:
        return {
            "cpu_s": self.cpu,
            "max_hook_cpu_s": self.max_cpu,
            "events": self.n_events,
            "decisions": self.n_decisions,
        }


def _ids(out, hook: str) -> list:
    """A hook's list of call ids, as a plain list of strings."""
    if not isinstance(out, (list, tuple)) or not all(isinstance(x, str) for x in out):
        raise PolicyError("invalid", f"{hook} must return a list of call ids (strings)")
    return list(out)
