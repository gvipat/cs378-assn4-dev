"""Check the Vidur adapter's causal observations and parallel replay."""

import math

import pytest

from agentkv.policy import Policy
from agentkv.sandbox.client import InProcessClient
from agentkv.vidur_replay import VidurReplay, prepare_trace, run_policy
from conftest import FIFO, build_chat_trace, write_policy


class ObservedPolicy(Policy):
    def __init__(self, params=None):
        super().__init__(params)
        self.known = {}
        self.calls = {}
        self.prefixes = {}
        self.saw_two = False
        self.admitted = set()

    def on_event(self, event, state):
        if event.kind == "ready":
            self.calls[event.call] = event.input_tokens
        elif event.kind == "admit":
            if event.call not in self.admitted:
                assert event.cached_tokens <= self.calls[event.call]
            self.admitted.add(event.call)
        elif event.kind == "alloc":
            assert event.block not in self.known
            self.known[event.block] = event.replica
            self.prefixes[event.block_hash] = event.replica
        elif event.kind == "hit":
            assert event.block in self.known
        elif event.kind == "evict":
            del self.known[event.block]

    def target_concurrency(self, view, state):
        self.saw_two |= len(view.replicas) == 2
        assert not hasattr(view, "next_arrival")
        return 8

    def choose_victim(self, view, candidates, state):
        assert all(
            view.blocks[b].replica == view.replica and view.blocks[b].ref_cnt == 0
            for b in candidates
        )
        return candidates[0]


def test_reblocking_preserves_workload(chat_trace):
    converted = prepare_trace(chat_trace)
    assert (
        converted.memory.num_blocks * 16
        == chat_trace.memory.num_blocks * chat_trace.memory.block_size
    )
    assert converted.jobs is not chat_trace.jobs
    assert [(n.n_prompt, n.n_output) for _, n in converted.llm_nodes()] == [
        (n.n_prompt, n.n_output) for _, n in chat_trace.llm_nodes()
    ]


@pytest.mark.parametrize("replicas", [1, 2])
def test_causal_cache_events_and_complete_release(chat_trace, replicas):
    trace = prepare_trace(chat_trace)
    policy = ObservedPolicy()
    client = InProcessClient(
        policy, seed=0, block_size=16, total_blocks=trace.memory.num_blocks * replicas
    )
    replay = VidurReplay(trace, client, replicas=replicas)
    result = replay.run()
    assert result.valid, result.violation
    assert all(math.isfinite(f) for f in result.flows)
    assert result.diag["cache_hit_tokens"] > 0
    assert policy.saw_two == (replicas == 2)
    assert all(
        s.pool.get_num_free_blocks() == s.pool.num_gpu_blocks for s in replay.schedulers
    )
    assert not any(s.pool.pending for s in replay.schedulers)


def test_independent_gpus_overlap(tmp_path):
    trace = build_chat_trace(
        tmp_path / "overlap.json", jobs=2, turns=1, num_blocks=1024, gap=0, think=(0, 0)
    )
    single = run_policy(trace, FIFO, params={"cap": 2}, replicas=1, isolate=False)
    dual = run_policy(trace, FIFO, params={"cap": 2}, replicas=2, isolate=False)
    assert single.valid and dual.valid
    assert all(t > 0 for t in dual.diag["replica_busy_time"])
    assert sum(dual.diag["replica_busy_time"]) > dual.diag["makespan"]


def test_sandbox_matches_local(chat_trace):
    local = run_policy(chat_trace, FIFO, replicas=2, isolate=False)
    sandbox = run_policy(chat_trace, FIFO, replicas=2, isolate=True)
    assert local.valid and sandbox.valid
    assert sandbox.flows == local.flows


@pytest.mark.parametrize(
    "method",
    [
        "def route_call(self, view, call, state): return 2",
        "def batch_token_budget(self, view, state): return 8192",
    ],
)
def test_invalid_hooks_rejected(chat_trace, tmp_path, method):
    path = write_policy(
        tmp_path,
        "from agentkv.policy import Policy\nclass Bad(Policy):\n"
        "    def target_concurrency(self, view, state): return 4\n"
        "    def choose_victim(self, view, candidates, state): return candidates[0]\n    "
        + method
        + "\n",
    )
    result = run_policy(chat_trace, path, isolate=True)
    assert not result.valid
    assert result.violation.startswith("invalid:")


def test_preemption_keeps_unfinished_prompt(pressure_trace):
    result = run_policy(
        pressure_trace, FIFO, params={"cap": 16}, isolate=False, replicas=1
    )
    assert result.valid, result.violation
    assert result.diag["preemptions"] > 0
    assert result.diag["recomputed_tokens"] > 0
