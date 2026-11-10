"""Simulator correctness, determinism, causality, and opaque ids."""

import json

import pytest

from conftest import FIFO, build_chat_trace

from agentkv.policy import EVENT_FIELDS, Policy
from agentkv.replay import Replay
from agentkv.sandbox.client import InProcessClient
from agentkv.trace import load


def replay(trace, policy=FIFO, params=None, seed=0, macro=True, check=0):
    m = trace.memory
    client = InProcessClient(
        policy,
        params=params,
        seed=seed,
        block_size=m.block_size,
        total_blocks=m.num_blocks,
    )
    rp = Replay(trace, client, seed=seed, check_every=check)
    rp.engine.macro = macro
    return rp.run(), client


def test_decode_runs_match_single_steps(pressure_trace):
    for cap in (2, 8):
        a, _ = replay(pressure_trace, params={"cap": cap}, check=1)
        b, _ = replay(pressure_trace, params={"cap": cap}, macro=False, check=1)
        assert a.valid and b.valid
        assert a.flows == pytest.approx(b.flows, rel=1e-12)
        assert a.diag["preemptions"] == b.diag["preemptions"]
        assert a.diag["evictions"] == b.diag["evictions"]


def test_deterministic(pressure_trace):
    a, _ = replay(pressure_trace, params={"cap": 8})
    b, _ = replay(pressure_trace, params={"cap": 8})
    assert a.flows == b.flows and a.diag["events"] == b.diag["events"]


def test_rekeyed_ids_produce_identical_decisions(pressure_trace):
    a, ca = replay(pressure_trace, params={"cap": 8}, seed=0)
    b, cb = replay(pressure_trace, params={"cap": 8}, seed=1)
    assert a.flows == b.flows
    ids_a = set(ca.rt.blocks) | set(ca.rt.jobs)
    ids_b = set(cb.rt.blocks) | set(cb.rt.jobs)
    assert not ids_a & ids_b  # different opaque ids...
    assert a.diag == {
        **b.diag,
        "wall_s": a.diag["wall_s"],
        "policy_cpu_s": a.diag["policy_cpu_s"],
        "max_hook_cpu_s": a.diag["max_hook_cpu_s"],
    }  # ...same behavior


class Recorder(Policy):
    def __init__(self, params=None):
        super().__init__(params)
        self.log = []

    def on_event(self, event, state):
        self.log.append((event.kind, event.t, dict(event._f)))

    def target_concurrency(self, view, state):
        self.log.append(
            ("tc", view.now, {"ready": len(view.ready), "flight": view.in_flight})
        )
        return 4

    def choose_victim(self, view, candidates, state):
        return candidates[0]


def _normalize(log):
    """Replace opaque ids by first-appearance order so two traces can be compared."""
    names = {}

    def norm(v):
        if isinstance(v, str) and v[:1] in "bjcsr" and len(v) == 17:
            return names.setdefault(v, f"{v[0]}{len(names)}")
        if isinstance(v, list):
            return [norm(x) for x in v]
        return v

    return [(k, t, {f: norm(v) for f, v in d.items()}) for k, t, d in log]


def test_events_preserve_prefix_identity_and_residence_lifetimes(pressure_trace):
    result, client = replay(pressure_trace, policy=Recorder, check=1)
    assert result.valid, result.violation
    resident, seen_ids, seen_hashes, prompts = {}, set(), set(), {}
    recomputed = False
    for kind, _, fields in client.rt.policy.log:
        if kind == "tc":
            continue
        assert set(fields) == set(EVENT_FIELDS[kind]), (kind, fields)
        if kind == "ready":
            prompts[fields["call"]] = fields["block_hashes"]
        elif kind == "dispatch":
            assert fields["replica"] == 0
        elif kind == "alloc":
            block, prefix = fields["block"], fields["block_hash"]
            assert block != prefix and block not in seen_ids
            assert fields["replica"] == 0
            parent, depth = fields["parent_hash"], fields["depth"]
            assert (parent is None) == (depth == 0)
            if parent is not None:
                assert parent in seen_hashes
            prompt = prompts[fields["call"]]
            if depth < len(prompt):
                assert prefix == prompt[depth]
                assert parent == (prompt[depth - 1] if depth else None)
            recomputed |= prefix in seen_hashes
            resident[block] = prefix
            seen_ids.add(block)
            seen_hashes.add(prefix)
        elif kind in ("hit", "release"):
            assert fields["block"] in resident
        elif kind == "evict":
            del resident[fields["block"]]
    assert result.diag["evictions"] > 0 and recomputed
    assert seen_ids.isdisjoint(seen_hashes)
    assert {b.id: b.block_hash for b in client.rt.blocks.values()} == resident


def test_no_future_information(tmp_path):
    """Two traces identical except for the last job: everything the policy saw
    before that job arrived must be identical."""
    a = build_chat_trace(tmp_path / "a.json", jobs=6, gap=2.0)
    d = json.loads((tmp_path / "a.json").read_text())
    last = d["jobs"][-1]
    last["arrival"] = 60.0
    for n in last["nodes"]:
        if n["type"] == "tool":
            n["delay"] = 7.5  # tool durations of the future job differ too
    (tmp_path / "b.json").write_text(json.dumps(d))
    b = load(tmp_path / "b.json")
    t_cut = min(j.arrival for j in a.jobs[-1:])
    ra, ca = replay(a, policy=Recorder)
    rb, cb = replay(b, policy=Recorder)
    la = [e for e in ca.rt.policy.log if e[1] < t_cut]
    lb = [e for e in cb.rt.policy.log if e[1] < t_cut]
    assert la and _normalize(la) == _normalize(lb)
    for kind, _, fields in ca.rt.policy.log:  # no output lengths before completion
        if kind in ("ready", "dispatch", "admit"):
            assert "output_tokens" not in fields


def test_liveness_with_largest_call_at_half_memory(tmp_path):
    t = build_chat_trace(
        tmp_path / "tight.json", jobs=8, num_blocks=100, fanout=2, gap=0.0, think=(0, 0)
    )
    biggest = max(-(-n.n_total // t.memory.block_size) for _, n in t.llm_nodes())
    assert biggest <= t.memory.num_blocks // 2
    for policy_params in ({"cap": 8}, {"cap": 1}):
        r, _ = replay(t, params=policy_params, check=1)
        assert r.valid and all(f < float("inf") for f in r.flows)


def test_cap_one_serializes(chat_trace):
    r, c = replay(chat_trace, params={"cap": 1})
    assert r.valid
    assert r.diag["preemptions"] == 0


@pytest.mark.parametrize("seed", range(4))
def test_decode_runs_match_single_steps_sweep(tmp_path, seed):
    for blocks in (96, 130, 200):
        t = build_chat_trace(
            tmp_path / f"s{seed}-{blocks}.json",
            jobs=10,
            seed=seed,
            num_blocks=blocks,
            fanout=seed % 3,
            gap=0.05 * seed,
            think=(0.0, 0.5 * seed),
        )
        for cap in (3, 8):
            a, _ = replay(t, params={"cap": cap})
            b, _ = replay(t, params={"cap": cap}, macro=False)
            assert a.flows == pytest.approx(b.flows, rel=1e-9), (blocks, cap)
            for k in (
                "preemptions",
                "evictions",
                "cache_hit_tokens",
                "recomputed_tokens",
                "events",
            ):
                assert a.diag[k] == b.diag[k], (blocks, cap, k)
