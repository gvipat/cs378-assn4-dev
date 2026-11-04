"""Part 2 prefix-aware eviction and cache-aware routing (policies/prefix_policy.py)."""

import random
from pathlib import Path

import pytest
from conftest import REPO

from agentkv import score
from agentkv.harness import CacheHarness
from agentkv.policy import Block, Call, Event, ReplicaView, View
from agentkv.sandbox.loader import instantiate, load_policy_class

PATH = REPO / "policies" / "prefix_policy.py"


def policy(**params):
    return instantiate(load_policy_class(PATH.read_text(), str(PATH)), params)


def chain(h, *names):
    parent = None
    for n in names:
        h.allocate(n, parent)
        parent = n


def test_evicts_a_tail_where_lru_would_evict_the_root():
    h = CacheHarness(policy(), capacity=5)
    chain(h, "S1", "S2", "R1")
    for n in ("S1", "S2", "R1"):
        h.release(n)
    h.allocate("X")
    h.allocate("Y")
    h.allocate("Z")
    assert h.evicted == ["R1"]
    h.allocate("W")
    assert h.evicted == [
        "R1",
        "S2",
    ], "a parent becomes a leaf once its last child is evicted"


def test_plain_lru_without_leaf_first():
    h = CacheHarness(policy(leaf_first=False), capacity=4)
    chain(h, "S1", "S2", "R1")
    for n in ("S1", "S2", "R1"):
        h.release(n)
    h.allocate("X")
    h.allocate("Y")
    assert h.evicted == ["S1"]


def test_falls_back_to_lru_when_no_candidate_is_a_leaf():
    h = CacheHarness(policy(), capacity=3)
    chain(h, "A", "B")
    h.release("A")
    h.allocate("C")
    h.allocate("D")
    assert h.evicted == ["A"], "B (active) extends A, but A is the only cached block"


def test_eviction_is_tracked_from_events_alone():
    p = policy()
    for i, (block, parent) in enumerate(
        [("b0", None), ("b1", "p0"), ("b2", "p1"), ("c0", None)]
    ):
        prefix = "p" + block[1] if block[0] == "b" else "q0"
        p.on_event(
            Event(
                "alloc",
                i,
                {
                    "block": block,
                    "parent_hash": parent,
                    "depth": i,
                    "call": "x",
                    "block_hash": prefix,
                    "replica": 0,
                },
            ),
            {},
        )
    assert p.choose_victim(None, ["b0", "b1", "b2", "c0"], {}) == "b2"
    p.on_event(Event("evict", 9, {"block": "b2"}), {})
    p.on_event(Event("hit", 10, {"block": "c0", "call": "y"}), {})
    assert p.choose_victim(None, ["b0", "b1", "c0"], {}) == "b1"


def view(loads, resident, prefix):
    replicas = {
        r: ReplicaView(r, 100, 50, 25, 25, 0, n, n, True) for r, n in loads.items()
    }
    blocks = {
        f"r{r}-{p}": Block(f"r{r}-{p}", None, 0, 0, block_hash=p, replica=r)
        for r, ps in resident.items()
        for p in ps
    }
    call = Call("c", "j", "s", 48, tuple(prefix), 0.0)
    return View(
        1.0,
        16,
        200,
        100,
        50,
        50,
        0,
        0,
        {"c": call},
        {},
        blocks,
        {},
        random.Random(0),
        replicas,
    )


@pytest.mark.parametrize(
    "threshold,loads,expected",
    [
        (
            1,
            {0: 3, 1: 2},
            0,
        ),  # replica 0 is within the threshold and caches more of the prompt
        (0, {0: 3, 1: 2}, 1),  # only the least loaded replica is eligible
        (-1, {0: 3, 1: 2}, 1),  # affinity disabled
        (5, {0: 9, 1: 2}, 1),  # replica 0 is too far behind
    ],
)
def test_route_prefers_the_most_cached_blocks_within_the_threshold(
    threshold, loads, expected
):
    v = view(loads, {0: ["p1", "p2"], 1: ["p1", "p3"]}, ["p1", "p2", "p3"])
    assert policy(balance_threshold=threshold).route_call(v, "c", {}) == expected


def test_route_breaks_ties_by_load_then_id():
    v = view({0: 3, 1: 2}, {0: ["p1"], 1: ["p1"]}, ["p1", "p2"])
    assert policy(balance_threshold=2).route_call(v, "c", {}) == 1
    v = view({0: 2, 1: 2}, {}, ["p1"])
    assert policy(balance_threshold=2).route_call(v, "c", {}) == 0


@pytest.mark.slow
def test_reference_settings_reproduce_the_reference():
    r = score(
        str(PATH),
        split=str(REPO / "data/traces/public_dev/dev-W3-0.json"),
        params={"leaf_first": False, "balance_threshold": -1},
        workers=1,
    )
    assert r.valid, r.violations
    assert all(t.ratio == 1.0 for t in r.by_trace.values())
