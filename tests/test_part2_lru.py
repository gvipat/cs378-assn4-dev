"""Part 2: the public LRU oracle and invariant tests. Run: pytest tests/test_part2_lru.py"""

from conftest import LRU

from agentkv.harness import CacheHarness
from agentkv.replay import Replay
from agentkv.sandbox.client import InProcessClient, WorkerClient
from agentkv.sandbox.loader import instantiate, load_policy_class


def lru(params=None):
    return instantiate(load_policy_class(open(LRU).read(), LRU), params)


def test_oracle_one_token_blocks_capacity_three():
    h = CacheHarness(lru(), capacity=3)

    # Allocate A and B, then release both: A and B are reusable, A older than B.
    h.allocate("A")
    h.allocate("B")
    h.release("A")
    h.release("B")
    assert h.cached() == ["A", "B"]
    assert h.empty_slots() == 1

    # Allocate C, then release C: C uses the empty slot; nothing is evicted.
    h.allocate("C")
    h.release("C")
    assert h.evicted == []
    assert h.empty_slots() == 0

    # Access A, then release A: an exact-prefix hit. Recency is now B, C, A.
    assert h.access("A")
    h.release("A")

    # Allocate D under pressure: LRU evicts B. A and C stay reusable.
    h.allocate("D")
    assert h.evicted == ["B"]
    assert set(h.cached()) == {"A", "C"}
    assert not h.access("B"), "B's old lookup entry must be gone"

    # Hold A active, then allocate E: A cannot be chosen; LRU evicts C.
    assert h.access("A")
    h.allocate("E")
    assert h.evicted == ["B", "C"]
    assert h.active() == ["A", "D", "E"]
    h.kv.check()


def test_release_does_not_refresh_recency():
    h = CacheHarness(lru(), capacity=2)
    h.allocate("A")
    h.allocate("B")
    h.release("B")  # B released first...
    h.release("A")  # ...A released last, but A is still older by access
    h.allocate("C")
    assert h.evicted == ["A"]


def test_access_order_breaks_timestamp_ties():
    h = CacheHarness(lru(), capacity=2)
    # One completed GPU batch may report several events at the same time.
    # Release order is A, B, but the most recent access is to A.
    h._tick = lambda: None
    h.allocate("A")
    h.allocate("B")
    h.access("A")
    h.release("A")
    h.release("A")
    h.release("B")
    h.allocate("C")
    assert h.evicted == ["B"]


def test_least_recent_block_still_active_is_skipped():
    h = CacheHarness(lru(), capacity=3)
    h.allocate("A")  # A stays active: the least recent block, but not a candidate
    h.allocate("B")
    h.release("B")
    h.allocate("C")
    h.release("C")
    h.allocate("D")  # LRU among the candidates B and C is B
    assert h.evicted == ["B"]


def test_fixed_cap_and_valid_replay(chat_trace):
    for cap in (1, 4, 8):
        r = Replay(
            chat_trace,
            InProcessClient(
                LRU,
                params={"cap": cap},
                block_size=16,
                total_blocks=chat_trace.memory.num_blocks,
            ),
            check_every=1,
        ).run()
        assert r.valid, r.violation
        assert r.diag["target_concurrency_calls"] > 0


def test_lru_under_pressure_in_the_sandbox(pressure_trace):
    m = pressure_trace.memory
    r = Replay(
        pressure_trace,
        WorkerClient(
            LRU, params={"cap": 8}, block_size=m.block_size, total_blocks=m.num_blocks
        ),
        check_every=1,
    ).run()
    assert r.valid, r.violation
    assert r.diag["evictions"] > 0
    assert all(f > 0 for f in r.flows)
