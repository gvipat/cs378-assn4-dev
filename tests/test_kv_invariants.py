"""Property tests of the KV lifecycle that the engine enforces."""

from array import array

import hypothesis.strategies as st
from hypothesis import given, settings

from agentkv.engine.kv import KVCacheManager
from agentkv.engine.scheduler import Scheduler
from agentkv.engine.seq import Seq
from agentkv.trace import chain_hashes

BS = 2


def make_seq(tokens, weak_hash=False, max_tokens=1):
    arr = array("I", tokens + [0] * max_tokens)
    hashes = chain_hashes(arr, BS)
    if weak_hash:  # force collisions: many different prefixes share a hash
        hashes = [h % 3 for h in hashes]
    return Seq(len(tokens), max_tokens, tuple(hashes), tokens=arr)


ops = st.lists(
    st.tuples(
        st.sampled_from(["admit", "admit", "step", "finish", "preempt"]),
        st.lists(st.integers(0, 2), min_size=1, max_size=9),
        st.integers(0, 10**6),
    ),
    min_size=1,
    max_size=40,
)


@settings(max_examples=150, deadline=None)
@given(ops, st.integers(3, 10), st.booleans())
def test_random_streams_keep_invariants(stream, num_blocks, weak_hash):
    Seq.block_size = BS
    import random

    victims = random.Random(0)

    def choose(cached):
        return victims.choice(list(cached))

    kv = KVCacheManager(num_blocks, BS, choose_victim=choose, verify=True)
    live: list[Seq] = []
    for op, tokens, r in stream:
        if op == "admit":
            s = make_seq(tokens, weak_hash)
            if s.num_blocks > num_blocks:
                continue
            hits = kv.can_allocate(s)
            if hits is None:
                continue
            for i, bid in enumerate(hits):  # no false hits, even with colliding hashes
                assert kv.blocks[bid].tokens == s.block(i)
            kv.allocate(s, hits)
            live.append(s)
        elif op == "step" and live:
            s = live[r % len(live)]
            if s.num_cached_tokens < s.num_tokens:  # compute the rest of the prompt
                kv.register(s, s.num_cached_tokens, s.num_tokens)
                s.num_cached_tokens = s.num_tokens
        elif op in ("finish", "preempt") and live:
            s = live.pop(r % len(live))
            kv.deallocate(s)
        kv.check()
        for blk in kv.blocks:
            assert blk.ref >= 0
        # unique ownership: a physical block appears at most once per sequence
        for s in live:
            assert len(set(s.block_table)) == len(s.block_table)
    for s in live:
        kv.deallocate(s)
    kv.check()
    assert kv.num_active == 0


def test_stale_entry_removed_on_overwrite():
    Seq.block_size = BS
    kv = KVCacheManager(
        2, BS, choose_victim=lambda cached: next(iter(cached)), verify=True
    )
    a = make_seq([1, 1, 9])
    kv.allocate(a, kv.can_allocate(a))
    kv.register(a, 0, 3)
    kv.deallocate(a)  # [1,1] cached; the partial block went back to empty
    old = a.hashes[0]
    assert old in kv.prefix_map
    b = make_seq([2, 2, 3, 3, 9])  # needs 3 blocks > 2: cannot be admitted
    assert kv.can_allocate(b) is None
    c = make_seq([2, 2, 9])
    kv.allocate(c, kv.can_allocate(c))  # takes the empty block and overwrites [1,1]
    assert old not in kv.prefix_map
    assert kv.match(make_seq([1, 1, 9])) == []
    kv.check()


def test_same_tokens_different_parent_is_a_different_prefix():
    x = chain_hashes(array("I", [5, 5, 7, 7]), BS)
    y = chain_hashes(array("I", [6, 6, 7, 7]), BS)
    assert x[1] != y[1]


def test_partial_blocks_are_never_shared():
    Seq.block_size = BS
    kv = KVCacheManager(8, BS, choose_victim=lambda c: next(iter(c)))
    a = make_seq([1, 2, 3])
    kv.allocate(a, kv.can_allocate(a))
    kv.register(a, 0, 3)
    assert len(kv.prefix_map) == 1  # only the full block [1, 2]
    b = make_seq([1, 2, 3, 4])
    assert len(kv.match(b)) == 1


def test_active_blocks_are_never_victims_and_engine_waits():
    Seq.block_size = BS
    asked = []
    kv = KVCacheManager(
        4, BS, choose_victim=lambda c: asked.append(list(c)) or next(iter(c))
    )
    sched = Scheduler(kv, max_num_seqs=8, max_num_batched_tokens=64)
    a = make_seq([1, 1, 2, 2, 3], max_tokens=1 + BS)  # 3 blocks now, a 4th later
    sched.add(a)
    seqs, prefill = sched.schedule()
    assert seqs == [a] and prefill
    sched.postprocess(seqs, True)
    b = make_seq([4, 4, 5])  # needs 2 blocks; only 1 is free
    sched.add(b)
    seqs, prefill = sched.schedule()  # b waits; a decodes
    assert seqs == [a] and not prefill
    sched.postprocess(seqs, False)
    seqs, prefill = sched.schedule()  # a takes the last free block for its next token
    assert seqs == [a] and not prefill and kv.available() == 0
    assert asked == []  # nothing was ever evictable, so no victim was requested


def test_colliding_hashes_never_reuse_a_block_at_another_position():
    """Two identical token blocks at different positions whose hashes collide
    must not map to the same physical block (found by a randomized run)."""
    Seq.block_size = BS
    kv = KVCacheManager(
        6, BS, choose_victim=lambda cached: next(iter(cached)), verify=True
    )
    first = make_seq([1, 0, 1, 0, 0], weak_hash=True)
    kv.allocate(first, kv.can_allocate(first))
    kv.register(first, 0, first.num_tokens)
    again = make_seq([1, 0, 1, 0, 0], weak_hash=True)
    hits = kv.can_allocate(again)
    kv.allocate(again, hits)
    assert len(set(again.block_table)) == len(again.block_table)
    for i, bid in enumerate(hits):
        assert kv.blocks[bid].tokens == again.block(i)
