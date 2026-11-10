"""The teaching cache uses the evaluator's block identity and event contracts."""

from agentkv.harness import CacheHarness
from agentkv.policy import EVENT_FIELDS, Policy


class RecordingFIFO(Policy):
    def __init__(self):
        super().__init__()
        self.events = []

    def on_event(self, event, state):
        assert set(event._f) == set(EVENT_FIELDS[event.kind])
        self.events.append(event)

    def choose_victim(self, view, candidates, state):
        assert all(view.blocks[b].ref_cnt == 0 for b in candidates)
        return candidates[0]


def test_children_reference_prefix_hashes_even_after_parent_eviction():
    policy = RecordingFIFO()
    h = CacheHarness(policy, capacity=2)
    h.allocate("A")
    h.allocate("B", parent="A")
    root, child = [e for e in policy.events if e.kind == "alloc"]
    assert root.block != root.block_hash
    assert child.block != child.block_hash
    assert root.parent_hash is None and root.depth == 0
    assert child.parent_hash == root.block_hash and child.depth == 1
    assert root.replica == child.replica == 0

    h.release("A")
    h.allocate("C", parent="A")
    assert h.evicted == ["A"]
    newest = policy.events[-1]
    assert newest.parent_hash == root.block_hash
    assert root.block not in h.rt.blocks
    assert h.rt.blocks[child.block].parent_hash == root.block_hash
    assert h.rt.blocks[newest.block].block_hash == newest.block_hash
    assert h.resident() == {"B", "C"}
    h.kv.check()


def test_recomputing_a_prefix_changes_its_residence_id_but_not_its_hash():
    policy = RecordingFIFO()
    h = CacheHarness(policy, capacity=1)
    h.allocate("A")
    original = policy.events[-1]
    assert h.access("A")
    assert policy.events[-1].block == original.block
    h.release("A")
    h.release("A")
    h.allocate("B")
    h.release("B")
    h.allocate("A")
    recomputed = policy.events[-1]
    assert recomputed.block != original.block
    assert recomputed.block_hash == original.block_hash
    assert h.evicted == ["A", "B"]
    assert set(h.rt.blocks) == {recomputed.block}
    assert set(h.rt.blocks).isdisjoint(b.block_hash for b in h.rt.blocks.values())
    h.kv.check()
