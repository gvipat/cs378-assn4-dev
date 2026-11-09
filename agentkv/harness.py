"""A small KV-cache harness for testing a policy's eviction decisions.

The harness drives a block-level cache with one-token blocks named by strings
and emits allocation, hit, release, and eviction events on replica 0. A block's
name is both its residence id and its prefix hash. After completing the LRU
policy:

    from agentkv.harness import CacheHarness
    from policies.reference import ReferencePolicy

    h = CacheHarness(ReferencePolicy(), capacity=3)
    h.allocate("A"); h.allocate("B"); h.release("A"); h.release("B")
    assert h.cached() == ["A", "B"] and h.empty_slots() == 1
"""

from __future__ import annotations

import hashlib

from .engine.kv import KVCacheManager
from .engine.seq import Seq
from .policy import Policy
from .sandbox.runtime import PolicyError, PolicyRuntime

CALL = "c0"


def _h(name: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(name.encode(), digest_size=16).digest(), "little"
    )


class CacheHarness:

    def __init__(self, policy: Policy, capacity: int = 3):
        self.rt = PolicyRuntime(policy, block_size=1, total_blocks=capacity, seed=0)
        self.kv = KVCacheManager(
            capacity, 1, choose_victim=self._victim, emit=self._emit
        )
        self.names: dict[int, str] = {}
        self.seqs: dict[str, list[Seq]] = {}
        self.parents: dict[str, str | None] = {}
        self.evicted: list[str] = []
        self.t = 0.0
        self._events: list = []

    # ------------------------------------------------------------ operations

    def allocate(self, name: str, parent: str | None = None) -> None:
        """Compute and hold a new block named name, extending parent when one is provided."""
        h = _h(name)
        if h in self.kv.prefix_map:
            raise ValueError(f"{name} is already resident; use access()")
        self.names[h] = name
        self.parents[name] = parent
        self._tick()
        seq = self._seq(h, parent)
        seq.block_table.append(self.kv._take())
        self._flush()
        self.kv.register(seq, 0, 1)
        self.seqs.setdefault(name, []).append(seq)
        self._flush()

    def access(self, name: str) -> bool:
        """Look up the named prefix block, hold a reference if it matches, and return whether it hit."""
        h = _h(name)
        self._tick()
        bid = self.kv.prefix_map.get(h)
        if bid is None:
            return False
        seq = self._seq(h, self.parents.get(name))
        self.kv.allocate(seq, [bid])
        self.seqs.setdefault(name, []).append(seq)
        self._flush()
        return True

    def release(self, name: str) -> None:
        """Drop one reference to `name`."""
        self._tick()
        self.kv.deallocate(self.seqs[name].pop())
        self._flush()

    # ------------------------------------------------------------ queries

    def cached(self) -> list[str]:
        """Return resident blocks with no active references, ordered by release time."""
        return [self.names[self.kv.blocks[b].hash] for b in self.kv.cached]

    def active(self) -> list[str]:
        return sorted(
            self.names[b.hash]
            for b in self.kv.blocks
            if b.ref > 0 and b.hash is not None
        )

    def resident(self) -> set[str]:
        return {self.names[h] for h in self.kv.prefix_map}

    def empty_slots(self) -> int:
        return len(self.kv.empty)

    # ------------------------------------------------------------ engine callbacks

    def _seq(self, h: int, parent: str | None) -> Seq:
        Seq.block_size = 1
        return Seq(1, 1, (h,), owner=self)

    def _depth(self, name: str) -> int:
        depth, parent = 0, self.parents.get(name)
        while parent is not None:
            depth, parent = depth + 1, self.parents.get(parent)
        return depth

    def _tick(self) -> None:
        self.t += 1.0

    def _emit(self, kind: str, h: int, seq, **kw) -> None:
        name = self.names.get(h)
        if kind == "hit":
            self._events.append(("hit", self.t, name, CALL))
        elif kind == "alloc":
            parent = self.parents.get(name)
            self._events.append(
                ("alloc", self.t, name, parent, self._depth(name), CALL, name, 0)
            )
        elif kind == "release":
            self._events.append(("release", self.t, name, CALL, kw["refs"]))
        elif kind == "evict":
            self.evicted.append(name)
            self._events.append(("evict", self.t, name))

    def _flush(self) -> None:
        ev, self._events = self._events, []
        self.rt.apply(ev)

    def _victim(self, cached: dict) -> int:
        self._flush()
        hdr = [self.t, len(self.kv.empty), self.kv.num_active, 0, 0]
        pid = self.rt.choose_victim(hdr)
        bid = self.kv.prefix_map.get(_h(pid)) if isinstance(pid, str) else None
        if bid is None or bid not in cached:
            raise PolicyError(
                "invalid", f"choose_victim returned {pid!r}, which is not a candidate"
            )
        return bid
