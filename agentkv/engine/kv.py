"""Manage KV blocks in the legacy single-replica simulator.

Blocks are empty, active (referenced by a sequence), or cached (unreferenced
but reusable). Full computed blocks receive chained prefix hashes; partial
blocks are never shared. If a hash already has a resident block, a duplicate
stays anonymous and becomes empty when released.

Allocation uses empty slots first, then asks choose_victim for a cached block
to overwrite. The implementation follows nano-vLLM's block manager, with
separate empty and cached queues. Assignment scoring uses the Vidur cache
adapter in engine.vidur_adapter.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

from .seq import Seq


class EngineError(RuntimeError):
    """Report an internal engine invariant failure rather than an invalid policy action."""


class KVBlock:
    __slots__ = ("id", "ref", "hash", "tokens", "gen", "parent")

    def __init__(self, block_id: int):
        self.id = block_id
        self.ref = 0
        self.hash: int | None = None
        self.tokens: tuple | None = None
        self.gen = 0  # bumped each time the block gets a new identity
        self.parent: tuple | None = (
            None  # (id, gen) of the block before it (verify mode)
        )


# emit(kind, block_hash, seq, **fields) with kind in hit / alloc / release / evict
Emit = Callable[..., None]
# choose(cached) -> block id; `cached` maps block id -> None, oldest-released first
Choose = Callable[[dict], int]


class KVCacheManager:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        *,
        choose_victim: Choose,
        emit: Emit | None = None,
        verify: bool = False,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [KVBlock(i) for i in range(num_blocks)]
        self.prefix_map: dict[int, int] = {}
        self.empty: deque[int] = deque(range(num_blocks))
        self.cached: dict[int, None] = {}
        self.num_active = 0
        self.choose_victim = choose_victim
        self.emit = emit or (lambda *a, **k: None)
        self.verify = verify
        # diagnostics
        self.seen: set[int] = set()
        self.evictions = 0
        self.recomputed_blocks = 0
        self.duplicate_blocks = 0

    # ------------------------------------------------------------ queries

    def available(self) -> int:
        return len(self.empty) + len(self.cached)

    def match(self, seq: Seq) -> list[int]:
        """Return physical block ids for the longest resident prefix of seq.

        Following nano-vLLM, always recompute the block containing the last token.
        """
        out: list[int] = []
        limit = min((seq.num_tokens - 1) // self.block_size, len(seq.hashes))
        for i in range(limit):
            bid = self.prefix_map.get(seq.hashes[i])
            if bid is None:
                break
            if self.verify:
                # A hash collision is a miss: the block must hold these tokens and
                # follow the block matched just before it, as it did when computed.
                blk = self.blocks[bid]
                before = (out[-1], self.blocks[out[-1]].gen) if out else None
                if blk.tokens != seq.block(i) or blk.parent != before:
                    break
            out.append(bid)
        return out

    def can_allocate(self, seq: Seq) -> list[int] | None:
        hits = self.match(seq)
        need = seq.num_blocks - sum(1 for b in hits if self.blocks[b].ref > 0)
        return hits if self.available() >= need else None

    def needs_block(self, seq: Seq) -> bool:
        return seq.num_tokens > len(seq.block_table) * self.block_size

    def can_append(self, seq: Seq) -> bool:
        return not self.needs_block(seq) or self.available() >= 1

    # ------------------------------------------------------------ mutations

    def allocate(self, seq: Seq, hits: list[int]) -> None:
        if seq.block_table:
            raise EngineError("allocate on a sequence that already has blocks")
        for bid in hits:
            blk = self.blocks[bid]
            if blk.ref == 0:
                del self.cached[bid]
                self.num_active += 1
            blk.ref += 1
            seq.block_table.append(bid)
            self.emit("hit", blk.hash, seq)
        for _ in range(len(hits), seq.num_blocks):
            seq.block_table.append(self._take())
        seq.num_cached_tokens = len(hits) * self.block_size

    def may_append(self, seq: Seq) -> None:
        if self.needs_block(seq):
            seq.block_table.append(self._take())

    def deallocate(self, seq: Seq) -> None:
        for bid in reversed(seq.block_table):
            blk = self.blocks[bid]
            blk.ref -= 1
            if blk.ref < 0:
                raise EngineError(f"negative reference count on block {bid}")
            if blk.ref == 0:
                self.num_active -= 1
                if blk.hash is not None:
                    self.cached[bid] = None
                else:
                    blk.tokens = None
                    self.empty.append(bid)
            if blk.hash is not None:
                self.emit("release", blk.hash, seq, refs=blk.ref)
        seq.block_table.clear()
        seq.num_cached_tokens = 0

    def register(self, seq: Seq, start: int, end: int) -> None:
        """Register full blocks after computing tokens [start, end) of seq.

        Assign each newly completed block its chained prefix hash, following
        nano-vLLM's hash_blocks behavior.
        """
        bs = self.block_size
        for i in range(start // bs, end // bs):
            h = seq.hashes[i]
            if h in self.seen:
                self.recomputed_blocks += 1
            else:
                self.seen.add(h)
            blk = self.blocks[seq.block_table[i]]
            if blk.hash is not None:
                continue
            if h in self.prefix_map:
                self.duplicate_blocks += (
                    1  # same contents already resident; stay anonymous
                )
                continue
            blk.hash = h
            blk.gen += 1
            if self.verify:
                blk.tokens = seq.block(i)
                prev = seq.block_table[i - 1] if i else None
                blk.parent = (prev, self.blocks[prev].gen) if i else None
            self.prefix_map[h] = blk.id
            self.emit("alloc", h, seq, depth=i, parent=seq.hashes[i - 1] if i else None)

    def _take(self) -> int:
        if self.empty:
            bid = self.empty.popleft()
        else:
            if not self.cached:
                raise EngineError(
                    "no block available; the scheduler should have checked"
                )
            bid = self.choose_victim(self.cached)
            if bid not in self.cached:
                raise EngineError(f"victim {bid} is not a zero-reference cached block")
            blk = self.blocks[bid]
            del self.cached[bid]
            if self.prefix_map.get(blk.hash) != bid:
                raise EngineError("prefix map out of sync with cached block")
            del self.prefix_map[blk.hash]
            self.evictions += 1
            self.emit("evict", blk.hash, None)
            blk.hash = None
            blk.tokens = None
        blk = self.blocks[bid]
        if blk.ref != 0:
            raise EngineError(f"block {bid} taken with ref {blk.ref}")
        blk.ref = 1
        self.num_active += 1
        return bid

    # ------------------------------------------------------------ checks

    def check(self) -> None:
        """Check all cache invariants; used by tests and debug runs."""
        empty, cached = set(self.empty), set(self.cached)
        if len(empty) != len(self.empty) or empty & cached:
            raise EngineError("free lists overlap or repeat")
        active = 0
        for blk in self.blocks:
            if blk.ref < 0:
                raise EngineError("negative ref")
            if blk.ref > 0:
                active += 1
                if blk.id in empty or blk.id in cached:
                    raise EngineError("active block on a free list")
            elif blk.id not in empty and blk.id not in cached:
                raise EngineError(f"zero-ref block {blk.id} on no list")
            if blk.id in cached and blk.hash is None:
                raise EngineError("cached block without identity")
            if blk.id in empty and blk.hash is not None:
                raise EngineError("empty block with identity")
            if blk.hash is not None and self.prefix_map.get(blk.hash) != blk.id:
                raise EngineError("identity not in prefix map")
        if active != self.num_active:
            raise EngineError("active count drifted")
        for h, bid in self.prefix_map.items():
            if self.blocks[bid].hash != h:
                raise EngineError("stale prefix map entry")
