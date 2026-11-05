"""The engine scheduler, shared by the CPU simulator and the GPU replay.

It follows nano-vLLM's Scheduler at the pinned commit: only the first sequence
of a prefill batch may be chunked, prefill and decode run in separate steps,
and a preempted sequence is recomputed later. Three of its choices are
delegated to the policy through `hooks`, and default to nano-vLLM's rules:

  admission order   which waiting sequences to admit first (default: FIFO)
  prefill first     whether a step prefills when it could also decode (default: yes)
  preemption        which running sequence to preempt when decode runs out of
                    blocks (default: the most recently scheduled)

The one addition is a liveness guard: if nothing can be scheduled while
sequences part-way through a chunked prefill hold blocks, those sequences are
preempted so that the head of the queue can be admitted.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

from .kv import EngineError, KVCacheManager
from .seq import Seq, SeqStatus

# on_call(kind, seq) with kind in admit / preempt
OnCall = Callable[[str, Seq], None]


class SchedulingHooks:
    """Control admission order and preemption.

    Each hook is called only when the scheduler has a choice to make.
    The default implementations follow nano-vLLM's rules.
    """

    def admission_order(self, waiting: list[Seq]) -> list[Seq]:
        return waiting

    def choose_preempted(self, running: list[Seq]) -> Seq:
        return running[-1]


class Scheduler:

    def __init__(
        self,
        kv: KVCacheManager,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        on_call: OnCall | None = None,
        hooks: SchedulingHooks | None = None,
    ):
        self.kv = kv
        self.hooks = hooks or SchedulingHooks()
        self.block_size = kv.block_size
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.on_call = on_call or (lambda kind, seq: None)
        self.waiting: deque[Seq] = deque()
        self.running: deque[Seq] = deque()
        self.preemptions = 0

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Seq) -> None:
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Seq], bool]:
        seqs, is_prefill = self._schedule()
        if not seqs and self.waiting and not self.running:
            # Liveness guard: release blocks held by stalled chunked prefills.
            for seq in list(self.waiting):
                if seq.block_table:
                    self._release_waiting(seq)
            seqs, is_prefill = self._schedule()
        if not seqs and not self.is_finished():
            raise EngineError("engine stalled: a prompt does not fit in memory")
        return seqs, is_prefill

    def _order_waiting(self) -> None:
        """Let the policy order the waiting queue. A sequence part-way through a
        chunked prefill (it holds blocks) stays at the front."""
        started = [s for s in self.waiting if s.block_table]
        fresh = [s for s in self.waiting if not s.block_table]
        if len(fresh) > 1:
            order = self.hooks.admission_order(fresh)
            if sorted(map(id, order)) != sorted(map(id, fresh)):
                raise EngineError(
                    "admission_order did not return a permutation of the waiting calls"
                )
            fresh = order
        self.waiting = deque(started + fresh)

    def _schedule(self) -> tuple[list[Seq], bool]:
        bs = self.block_size
        scheduled: list[Seq] = []
        num_batched = 0
        if self.waiting:
            self._order_waiting()

        # prefill
        while self.waiting and len(scheduled) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched
            if remaining == 0:
                break
            hits = None
            if not seq.block_table:
                hits = self.kv.can_allocate(seq)
                if hits is None:
                    break
                num_tokens = seq.num_tokens - len(hits) * bs
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if (
                remaining < num_tokens and scheduled
            ):  # only the first sequence may be chunked
                break
            if hits is not None:
                self.kv.allocate(seq, hits)
                seq.admissions += 1
                if seq.first_cached_tokens is None:
                    seq.first_cached_tokens = seq.num_cached_tokens
                self.on_call("admit", seq)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SeqStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled.append(seq)

        if scheduled:
            return scheduled, True

        # decode
        while self.running and len(scheduled) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.kv.can_append(seq):
                # Candidates in scheduling order: the one being scheduled now came
                # first, then the rest (nano-vLLM preempts the last one).
                victim = (
                    self.hooks.choose_preempted([seq] + list(self.running))
                    if self.running
                    else seq
                )
                if victim is seq:
                    self.preempt(seq)
                    break
                if victim not in self.running:
                    raise EngineError(
                        "choose_preempted returned a call that is not running"
                    )
                self.running.remove(victim)
                self.preempt(victim)
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.kv.may_append(seq)
                scheduled.append(seq)
        self.running.extendleft(reversed(scheduled))
        return scheduled, False

    def preempt(self, seq: Seq) -> None:
        seq.status = SeqStatus.WAITING
        seq.is_prefill = True
        seq.preemptions += 1
        self.preemptions += 1
        self.kv.deallocate(seq)
        self.waiting.appendleft(seq)
        self.on_call("preempt", seq)

    def _release_waiting(self, seq: Seq) -> None:
        seq.preemptions += 1
        self.preemptions += 1
        self.kv.deallocate(seq)
        self.on_call("preempt", seq)

    def postprocess(self, seqs: list[Seq], is_prefill: bool) -> list[Seq]:
        """Apply one completed step. Returns the sequences that finished."""
        finished = []
        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            self.kv.register(seq, start, end)
            seq.num_cached_tokens = end
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token()
            if seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SeqStatus.FINISHED
                self.kv.deallocate(seq)
                self.running.remove(seq)
                finished.append(seq)
        return finished
