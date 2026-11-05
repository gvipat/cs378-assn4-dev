"""Run the shared scheduler and KV manager using estimated step times on the CPU.

Decode steps are batched into runs. Between two observable moments (a block
filling up, a sequence finishing, or an external event such as an arrival or
a tool finishing) the batch is fixed and nothing reaches the policy, so k
steps are applied at once with the closed-form time of `TimingModel.decode_run`.
The result is identical to stepping one token at a time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..trace import EngineConfig, Memory
from .kv import KVCacheManager
from .scheduler import Scheduler
from .seq import Seq
from .timing import TimingModel


@dataclass
class Step:
    seqs: list[Seq]
    is_prefill: bool
    start: float
    duration: float
    k: int = 1  # decode steps in this run

    @property
    def end(self) -> float:
        return self.start + self.duration


class SimEngine:

    def __init__(
        self,
        memory: Memory,
        config: EngineConfig,
        *,
        timing: TimingModel,
        choose_victim,
        emit_block=None,
        on_call=None,
        hooks=None,
        verify: bool = False,
    ):
        Seq.block_size = memory.block_size
        self.timing = timing
        self.kv = KVCacheManager(
            memory.num_blocks,
            memory.block_size,
            choose_victim=choose_victim,
            emit=emit_block,
            verify=verify,
        )
        self.sched = Scheduler(
            self.kv,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            on_call=on_call,
            hooks=hooks,
        )
        self.prefill_steps = 0
        self.decode_steps = 0
        self.prefill_tokens = 0
        self.busy_time = 0.0
        self.prefill_time = 0.0
        self.macro = (
            True  # False steps decode one token at a time (tests compare the two)
        )

    def add(self, seq: Seq) -> None:
        self.sched.add(seq)

    def has_work(self) -> bool:
        return not self.sched.is_finished()

    def begin_step(self, now: float, horizon: float = math.inf) -> Step:
        """Schedule at `now` and return the step (or decode run) without applying
        its results. `horizon` is the time of the next external event."""
        seqs, is_prefill = self.sched.schedule()
        if is_prefill:
            chunks = [(s.num_cached_tokens, s.num_scheduled_tokens) for s in seqs]
            self.prefill_tokens += sum(n for _, n in chunks)
            return Step(seqs, True, now, self.timing.prefill(chunks))
        bs = self.kv.block_size
        k_max = 1 << 62
        for s in seqs:
            to_block = (-s.num_cached_tokens) % bs or bs
            to_finish = s.max_tokens - s.num_completion_tokens
            k_max = min(k_max, to_block, to_finish)
        batch, context = len(seqs), sum(s.num_tokens for s in seqs)
        waiting = self.sched.waiting
        prefill_next = waiting and (
            waiting[0].block_table or self.kv.can_allocate(waiting[0]) is not None
        )
        if not self.macro or prefill_next:
            # A preemption while scheduling this step freed blocks, so prefill
            # takes over at the next step: no run.
            k_max = 1
        k = k_max
        if horizon - now < self.timing.decode_run(batch, context, k_max):
            lo, hi = 1, k_max  # smallest k whose run reaches the horizon
            while lo < hi:
                mid = (lo + hi) // 2
                if now + self.timing.decode_run(batch, context, mid) >= horizon:
                    hi = mid
                else:
                    lo = mid + 1
            k = lo
        return Step(seqs, False, now, self.timing.decode_run(batch, context, k), k)

    def end_step(self, step: Step) -> list[Seq]:
        if step.k > 1:
            for (
                s
            ) in step.seqs:  # the first k-1 steps of a run change nothing observable
                s.num_cached_tokens += step.k - 1
                s.num_tokens += step.k - 1
        if step.is_prefill:
            self.prefill_steps += 1
            self.prefill_time += step.duration
        else:
            self.decode_steps += step.k
        self.busy_time += step.duration
        return self.sched.postprocess(step.seqs, step.is_prefill)
