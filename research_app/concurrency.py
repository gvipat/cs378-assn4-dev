"""Limit the number of LLM calls running at once across all reports (Part 1).

One ConcurrencyLimiter is shared by every report in a run, so C bounds LLM calls
across the whole run. The cap must satisfy the following rules:

- `async with cap.slot(): ...` holds one slot for the whole operation,
  including its retries; a retry never takes a second slot.
- A new slot is granted only while fewer than `limit` slots are held.
- `await cap.set_limit(n)` takes effect immediately for new acquisitions.
  Lowering it never cancels or interrupts calls already in flight; they
  finish and release, and waiting calls proceed once in_flight < limit.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

MIN_CAP, MAX_CAP = 1, 8


class ConcurrencyLimiter:
    def __init__(self, limit: int):
        if not MIN_CAP <= limit <= MAX_CAP:
            raise ValueError(f"cap must be in {MIN_CAP}..{MAX_CAP}")
        self._limit = limit
        self._in_flight = 0
        self._cond = asyncio.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def acquire(self) -> None:
        """Wait for an available slot and reserve it."""
        # [Part 1] TODO: Reserve one slot for an LLM operation.
        # Several reports may call this method at once. Wait until _in_flight is
        # below _limit, then update the count while still holding the condition
        # lock so another caller cannot take the same slot. A cancelled waiter
        # must leave _in_flight unchanged.
        raise NotImplementedError("Part 1: fill acquire")
        # END OF YOUR CODE

    async def release(self) -> None:
        """Return a slot and notify waiting calls."""
        # [Part 1] TODO: Release a slot held by a completed operation.
        # Update the count while holding the condition lock, then wake waiting
        # calls so they can check whether a slot is available. Releasing when
        # no slot is held is a programming error: raise RuntimeError rather
        # than allowing the count to become negative.
        raise NotImplementedError("Part 1: fill release")
        # END OF YOUR CODE

    async def set_limit(self, limit: int) -> None:
        """Change the cap without interrupting running calls."""
        if not MIN_CAP <= limit <= MAX_CAP:
            raise ValueError(f"cap must be in {MIN_CAP}..{MAX_CAP}")
        # [Part 1] TODO: Apply the new limit to future acquisitions.
        # The range check above is provided. Change _limit while holding the
        # condition lock and wake waiting calls, including when the limit grows
        # and no running call has released a slot. If the limit shrinks below
        # _in_flight, let those calls finish; do not cancel them or change the
        # count to make it match the new limit.
        raise NotImplementedError("Part 1: fill set_limit")
        # END OF YOUR CODE

    @asynccontextmanager
    async def slot(self):
        await self.acquire()
        try:
            yield
        finally:
            await self.release()
