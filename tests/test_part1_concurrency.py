"""Part 1: grant slots only below the cap and let running calls finish when
the cap is lowered. These tests need only research_app/concurrency.py;
the agent's use of the cap is tested in test_part1_agent.py."""

import asyncio
import random

import pytest

from research_app.concurrency import ConcurrencyLimiter


def test_cap_never_exceeded_under_random_load_and_limit_changes():
    async def go():
        cap = ConcurrencyLimiter(3)
        rng = random.Random(7)
        held = 0
        grants_over = []

        async def worker(i):
            nonlocal held
            await asyncio.sleep(rng.random() * 0.01)
            async with cap.slot():
                if held >= cap.limit:
                    grants_over.append((i, held, cap.limit))
                held += 1
                await asyncio.sleep(rng.random() * 0.01)
                held -= 1

        async def changer():
            for limit in [1, 5, 2, 8, 3]:
                await asyncio.sleep(0.005)
                await cap.set_limit(limit)

        await asyncio.gather(changer(), *(worker(i) for i in range(60)))
        return cap, grants_over

    cap, over = asyncio.run(go())
    assert over == []
    assert cap.over_limit == 0
    assert cap.in_flight == 0
    assert cap.grants == 60


def test_lowering_cap_drains_without_cancelling():
    async def go():
        cap = ConcurrencyLimiter(4)
        log = []

        async def call(name, dur):
            async with cap.slot():
                log.append(("start", name, asyncio.get_running_loop().time()))
                await asyncio.sleep(dur)
                log.append(("end", name, asyncio.get_running_loop().time()))

        first = [asyncio.create_task(call(f"a{i}", 0.05)) for i in range(4)]
        await asyncio.sleep(0.01)
        await cap.set_limit(1)
        later = [asyncio.create_task(call(f"b{i}", 0.01)) for i in range(3)]
        await asyncio.gather(*first, *later)
        return first, log

    first, log = asyncio.run(go())
    assert all(not t.cancelled() for t in first)
    ends_a = max(t for kind, name, t in log if kind == "end" and name.startswith("a"))
    starts_b = [t for kind, name, t in log if kind == "start" and name.startswith("b")]
    assert all(t >= ends_a for t in starts_b)  # waited for the drain
    # under the new cap of 1 the b calls never overlap
    events = [(kind, name) for kind, name, _ in log if name.startswith("b")]
    depth = 0
    for kind, _ in events:
        depth += 1 if kind == "start" else -1
        assert depth <= 1


def test_cap_rejects_out_of_range():
    with pytest.raises(ValueError):
        ConcurrencyLimiter(0)
    with pytest.raises(ValueError):
        ConcurrencyLimiter(9)
