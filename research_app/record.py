"""Record the agent workflow without waiting for simulated delays (Part 1).

Staff build the D-family traces this way from the reference agent; the Part 1
tests use it to check your agent's trace export: `record_reports` and then
`to_trace_job` for each recording. Record mode never sleeps: the FakeLLM
answers instantly and tool latencies are sampled and recorded, keyed by
(seed, op id, attempt), so the same inputs always give the same recordings
regardless of scheduling.

    from research_app.record import record_reports
    from research_app.tracing import to_trace_job
    recs = record_reports(questions, snapshot="v2", seed=7)
    for rec, arrival in zip(recs, arrivals):
        to_trace_job(rec, builder.job(arrival, meta={"question": rec.question_id}))
"""

from __future__ import annotations

import asyncio

from .agent import AgentSettings, run_reports
from .concurrency import ConcurrencyLimiter
from .corpus import FaultFn, ToolLatency
from .llm import FakeLLM
from .tracing import Recording


def record_reports(
    questions: list[dict],
    *,
    snapshot: str = "v2",
    seed: int = 0,
    cap: int = 8,
    latency: ToolLatency | None = None,
    failure_rate: float = 0.0,
    bad_citation_rate: float = 0.2,
    faults: FaultFn | None = None,
    settings: AgentSettings = AgentSettings(),
) -> list[Recording]:
    """Return one recording per question in input order, with ids such as R00 and R01.

    `latency` defaults to ToolLatency(seed=seed, failure_rate=failure_rate);
    pass one to change the delays or the frequency of slow tool calls. Transient tool
    failures add retried attempts (and their delays) to the recording.
    """
    latency = latency or ToolLatency(seed=seed, failure_rate=failure_rate)
    llm = FakeLLM(bad_citation_rate=bad_citation_rate)

    async def go() -> list[Recording]:
        _, recs = await run_reports(
            questions,
            llm=llm,
            cap=ConcurrencyLimiter(cap),
            snapshot=snapshot,
            live=False,
            latency=latency,
            faults=faults,
            settings=settings,
            backend="fake-record",
        )
        return recs

    return asyncio.run(go())
