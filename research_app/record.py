"""Record research runs without waiting for simulated delays.

Used by trace generation and trace-export tests. FakeLLM responds immediately;
tool delays are sampled from (seed, operation id, attempt) and recorded rather
than slept. Repeated inputs produce the same recordings regardless of task
scheduling.

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
