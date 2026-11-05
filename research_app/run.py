"""Run the deep research agent on a set of questions (Part 1).

    python -m research_app.run --cap 2 --replay --out runs/c2     # 16 public questions, FakeLLM
    python -m research_app.run --cap 8 --replay --out runs/c8
    python -m research_app.run --cap 8 --failure-rate 0.2 --replay --out runs/c8-faults
    AGENTKV_BEDROCK_MODEL=... python -m research_app.run --backend bedrock --cap 2 --replay --out runs/c2

Reports start `--stagger` seconds apart. Writes to --out:
    trace.jsonl    every LLM and tool call (the Part 1 trace)
    reports.json   the reports, their evidence, and validation status
    summary.json   per-report latency and counts, peak concurrency
    replay.json    the same run as an agentkv replay trace (with --replay)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from agentkv.trace import TraceBuilder

from .agent import AgentSettings, run_reports
from .concurrency import ConcurrencyLimiter
from .corpus import ToolLatency
from .llm import FakeLatency, make_llm
from .questions import load_questions
from .tracing import to_trace_job


def select(spec: str) -> list[dict]:
    if spec == "public":
        return load_questions(public=True)
    if spec == "all":
        return load_questions()
    ids = spec.split(",")
    by_id = {q["id"]: q for q in load_questions()}
    return [by_id[i] for i in ids]


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--questions",
        default="public",
        help="public | all | comma-separated question ids",
    )
    ap.add_argument(
        "--cap", type=int, default=2, help="global cap C on in-flight LLM calls (1-8)"
    )
    ap.add_argument("--backend", default="fake", choices=["fake", "bedrock"])
    ap.add_argument(
        "--stagger", type=float, default=2.0, help="seconds between report arrivals"
    )
    ap.add_argument("--snapshot", default="v2", choices=["v1", "v2"])
    ap.add_argument("--seed", type=int, default=0, help="tool latency seed")
    ap.add_argument(
        "--latency-scale",
        type=float,
        default=1.0,
        help="scale simulated tool and FakeLLM latency",
    )
    ap.add_argument(
        "--failure-rate",
        type=float,
        default=0.0,
        help="probability that a tool attempt fails transiently",
    )
    ap.add_argument(
        "--record", action="store_true", help="do not sleep; only record durations"
    )
    ap.add_argument(
        "--replay", action="store_true", help="also write replay.json (agentkv trace)"
    )
    ap.add_argument(
        "--num-blocks", type=int, default=192, help="KV blocks for replay.json"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    questions = select(args.questions)
    arrivals = [round(i * args.stagger, 3) for i in range(len(questions))]
    kw = (
        {"latency": FakeLatency(scale=args.latency_scale)}
        if args.backend == "fake" and not args.record
        else {}
    )
    llm = make_llm(args.backend, **kw)

    latency = ToolLatency(
        seed=args.seed, scale=args.latency_scale, failure_rate=args.failure_rate
    )

    async def go():
        return await run_reports(
            questions,
            llm=llm,
            cap=ConcurrencyLimiter(args.cap),
            arrivals=arrivals,
            snapshot=args.snapshot,
            live=not args.record,
            latency=latency,
            settings=AgentSettings(),
            backend=args.backend,
        )

    recorder, recs = asyncio.run(go())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recorder.meta.update(
        {
            "questions": args.questions,
            "stagger": args.stagger,
            "snapshot": args.snapshot,
            "failure_rate": args.failure_rate,
        }
    )
    recorder.export_jsonl(out / "trace.jsonl")
    (out / "reports.json").write_text(
        json.dumps(
            [
                {
                    "report_id": r.report_id,
                    "question_id": r.question_id,
                    "valid": r.valid,
                    "errors": r.errors,
                    "branches": r.branches,
                    "report": r.report,
                }
                for r in recs
            ],
            indent=1,
            ensure_ascii=False,
        )
    )
    rows = [
        {
            "report_id": r.report_id,
            "question_id": r.question_id,
            "arrival": r.arrival,
            "latency": round(r.end - r.arrival, 3),
            "valid": r.valid,
            **r.counts(),
            "retries": sum(o.retries for o in r.ops),
        }
        for r in recs
    ]
    summary = {
        "cap": args.cap,
        "backend": args.backend,
        "reports": len(recs),
        "valid": sum(bool(r.valid) for r in recs),
        "mean_latency": round(statistics.mean(x["latency"] for x in rows), 3),
        "makespan": round(max(r.end for r in recs), 3),
        "llm_peak": recorder.llm_peak,
        "cap_violations": recorder.cap_violations,
        "reports_detail": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if args.replay:
        b = TraceBuilder(
            f"part1-c{args.cap}",
            "P1",
            "part1",
            num_blocks=args.num_blocks,
            description=f"Part 1 run, {len(recs)} reports, C={args.cap}, backend={args.backend}",
        )
        for r in recs:
            to_trace_job(r, b.job(r.arrival, meta={"question": r.question_id}))
        b.save(out / "replay.json")
    print(
        f"C={args.cap}: {summary['valid']}/{summary['reports']} valid, mean report latency "
        f"{summary['mean_latency']}s, makespan {summary['makespan']}s, peak in-flight LLM calls "
        f"{summary['llm_peak']}, cap violations {summary['cap_violations']} -> {out}"
    )
    return summary


if __name__ == "__main__":
    main()
