"""Part 1: the JSONL trace and the conversion to an agentkv replay trace."""

import json

from agentkv import trace as T
from research_app.questions import load_questions
from research_app.record import record_reports
from research_app.tracing import BLOCK_SIZE, to_trace_job


def build(recs, path, num_blocks=192):
    b = T.TraceBuilder("t-part1", "P1", "test", num_blocks=num_blocks)
    maps = [
        to_trace_job(r, b.job(2.0 * i, meta={"question": r.question_id}))
        for i, r in enumerate(recs)
    ]
    d = b.save(path)
    return d, maps


def test_exported_jobs_load_and_match_the_recording(tmp_path):
    recs = record_reports(load_questions(public=True)[:5], seed=3)
    path = tmp_path / "replay.json"
    d, maps = build(recs, path)
    tr = T.load(path)
    assert len(tr.jobs) == 5
    for rec, job, ops_to_nodes in zip(recs, tr.jobs, maps):
        nodes = {n.id: n for n in job.nodes}
        for op in rec.ops:
            if op.kind == "llm":
                node = nodes[ops_to_nodes[op.op_id]]
                assert node.type == "llm" and node.stage == op.stage
                assert node.n_prompt == op.prompt_tokens
                full = node.n_prompt // BLOCK_SIZE
                assert [f"{h:032x}" for h in node.hashes[:full]] == op.block_hashes
            else:
                node = nodes[ops_to_nodes[op.op_id]]
                assert (
                    node.type == "tool" and abs(node.delay - round(op.delay, 6)) < 1e-6
                )
            assert [ops_to_nodes[x] for x in op.deps if x in ops_to_nodes] == list(
                node.deps
            )


def test_extract_calls_share_the_system_prefix(tmp_path):
    recs = record_reports(load_questions(public=True)[:4], seed=0)
    d, _ = build(recs, tmp_path / "r.json")
    tr = T.load(tmp_path / "r.json")
    extracts = [n for _, n in tr.llm_nodes() if n.stage == "extract"]
    first = {n.hashes[0] for n in extracts}
    assert (
        len(first) == 1
    )  # system prompt + instructions fill at least one shared block


def test_record_reports_is_deterministic(tmp_path):
    qs = load_questions(public=True)[:4]
    a, _ = build(record_reports(qs, seed=5, failure_rate=0.2), tmp_path / "a.json")
    b, _ = build(record_reports(qs, seed=5, failure_rate=0.2), tmp_path / "b.json")
    c, _ = build(record_reports(qs, seed=6, failure_rate=0.2), tmp_path / "c.json")
    assert a == b
    assert a["jobs"] != c["jobs"]


def test_failed_llm_ops_are_dropped_and_deps_inherited(tmp_path):
    recs = record_reports(load_questions(public=True)[:1], seed=0)
    rec = recs[0]
    extract = rec.op("R00/f1/extract")
    extract.status, extract.output = "failed", None
    b = T.TraceBuilder("t", "P1", "test", num_blocks=192)
    m = to_trace_job(rec, b.job(0.0))
    assert "R00/f1/extract" not in m
    synth = b.d["jobs"][0]["nodes"][int(m["R00/synthesize"][1:])]
    fetches = {m[o.op_id] for o in rec.ops if o.op_id.startswith("R00/f1/fetch")}
    assert fetches <= set(synth["deps"])


def test_jsonl_trace_has_every_op(tmp_path):
    import asyncio

    from research_app.agent import run_reports
    from research_app.concurrency import ConcurrencyLimiter
    from research_app.llm import FakeLLM

    async def go():
        return await run_reports(
            load_questions(public=True)[:3],
            llm=FakeLLM(),
            cap=ConcurrencyLimiter(2),
            live=False,
            backend="fake",
        )

    recorder, recs = asyncio.run(go())
    path = tmp_path / "trace.jsonl"
    recorder.export_jsonl(path)
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert lines[0]["event"] == "run_start" and lines[0]["schema"] == "research-trace/1"
    segs = {x["sid"] for x in lines if x["event"] == "segment"}
    starts = {x["op_id"] for x in lines if x["event"] == "op_start"}
    ends = {x["op_id"]: x for x in lines if x["event"] == "op_end"}
    assert starts == set(ends) == {o.op_id for r in recs for o in r.ops}
    for e in ends.values():
        if e.get("prompt"):
            assert (
                set(e["prompt"]) <= segs and e["output"] in segs and e["block_hashes"]
            )
    assert sum(x["event"] == "report_end" for x in lines) == 3
    assert all(x["valid"] is not None for x in lines if x["event"] == "report_end")
