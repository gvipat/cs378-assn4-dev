"""Part 1: the agent's fixed shape, budgets, partial failures, and the public questions."""

import asyncio

import pytest

from research_app.agent import AgentSettings, run_reports
from research_app.artifacts import BranchStatus
from research_app.citations import is_valid
from research_app.concurrency import ConcurrencyLimiter
from research_app.corpus import Corpus, ToolLatency
from research_app.llm import FakeLLM
from research_app.questions import groups, load_questions


def run(questions, **kw):
    kw.setdefault("live", False)
    cap = kw.pop("cap", ConcurrencyLimiter(8))

    async def go():
        return await run_reports(questions, llm=kw.pop("llm", FakeLLM()), cap=cap, **kw)

    return asyncio.run(go())


def test_question_set_shape():
    qs = load_questions()
    assert len(qs) == 120 and len(groups()) == 12
    public = load_questions(public=True)
    assert len(public) == 16
    assert all(q["group"].endswith("-a") for q in public)
    assert len({q["topic"] for q in public}) == 6
    assert all(2 <= len(q["subtopics"]) <= 4 for q in qs)


def test_public_questions_produce_valid_reports():
    recorder, recs = run(load_questions(public=True))
    corpus = Corpus("v2")
    for rec in recs:
        assert rec.valid, (rec.question_id, rec.errors)
        assert is_valid(rec.report, corpus, snapshot="v2")
        assert rec.report["claims"]
    assert recorder.cap_violations == 0


def test_shape_and_budgets_hold_for_every_question():
    _, recs = run(load_questions())
    for rec in recs:
        c = rec.counts()
        assert c["llm_calls"] <= 8 and c["searches"] <= 5 and c["fetches"] <= 15
        stages = [o.stage for o in rec.ops if o.kind == "llm"]
        assert stages[0] == "plan" and stages.count("plan") == 1
        assert stages.count("synthesize") == 1 and stages.count("repair") <= 1
        assert 2 <= stages.count("extract") <= 5
        assert all(len(o.attempts) <= 3 for o in rec.ops)


def test_repair_path_runs_and_fixes_citations():
    _, recs = run(load_questions(public=True))
    repaired = [r for r in recs if any(o.stage == "repair" for o in r.ops)]
    assert repaired, "the FakeLLM should plant a bad citation in some public reports"
    for rec in repaired:
        first = rec.op(f"{rec.report_id}/validate")
        assert first.validation["valid"] is False
        assert rec.op(f"{rec.report_id}/revalidate").validation["valid"] is True


def test_partial_branch_failure_is_kept_and_ordered():
    def faults(kind, op_id, attempt):
        if op_id.endswith("/f1/fetch2"):
            return "permanent"
        if op_id.endswith("/f2/search"):
            return "permanent"
        return None

    q = [q for q in load_questions(public=True) if len(q["subtopics"]) == 3][:1]
    _, (rec,) = run(q, faults=faults)
    statuses = [(b["subtopic_id"], b["followup"], b["status"]) for b in rec.branches]
    assert statuses[:3] == [
        ("f1", False, BranchStatus.PARTIAL.value),
        ("f2", False, BranchStatus.FAILED.value),
        ("f3", False, BranchStatus.SUCCESS.value),
    ]
    assert statuses[3] == (
        "f2",
        True,
        BranchStatus.SUCCESS.value,
    )  # the follow-up covers f2
    assert rec.op("R00/f1/fetch2").status == "failed"
    assert rec.valid
    # same faults, same recording
    _, (again,) = run(q, faults=faults)
    assert [b["status"] for b in again.branches] == [b["status"] for b in rec.branches]
    assert again.report == rec.report


def test_transient_tool_failures_retry_up_to_three_attempts():
    _, recs = run(
        load_questions(public=True)[:6], latency=ToolLatency(seed=1, failure_rate=0.4)
    )
    tool_ops = [o for r in recs for o in r.ops if o.kind in ("search", "fetch")]
    assert any(o.retries for o in tool_ops)
    assert all(len(o.attempts) <= 3 for o in tool_ops)
    for o in tool_ops:
        if o.status == "failed":
            assert len(o.attempts) == 3 and all(
                a.status == "transient" for a in o.attempts
            )


def test_branch_timeout_keeps_completed_work():
    q = load_questions(public=True)[:1]
    _, (rec,) = run(
        q,
        live=True,
        latency=ToolLatency(seed=0, scale=1.0),
        settings=AgentSettings(branch_timeout=0.05),
    )
    assert {b["status"] for b in rec.branches} == {BranchStatus.TIMEOUT.value}
    assert any(o.status == "cancelled" for o in rec.ops)
    assert rec.valid is False and rec.report is not None


@pytest.mark.parametrize("snapshot", ["v1", "v2"])
def test_evidence_cites_the_snapshot_it_was_fetched_from(snapshot):
    _, recs = run(load_questions(public=True)[:3], snapshot=snapshot)
    corpus = Corpus(snapshot)
    for rec in recs:
        assert is_valid(rec.report, corpus, snapshot=snapshot)


# ---------------------------------------------------------------- the agent under the cap C


@pytest.mark.parametrize("c", [1, 2])
def test_agent_respects_cap_and_retries_keep_their_slot(c):
    """Every LLM call fails twice before succeeding; each retry stays in the
    slot it already holds, so the cap grants one slot per operation."""
    llm = FakeLLM(transient=lambda stage, attempt: attempt < 3)
    cap = ConcurrencyLimiter(c)
    qs = load_questions(public=True)[:4]

    async def go():
        return await run_reports(qs, llm=llm, cap=cap, live=False)

    recorder, recs = asyncio.run(go())
    llm_ops = [o for r in recs for o in r.ops if o.kind == "llm"]
    assert recorder.llm_peak <= c
    assert recorder.cap_violations == 0
    assert cap.grants == len(llm_ops)
    for op in llm_ops:
        assert [a.status for a in op.attempts] == ["transient", "transient", "ok"]
        assert op.retries == 2
    assert all(r.valid for r in recs)


def test_permanent_llm_failure_is_bounded_and_survivable():
    """The plan call never succeeds: three attempts, then a deterministic
    fallback plan, and the report still completes."""
    llm = FakeLLM(transient=lambda stage, attempt: stage == "plan")
    q = load_questions(public=True)[:1]

    async def go():
        return await run_reports(q, llm=llm, cap=ConcurrencyLimiter(2), live=False)

    _, (rec,) = asyncio.run(go())
    plan = rec.op("R00/plan")
    assert plan.status == "failed" and len(plan.attempts) == 3
    assert rec.counts()["llm_calls"] >= 3
    assert rec.report is not None
