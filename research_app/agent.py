"""Supplied Part 1 research workflow as a LangGraph graph.

    plan ─▶ research_subtopic × 2–4 (parallel) ─▶ coverage ─▶ [followup] ─▶ synthesize ─▶ validate ─▶ [repair]

- plan: one LLM call splits the question into 2–4 subtopics.
- research_subtopic: per subtopic, one search, up to three fetches, one extract call.
- coverage: a deterministic check; the first subtopic with no evidence gets one
  follow-up branch.
- synthesize: one LLM call writes claims that cite evidence ids.
- validate: the supplied citation checker; at most one repair call.

Per report: at most 8 LLM calls and 20 tool operations (5 searches, 15
fetches), and at most 3 attempts per operation (the tracing client enforces
these). LLM calls share one global cap C across every running report.

Use the LLM stage names plan, extract, synthesize and repair, and build every
prompt with the matching `prompts.*_messages` function: FakeLLM recognizes
calls by stage and reads the prompt in that layout, and the fixed layout is
what makes calls share prefixes.

The workflow is adapted from LangChain's Deep Research from Scratch (MIT):
it plans the research, gathers evidence in parallel, and writes a report.
The supplied parser locates each quoted passage in the source text to obtain
its offsets; the model only needs to provide the quote.
"""

from __future__ import annotations

import asyncio
import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import prompts
from .artifacts import (
    BranchResult,
    BranchStatus,
    Claim,
    Evidence,
    Subtopic,
    Report,
    assign_evidence_ids,
    locate,
    order_branches,
)
from .concurrency import ConcurrencyLimiter
from .corpus import Corpus, Doc, ToolLatency, Tools, terms
from .llm import LLM, _heuristic_subtopics
from .tracing import Recorder, Recording, TracingClient


@dataclass(frozen=True)
class AgentSettings:
    min_subtopics: int = 2
    max_subtopics: int = 4
    search_k: int = 5
    max_fetch: int = 3
    plan_tokens: int = 400
    extract_tokens: int = 1500
    synthesize_tokens: int = 1200
    branch_timeout: float | None = None  # seconds per branch; None disables the timeout


@dataclass
class ReportContext:
    client: TracingClient
    question: dict  # {"id", "question", ...} from questions.json
    settings: AgentSettings = field(default_factory=AgentSettings)

    @property
    def rid(self) -> str:
        return self.client.report_id

    @property
    def corpus(self) -> Corpus:
        return self.client.corpus


class ReportState(TypedDict, total=False):
    question: str
    question_id: str
    subtopics: list[Subtopic]
    plan_op: str
    branches: Annotated[list[BranchResult], operator.add]
    uncovered: int | None  # index of the first subtopic without evidence
    evidence: list[Evidence]
    synth_messages: list[prompts.Msg]
    draft: str
    synth_op: str
    report: Report
    errors: list[str]
    validate_op: str
    repaired: bool
    valid: bool


def ctx_of(config: RunnableConfig) -> ReportContext:
    """Return the report context passed to this graph node through its configuration."""
    return config["configurable"]["ctx"]


# ---------------------------------------------------------------- parsing (supplied)


def _list_field(obj: dict, key: str) -> list:
    """Treat a missing or incorrectly typed JSON array as empty."""
    value = obj.get(key)
    return value if isinstance(value, list) else []


def parse_subtopics(
    text: str | None, question: str, settings: AgentSettings
) -> list[Subtopic]:
    """Parse the planned subtopics, or split the question if the reply is unusable.

    The fallback produces a repeatable plan without another LLM call.
    """
    obj = prompts.parse_json(text or "") or {}
    raw = [
        f
        for f in _list_field(obj, "subtopics")
        if isinstance(f, dict) and f.get("title")
    ]
    if len(raw) < settings.min_subtopics:
        raw = _heuristic_subtopics(question)
    raw = raw[: settings.max_subtopics]
    return [
        Subtopic(
            f"f{i + 1}",
            str(f["title"]).strip(),
            str(f.get("query") or f["title"]).strip(),
        )
        for i, f in enumerate(raw)
    ]


def parse_evidence(
    text: str | None, docs: list[Doc], subtopic: Subtopic, call_id: str
) -> list[Evidence]:
    """Parse quoted passages and locate them in the fetched documents.

    Discard quotes that cannot be matched to a source passage. Store offsets
    into the original document text so the citation checker can verify them.
    """
    obj = prompts.parse_json(text or "") or {}
    by_id = {d.doc_id: d for d in docs}
    out = []
    for item in _list_field(obj, "evidence"):
        if not isinstance(item, dict):
            continue
        doc = by_id.get(str(item.get("doc_id", "")))
        quote = str(item.get("quote", ""))
        span = locate(quote, doc.text) if doc else None
        if span is None:
            continue
        start, end = span
        out.append(
            Evidence(
                "",
                doc.doc_id,
                doc.sha256,
                start,
                end,
                doc.text[start:end],
                subtopic.subtopic_id,
                doc.uri,
                call_id,
            )
        )
    return out


def parse_report(
    text: str | None, ctx: ReportContext, evidence: list[Evidence]
) -> Report:
    """Parse the model reply into a Report, omitting claims with no text."""
    obj = prompts.parse_json(text or "") or {}
    claims = [
        Claim(
            str(c.get("text", "")).strip(),
            [str(e) for e in _list_field(c, "evidence_ids")],
        )
        for c in _list_field(obj, "claims")
        if isinstance(c, dict) and c.get("text")
    ]
    return Report(
        ctx.rid,
        ctx.question["id"],
        ctx.question["question"],
        str(obj.get("title", "")),
        claims,
        list(evidence),
    )


# ---------------------------------------------------------------- graph nodes
# All graph nodes, fan-out, and graph wiring are supplied.


async def plan(state: ReportState, config: RunnableConfig) -> dict:
    """Plan the subtopics for one research question."""
    ctx = ctx_of(config)
    question = state["question"]
    resp, op = await ctx.client.llm_call(
        "plan",
        prompts.plan_messages(question),
        op_id=f"{ctx.rid}/plan",
        deps=[],
        order=(0,),
        max_tokens=ctx.settings.plan_tokens,
    )
    return {
        "subtopics": parse_subtopics(
            resp.text if resp else None, question, ctx.settings
        ),
        "plan_op": op.op_id,
    }


def fan_out(state: ReportState) -> list[Send]:
    """Create the parallel research tasks for the planned subtopics."""
    return [
        Send(
            "research_subtopic",
            {
                "question": state["question"],
                "subtopic": f,
                "index": i,
                "deps": [state["plan_op"]],
            },
        )
        for i, f in enumerate(state["subtopics"])
    ]


def branch_ids(
    subtopic: Subtopic, index: int, followup: bool
) -> tuple[str, tuple[int, ...], int]:
    """Return a branch's op_id segment, its order prefix, and its BranchResult index."""
    if followup:
        return "followup", (2, 0), 100 + index
    return subtopic.subtopic_id, (1, index), index


async def research_branch(
    ctx: ReportContext,
    question: str,
    subtopic: Subtopic,
    index: int,
    deps: list[str],
    *,
    query: str | None = None,
    followup: bool = False,
) -> BranchResult:
    """Research one subtopic and retain the completed work if an operation fails."""
    tag, phase, result_index = branch_ids(subtopic, index, followup)
    result = BranchResult(subtopic, result_index, followup=followup)

    async def run() -> None:
        search_id = f"{ctx.rid}/{tag}/search"
        result.ops.append(search_id)
        hits, search_op = await ctx.client.search(
            query or subtopic.query,
            op_id=search_id,
            deps=deps,
            order=phase + (0,),
            k=ctx.settings.search_k,
        )
        if hits is None:
            result.status = BranchStatus.FAILED
            result.errors.append("search failed")
            return
        if not hits:
            result.status = BranchStatus.FAILED
            result.errors.append("no search results")
            return

        async def fetch(k, hit):
            op_id = f"{ctx.rid}/{tag}/fetch{k + 1}"
            result.ops.append(op_id)
            doc, op = await ctx.client.fetch(
                hit.doc_id, op_id=op_id, deps=[search_op.op_id], order=phase + (1, k)
            )
            if doc is None:
                result.errors.append(f"fetch failed: {hit.doc_id}")
            else:
                result.docs.append(doc.doc_id)
                result.docs.sort()
            return doc, op

        fetched = await asyncio.gather(
            *(fetch(k, hit) for k, hit in enumerate(hits[: ctx.settings.max_fetch]))
        )
        docs = [doc for doc, op in fetched if doc is not None]
        if not docs:
            result.status = BranchStatus.FAILED
            return
        fetch_ops = [op.op_id for doc, op in fetched]
        extract_id = f"{ctx.rid}/{tag}/extract"
        result.ops.append(extract_id)
        resp, ex_op = await ctx.client.llm_call(
            "extract",
            prompts.extract_messages(question, subtopic, docs),
            op_id=extract_id,
            deps=fetch_ops,
            order=phase + (2,),
            max_tokens=ctx.settings.extract_tokens,
        )
        if resp is None:
            result.status = BranchStatus.FAILED
            result.errors.append("extract call failed")
            return
        result.evidence = parse_evidence(resp.text, docs, subtopic, ex_op.op_id)
        result.status = (
            BranchStatus.PARTIAL if len(docs) < len(fetched) else BranchStatus.SUCCESS
        )

    try:
        if ctx.settings.branch_timeout is not None:
            async with asyncio.timeout(ctx.settings.branch_timeout):
                await run()
        else:
            await run()
    except TimeoutError:
        result.status = BranchStatus.TIMEOUT
        result.errors.append(f"timed out after {ctx.settings.branch_timeout}s")
    return result


async def research_subtopic(state: dict, config: RunnableConfig) -> dict:
    """Run one parallel branch and return its result to the graph."""
    ctx = ctx_of(config)
    branch = await research_branch(
        ctx, state["question"], state["subtopic"], state["index"], state["deps"]
    )
    return {"branches": [branch]}


def coverage(state: ReportState) -> dict:
    """Find the first planned subtopic for which no evidence was collected."""
    covered = {
        b.index for b in state.get("branches", []) if b.evidence and not b.followup
    }
    missing = [i for i in range(len(state["subtopics"])) if i not in covered]
    return {"uncovered": missing[0] if missing else None}


def route_coverage(state: ReportState) -> str:
    """Choose whether to research an uncovered subtopic or write the report."""
    return "followup" if state.get("uncovered") is not None else "synthesize"


def _branch_tails(branches: list[BranchResult]) -> list[str]:
    """Collect the operation ids that the next research step must wait for.

    A failed or timed-out branch may have started several parallel operations.
    Include all of them to preserve those dependencies. This also includes
    some earlier operations whose completion is already implied by later ones.
    """
    return list(dict.fromkeys(op for b in order_branches(branches) for op in b.ops))


async def followup(state: ReportState, config: RunnableConfig) -> dict:
    """Make one additional research attempt for the first uncovered subtopic."""
    ctx = ctx_of(config)
    i = state["uncovered"]
    subtopic = state["subtopics"][i]
    # _branch_tails makes the follow-up wait for every original branch.
    query = " ".join(
        dict.fromkeys(terms(f"{subtopic.title} {subtopic.query} {state['question']}"))
    )[:200]
    branch = await research_branch(
        ctx,
        state["question"],
        subtopic,
        i,
        _branch_tails(state["branches"]),
        query=query,
        followup=True,
    )
    return {"branches": [branch]}


async def synthesize(state: ReportState, config: RunnableConfig) -> dict:
    """Combine the collected evidence into a report with citations."""
    ctx = ctx_of(config)
    evidence = assign_evidence_ids(state.get("branches", []))
    messages = prompts.synthesize_messages(
        state["question"], state["subtopics"], evidence
    )
    resp, op = await ctx.client.llm_call(
        "synthesize",
        messages,
        op_id=f"{ctx.rid}/synthesize",
        deps=_branch_tails(state.get("branches", [])),
        order=(3,),
        max_tokens=ctx.settings.synthesize_tokens,
    )
    return {
        "evidence": evidence,
        "synth_messages": messages,
        "draft": resp.text if resp else "",
        "synth_op": op.op_id,
        "report": parse_report(resp.text if resp else None, ctx, evidence),
    }


async def validate(state: ReportState, config: RunnableConfig) -> dict:
    """Check the report citations and save the validation result."""
    ctx = ctx_of(config)
    repaired = state.get("repaired", False)
    # Leave snapshot unset so evidence can cite either saved corpus version.
    errors, op = ctx.client.validate(
        state["report"],
        op_id=f"{ctx.rid}/{'revalidate' if repaired else 'validate'}",
        deps=[state["synth_op"]],
        order=(6,) if repaired else (4,),
    )
    return {
        "errors": [str(e) for e in errors],
        "validate_op": op.op_id,
        "valid": not errors,
    }


def route_validation(state: ReportState) -> str:
    """End the report or allow its one repair attempt."""
    if state["valid"] or state.get("repaired") or not state.get("draft"):
        return END
    return "repair"


async def repair(state: ReportState, config: RunnableConfig) -> dict:
    """Ask the model to fix citation errors in the current draft."""
    ctx = ctx_of(config)
    messages = prompts.repair_messages(
        state["synth_messages"], state["draft"], state["errors"]
    )
    resp, op = await ctx.client.llm_call(
        "repair",
        messages,
        op_id=f"{ctx.rid}/repair",
        deps=[state["validate_op"]],
        order=(5,),
        max_tokens=ctx.settings.synthesize_tokens,
    )
    report = (
        parse_report(resp.text, ctx, state["evidence"]) if resp else state["report"]
    )
    # synth_op moves to the repair so revalidation waits for it.
    return {"report": report, "repaired": True, "synth_op": op.op_id}


def build_graph():
    """Build and compile the required LangGraph workflow."""
    g = StateGraph(ReportState)
    g.add_node("plan", plan)
    g.add_node("research_subtopic", research_subtopic)
    g.add_node("coverage", coverage)
    g.add_node("followup", followup)
    g.add_node("synthesize", synthesize)
    g.add_node("validate", validate)
    g.add_node("repair", repair)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out, ["research_subtopic"])
    g.add_edge("research_subtopic", "coverage")
    g.add_conditional_edges("coverage", route_coverage, ["followup", "synthesize"])
    g.add_edge("followup", "synthesize")
    g.add_edge("synthesize", "validate")
    g.add_conditional_edges("validate", route_validation, ["repair", END])
    g.add_edge("repair", "validate")
    return g.compile()


# ---------------------------------------------------------------- running (supplied)

_GRAPH = None


async def run_report(ctx: ReportContext) -> ReportState:
    """Run the compiled graph for one report and return its final state."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    state = await _GRAPH.ainvoke(
        {"question": ctx.question["question"], "question_id": ctx.question["id"]},
        config={"configurable": {"ctx": ctx}, "recursion_limit": 50},
    )
    rec = ctx.client.rec
    rec.valid = state.get("valid", False)
    rec.errors = state.get("errors", [])
    rec.report = state["report"].to_dict() if state.get("report") else None
    rec.branches = [
        {
            "subtopic_id": b.subtopic.subtopic_id,
            "index": b.index,
            "followup": b.followup,
            "status": b.status.value,
            "docs": b.docs,
            "evidence": len(b.evidence),
            "errors": b.errors,
        }
        for b in order_branches(state.get("branches", []))
    ]
    return state


async def run_reports(
    questions: list[dict],
    *,
    llm: LLM,
    cap: ConcurrencyLimiter,
    arrivals: list[float] | None = None,
    snapshot: str = "v2",
    live: bool = True,
    latency: ToolLatency | None = None,
    faults=None,
    settings: AgentSettings = AgentSettings(),
    backend: str = "",
    recorder: Recorder | None = None,
) -> tuple[Recorder, list[Recording]]:
    """Run one report per question, sharing the same concurrency cap.

    In live mode, report i starts arrivals[i] seconds after the run begins.
    Return the recorder and a list of recordings in question order.
    """
    recorder = recorder or Recorder(cap=cap, live=live, backend=backend)
    tools = Tools(Corpus(snapshot), latency, faults)
    arrivals = arrivals or [0.0] * len(questions)

    async def one(i: int, q: dict, arrival: float) -> Recording:
        if live:
            await asyncio.sleep(max(0.0, arrival - recorder.now()))
        client = recorder.open_report(
            f"R{i:02d}", q, arrival=arrival, snapshot=snapshot, llm=llm, tools=tools
        )
        try:
            await run_report(ReportContext(client, q, settings))
        finally:
            recorder.close_report(client.rec)
        return client.rec

    recs = await asyncio.gather(
        *(one(i, q, a) for i, (q, a) in enumerate(zip(questions, arrivals)))
    )
    return recorder, list(recs)
