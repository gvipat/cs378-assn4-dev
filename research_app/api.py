"""Run the research workflow on a topic and save its report and sources."""

from __future__ import annotations

import hashlib
import json
import uuid
from html import escape
from dataclasses import dataclass
from pathlib import Path

from agentkv.trace import TraceBuilder

from .agent import AgentSettings, ReportContext, run_report
from .artifacts import Report
from .concurrency import ConcurrencyLimiter
from .llm import LLM
from .search import LocalSearch, SearchBackend, SourceArchive
from .tracing import Recorder, Recording, to_trace_job


@dataclass
class ResearchResult:
    report: Report | None
    recording: Recording
    recorder: Recorder
    sources: SourceArchive
    search_receipts: list[dict]

    @property
    def valid(self) -> bool:
        """Whether the structural citation checks passed; review claim support yourself."""
        return self.recording.valid is True

    def markdown(self) -> str:
        """Format the report with numbered citations and the passages they refer to."""
        if self.report is None:
            return "No report was produced. Inspect the trace for failed operations.\n"
        report = self.report
        lines = [f"# {escape(report.title or report.question)}", ""]
        for claim in report.claims:
            citations = " ".join(
                f"[{eid}](#{eid.lower()})" for eid in claim.evidence_ids
            )
            lines.extend([f"{escape(claim.text)} {citations}", ""])
        lines.extend(["## Sources", ""])
        for evidence in report.evidence:
            doc = self.sources.doc(evidence.doc_id, evidence.snapshot)
            lines.extend(
                [
                    f'<a id="{evidence.evidence_id.lower()}"></a>',
                    f"**{evidence.evidence_id}. [{escape(doc.title)}]({doc.uri})**",
                    "",
                    "> " + escape(evidence.quote).replace("\n", "\n> "),
                    "",
                ]
            )
        if not self.valid:
            lines.extend(["## Citation-check errors", "", *self.recording.errors])
        return "\n".join(lines).rstrip() + "\n"

    def save(self, directory: str | Path) -> Path:
        """Write a new run directory, including source snapshots for offline citation checks."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=False)
        self.recorder.export_jsonl(out / "trace.jsonl")
        self.sources.save(out / "sources.json")
        (out / "report.md").write_text(self.markdown())
        (out / "report.json").write_text(
            json.dumps(
                self.report.to_dict() if self.report else None,
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        rec = self.recording
        calls = [op for op in rec.ops if op.kind == "llm"]
        summary = {
            "topic": rec.question,
            "valid": self.valid,
            "errors": rec.errors,
            "latency": rec.end - rec.arrival,
            "backend": self.recorder.backend,
            "cap": self.recorder.cap.limit,
            **rec.counts(),
            "input_tokens": sum(op.input_tokens for op in calls),
            "output_tokens": sum(op.output_tokens for op in calls),
            "search_receipts": self.search_receipts,
            **self.recorder.meta,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        builder = TraceBuilder(
            "part1-topic",
            "P1",
            "part1",
            num_blocks=192,
            description=f"Research topic: {rec.question}",
        )
        to_trace_job(rec, builder.job(0.0, meta={"question": rec.question_id}))
        builder.save(out / "replay.json")
        return out


class ResearchAgent:
    """Use the implemented LangGraph with a chosen LLM and retrieval backend.

    Keep the same llm when comparing LocalSearch with ExaSearch. FakeLLM is a
    separate choice for testing without model access; it cannot perform real
    research. Concurrent research() calls share this agent's concurrency cap.
    """

    def __init__(
        self,
        *,
        llm: LLM,
        search: SearchBackend | None = None,
        cap: int = 2,
        settings: AgentSettings | None = None,
    ):
        self.llm = llm
        self.search = search if search is not None else LocalSearch()
        self.cap = ConcurrencyLimiter(cap)
        self.settings = settings or AgentSettings()

    async def research(
        self, topic: str, *, out: str | Path | None = None
    ) -> ResearchResult:
        """Research a question, check citations, and return the report and recorded run."""
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("topic must be a nonempty research question")
        if out is not None and Path(out).exists():
            raise FileExistsError(f"choose a new run directory: {out}")
        topic = topic.strip()
        question = {
            "id": "topic-" + hashlib.sha256(topic.encode()).hexdigest()[:12],
            "question": topic,
        }
        recorder = Recorder(
            cap=self.cap,
            backend=self.llm.name,
            meta={"search_backend": self.search.name},
        )
        snapshot = getattr(self.search.corpus, "snapshot", "web")
        client = recorder.open_report(
            "R" + uuid.uuid4().hex[:12],
            question,
            arrival=0.0,
            snapshot=snapshot,
            llm=self.llm,
            tools=self.search,
        )
        try:
            await run_report(ReportContext(client, question, self.settings))
        finally:
            recorder.close_report(client.rec)
        sources = SourceArchive()
        for op in client.rec.ops:
            if op.kind == "fetch" and op.status == "ok":
                doc_id, sha = op.result["doc_id"], op.result["snapshot"]
                version = self.search.corpus.snapshot_of(doc_id, sha)
                sources.add(self.search.corpus.doc(doc_id, version))
        # Receipts are kept by operation for adapters that expose provider costs.
        receipts = [
            r
            for r in getattr(self.search, "receipts", [])
            if r["op_id"] in {op.op_id for op in client.rec.ops}
        ]
        result = ResearchResult(
            Report.from_dict(client.rec.report) if client.rec.report else None,
            client.rec,
            recorder,
            sources,
            receipts,
        )
        if out is not None:
            result.save(out)
        return result
