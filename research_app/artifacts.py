"""Data classes for the research plan, collected evidence, and final report.

Evidence      a verbatim span of one document snapshot, found by code
Claim         one sentence of the report and the evidence ids it cites
Report        the question, its claims, and the evidence they cite
BranchResult  what one subtopic's research branch produced, success or not
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum


@dataclass(frozen=True)
class Subtopic:
    """A planned research subtopic, with a title and search query."""

    subtopic_id: str  # "f1", "f2", ...
    title: str
    query: str


@dataclass
class Evidence:
    """A quoted passage with its document version, text offsets, and source call."""

    evidence_id: str  # "E1", "E2", ... assigned after all branches finish
    doc_id: str
    snapshot: str  # SHA-256 of the snapshot text the span was taken from
    start: int  # text[start:end] == quote
    end: int
    quote: str
    subtopic_id: str
    uri: str
    extraction_call_id: str


@dataclass
class Claim:
    """A report statement and the evidence ids it cites."""

    text: str
    evidence_ids: list[str]


@dataclass
class Report:
    """The answer to a research question, including its claims and source passages."""

    report_id: str
    question_id: str
    question: str
    title: str
    claims: list[Claim]
    evidence: list[Evidence]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Report":
        return cls(
            d["report_id"],
            d["question_id"],
            d["question"],
            d.get("title", ""),
            [Claim(c["text"], list(c["evidence_ids"])) for c in d.get("claims", [])],
            [Evidence(**e) for e in d.get("evidence", [])],
        )


class BranchStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"  # some fetches failed, but extraction succeeded
    TIMEOUT = "timeout"
    FAILED = "permanent_failure"


@dataclass
class BranchResult:
    """The work completed for a subtopic, including any failures or timeout."""

    subtopic: Subtopic
    index: int  # subtopic position in the plan; follow-ups use 100 + index
    status: BranchStatus = BranchStatus.SUCCESS
    evidence: list[Evidence] = field(default_factory=list)  # evidence_id still ""
    docs: list[str] = field(default_factory=list)  # doc ids fetched successfully
    ops: list[str] = field(
        default_factory=list
    )  # ids of operations started by this branch
    errors: list[str] = field(default_factory=list)
    followup: bool = False

    @property
    def last_op(self) -> str | None:
        return self.ops[-1] if self.ops else None


def order_branches(branches: list[BranchResult]) -> list[BranchResult]:
    """Sort original branches by plan index, followed by any follow-up branches."""
    return sorted(branches, key=lambda b: (b.followup, b.index))


def assign_evidence_ids(branches: list[BranchResult]) -> list[Evidence]:
    """Combine branch evidence, remove duplicate passages, and assign ids E1, E2, ...

    Process branches in plan order, then passages by document id and offset.
    This makes the ids independent of the order in which branches finished.
    """
    out: list[Evidence] = []
    seen: set[tuple] = set()
    for b in order_branches(branches):
        for ev in sorted(b.evidence, key=lambda e: (e.doc_id, e.start, e.end)):
            key = (ev.doc_id, ev.snapshot, ev.start, ev.end)
            if key in seen:
                continue
            seen.add(key)
            ev.evidence_id = f"E{len(out) + 1}"
            out.append(ev)
    return out


def locate(quote: str, text: str) -> tuple[int, int] | None:
    """Find the offsets of a quoted passage in the document text.

    Try an exact match first, then allow differences in whitespace. Return
    None if no match is found; otherwise return offsets into the original text.
    """
    quote = quote.strip()
    if not quote:
        return None
    i = text.find(quote)
    if i >= 0:
        return i, i + len(quote)
    pattern = r"\s+".join(re.escape(w) for w in quote.split())
    m = re.search(pattern, text)
    return (m.start(), m.end()) if m else None
