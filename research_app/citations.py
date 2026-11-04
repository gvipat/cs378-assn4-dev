"""Citation checker (supplied).

A report is valid when it has at least one claim, every claim cites at least
one evidence id, every cited id exists, and every evidence span resolves:
the document exists, the snapshot hash names a real snapshot of it (the
expected one, when given), the offsets are in range, and text[start:end]
equals the quote.
"""

from __future__ import annotations

from dataclasses import dataclass

from .artifacts import Report
from .search import DocumentStore


@dataclass(frozen=True)
class CitationError:
    code: str  # no_claims | uncited_claim | unknown_evidence | unknown_doc |
    # wrong_snapshot | span_out_of_range | quote_mismatch
    claim: int | None  # claim index, when the problem is in a claim
    evidence_id: str | None
    detail: str

    def __str__(self) -> str:
        where = (
            f"claim {self.claim}"
            if self.claim is not None
            else f"evidence {self.evidence_id}"
        )
        return f"{self.code} ({where}): {self.detail}"


def check_report(
    report: Report | dict, corpus: DocumentStore, *, snapshot: str | None = None
) -> list[CitationError]:
    """Return the citation problems found in a report.

    If snapshot is "v1" or "v2", require every evidence item to cite that
    version. Otherwise accept any saved version in the source store. These checks verify the
    cited passages and their locations, not whether they support each claim.
    """
    if isinstance(report, dict):
        report = Report.from_dict(report)
    errs: list[CitationError] = []
    evidence = {e.evidence_id: e for e in report.evidence}
    for ev in report.evidence:
        errs.extend(_check_evidence(ev, corpus, snapshot))
    if not report.claims:
        errs.append(
            CitationError("no_claims", None, None, "the report makes no claims")
        )
    for i, claim in enumerate(report.claims):
        if not claim.evidence_ids:
            errs.append(CitationError("uncited_claim", i, None, claim.text[:80]))
        for eid in claim.evidence_ids:
            if eid not in evidence:
                errs.append(
                    CitationError(
                        "unknown_evidence",
                        i,
                        eid,
                        f"claim cites {eid}, which is not in the evidence list",
                    )
                )
    return errs


def _check_evidence(
    ev, corpus: DocumentStore, snapshot: str | None
) -> list[CitationError]:
    if not corpus.has(ev.doc_id):
        return [
            CitationError(
                "unknown_doc", None, ev.evidence_id, f"no document {ev.doc_id!r}"
            )
        ]
    name = corpus.snapshot_of(ev.doc_id, ev.snapshot)
    if name is None or (snapshot is not None and name != snapshot):
        want = f" (expected {snapshot})" if snapshot else ""
        return [
            CitationError(
                "wrong_snapshot",
                None,
                ev.evidence_id,
                f"{ev.doc_id}: hash {ev.snapshot[:12]} is {name or 'not a known snapshot'}{want}",
            )
        ]
    text = corpus.doc(ev.doc_id, name).text
    if not (0 <= ev.start < ev.end <= len(text)):
        return [
            CitationError(
                "span_out_of_range",
                None,
                ev.evidence_id,
                f"{ev.doc_id}[{ev.start}:{ev.end}] outside 0..{len(text)}",
            )
        ]
    if text[ev.start : ev.end] != ev.quote:
        return [
            CitationError(
                "quote_mismatch",
                None,
                ev.evidence_id,
                f"{ev.doc_id}[{ev.start}:{ev.end}] does not match the quote",
            )
        ]
    return []


def is_valid(
    report: Report | dict, corpus: DocumentStore, *, snapshot: str | None = None
) -> bool:
    return not check_report(report, corpus, snapshot=snapshot)
