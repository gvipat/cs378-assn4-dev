"""Load the research questions from research_app/data/questions.json."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

QUESTIONS_PATH = Path(__file__).parent / "data" / "questions.json"


@lru_cache(maxsize=1)
def _all() -> tuple[dict, ...]:
    return tuple(json.loads(QUESTIONS_PATH.read_text())["questions"])


def load_questions(
    *,
    public: bool | None = None,
    groups: list[str] | None = None,
    topics: list[str] | None = None,
) -> list[dict]:
    """Return questions in file order, optionally filtering by public flag, group, or topic."""
    out = []
    for q in _all():
        if public is not None and q["public"] != public:
            continue
        if groups is not None and q["group"] not in groups:
            continue
        if topics is not None and q["topic"] not in topics:
            continue
        out.append(dict(q))
    return out


def groups() -> list[str]:
    return sorted({q["group"] for q in _all()})
