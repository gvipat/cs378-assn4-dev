"""Supplied parsers tolerate malformed array fields in model replies."""

import json
from types import SimpleNamespace

import pytest

from research_app.agent import (
    AgentSettings,
    parse_evidence,
    parse_subtopics,
    parse_report,
)
from research_app.artifacts import Subtopic


@pytest.mark.parametrize("value", [None, 7, "not an array", {}])
def test_wrong_array_types_do_not_crash_the_agent(value):
    context = SimpleNamespace(
        rid="R-test",
        question={"id": "q-test", "question": "How does prefix caching work?"},
    )
    plan = parse_subtopics(
        json.dumps({"subtopics": value}), context.question["question"], AgentSettings()
    )
    assert 2 <= len(plan) <= 4
    assert (
        parse_evidence(
            json.dumps({"evidence": value}),
            [],
            Subtopic("f1", "Cache", "cache"),
            "extract",
        )
        == []
    )
    assert parse_report(json.dumps({"claims": value}), context, []).claims == []
    report = parse_report(
        json.dumps({"claims": [{"text": "Unsupported claim", "evidence_ids": value}]}),
        context,
        [],
    )
    assert report.claims[0].evidence_ids == []
