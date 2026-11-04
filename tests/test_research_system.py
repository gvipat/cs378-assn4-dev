"""Research objectives control selection without replacing the reported scores."""

import json
from types import SimpleNamespace

import pytest

from examples import evaluator, research_system


def test_example_evaluator_keeps_measurements_and_rejects_invalid(monkeypatch):
    for valid in [True, False]:
        measured = {
            "valid": valid,
            "speedup": 1.1,
            "by_family": {"W3": 1.4},
            "violations": [] if valid else ["invalid cap"],
        }
        result = SimpleNamespace(
            **measured,
            to_json=lambda: dict(measured),
            summary=lambda details=False: "table",
        )
        monkeypatch.setattr(evaluator, "score", lambda *a, **kw: result)
        feedback = evaluator.evaluate("policy.py")
        assert {k: feedback[k] for k in measured} == measured
        assert feedback["objective"] == (1.1 if valid else None)
        assert feedback["diagnostics"] == "table"


@pytest.mark.parametrize("objective", [float("nan"), float("inf"), None, "1.0", True])
def test_bad_objective_stops_the_campaign(objective):
    def evaluate(*a, **kw):
        return {
            "valid": True,
            "violations": [],
            "speedup": 1.0,
            "by_family": {"W3": 1.0},
            "objective": objective,
        }

    with pytest.raises(ValueError, match="finite numeric objective"):
        research_system.evaluate_candidate(evaluate, "policy.py")


def test_unfinished_starting_policy_stops_before_generation(tmp_path, monkeypatch):
    def evaluate(*a, **kw):
        return {
            "valid": False,
            "violations": ["NotImplementedError"],
            "speedup": 0.0,
            "by_family": {},
            "objective": None,
        }

    monkeypatch.setattr(research_system, "load_evaluator", lambda path: evaluate)
    monkeypatch.setattr(
        research_system,
        "ask_offline",
        lambda *a: pytest.fail("generated from an invalid start"),
    )
    out = tmp_path / "unfinished"
    monkeypatch.setattr(
        "sys.argv", ["research_system.py", "--offline", "--out", str(out)]
    )
    with pytest.raises(ValueError, match="Complete Part 2"):
        research_system.main()
    assert not (out / "calls.jsonl").exists()


def test_custom_objective_selects_candidate_and_keeps_official_scores(
    tmp_path, monkeypatch
):
    custom = tmp_path / "evaluator.py"
    custom.write_text(
        """
from pathlib import Path

def evaluate(policy_path, *, split):
    assert split == "public_dev"
    name = Path(policy_path).stem
    speedup, objective, valid = {
        "reference": (1.0, 1.0, True),
        "p00": (1.3, 0.8, True),
        "p01": (0.9, 1.5, True),
        "p02": (9.0, 100.0, False),
    }[name]
    return {"valid": valid, "violations": [] if valid else ["invalid cap"],
            "speedup": speedup, "by_family": {"W3": objective}, "objective": objective}
"""
    )
    validation_calls = []

    def validate(path, *, split):
        validation_calls.append((path, split))
        values = {
            "valid": True,
            "speedup": 0.95,
            "by_family": {"W3": 1.4},
            "violations": [],
        }
        return SimpleNamespace(**values, to_json=lambda: values)

    out = tmp_path / "campaign"
    monkeypatch.setattr(research_system, "score", validate)
    monkeypatch.setattr(
        "sys.argv",
        [
            "research_system.py",
            "--offline",
            "--iters",
            "3",
            "--evaluator",
            str(custom),
            "--out",
            str(out),
        ],
    )
    research_system.main()
    selection = json.loads((out / "selection.json").read_text())
    assert selection["candidate"] == "p01"  # best objective, not best overall speedup
    assert selection["dev_objective"] == 1.5
    assert selection["dev_S"] == 0.9
    assert selection["val_S"] == 0.95
    assert (out / "best.py").read_bytes() == (out / "p01.py").read_bytes()
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [r["parent"] for r in rows] == ["start", "start", "p01"]
    assert rows[2]["objective"] is None
    assert validation_calls == [(str(out / "best.py"), "public_val")]
    assert (out / "evaluator.py").read_bytes() == custom.read_bytes()
    assert json.loads((out / "p01.scores.json").read_text())["speedup"] == 0.9
    assert "research evaluator" in (out / "p00.prompt.txt").read_text()
