"""Summaries distinguish partial experiments from full scores and explain the units."""

import json
import math

import pytest

from agentkv import metric
from agentkv.score import ScoreResult, TraceResult, main


def result(rows=None, *, split="custom", replicas=2):
    rows = rows or [("W3", 9.99, 10.0, 700, 1000)]
    traces, metrics = {}, []
    for i, (family, policy, reference, hits, prompt) in enumerate(rows):
        name = f"trace-{i}"
        diag = {
            "replicas": replicas,
            "block_size": 16,
            "kv_tokens_per_replica": 4096,
            "profile": {"model": "meta-llama/Meta-Llama-3-8B", "device": "A100 80GB"},
            "prompt_tokens": prompt,
            "cache_hit_tokens": hits,
            "prefill_tokens": prompt - hits,
            "recomputed_tokens": 0,
            "preemptions": 0,
            "evictions": 2,
            "policy_cpu_s": 0.01,
        }
        traces[name] = TraceResult(
            name,
            family,
            True,
            None,
            policy,
            reference,
            reference / policy,
            [policy] * 16,
            [reference] * 16,
            diag,
        )
        metrics.append(
            {
                "family": family,
                "ratio": reference / policy,
                "mean_flow": policy,
                "b0_mean_flow": reference,
            }
        )
    scores = metric.summarize(metrics)
    return ScoreResult(
        policy="policies/fifo.py",
        policy_sha256="a" * 64,
        split=split,
        backend="sim",
        valid=True,
        violations=[],
        by_trace=traces,
        g_target=0.1,
        sim_fingerprint="vidur-test",
        b0_sha256="b" * 64,
        **{k: v for k, v in scores.items() if k != "regressed_families"},
    )


def full_result(*, replicas=2, capped=False):
    rows = [
        (family, (13.0 if family == "W3" else 5.0) if capped else 10.0, 10.0, 700, 1000)
        for family in metric.FAMILY_ORDER
        for _ in range(2)
    ]
    return result(rows, split="public_val", replicas=replicas)


def test_single_trace_explains_context_and_does_not_project_a_grade():
    r = result()
    before = r.to_json()
    text = r.summary()
    assert "1 trace, 16 jobs, 1 of 8" in text
    assert "Document QA (W3)" in text and "2 independent replicas" in text
    assert "fixed staff LRU, concurrency limit 8" in text
    assert "Status: VALID — 16/16 jobs completed" in text
    assert "arrival to completion, including queueing and tool waits" in text
    assert "700 / 1,000 (70.0%)" in text
    assert "excluded from simulated job latency" in text
    assert "Projected performance credit on this split:" not in text
    assert "Scope: this result covers only the selected traces" in text
    assert r.to_json() == before


def test_reuse_is_weighted_by_tokens_and_details_keep_trace_diagnostics():
    r = result([("W3", 9, 10, 100, 100), ("W3", 9, 10, 0, 900)])
    text = r.summary(details=True)
    assert "100 / 1,000 (10.0%)" in text
    assert "Per-trace diagnostics:" in text
    assert "Recomputed tokens" in text and "Policy CPU (ms)" in text
    assert "20.0 ms" in text


@pytest.mark.parametrize("replicas", [1, 2])
def test_points_require_the_full_two_replica_split(replicas):
    text = full_result(replicas=replicas).summary()
    parity = f"Projected performance credit on this split: {metric.PARITY_POINTS:.1f}% of {metric.MAX_POINTS:g}%"
    assert (parity in text) == (replicas == 2)
    assert ("one-replica experiment" in text) == (replicas == 1)
    partial = result(
        [(f, 10, 10, 0, 100) for f in metric.FAMILY_ORDER], split="public_val"
    )
    assert "Projected performance credit on this split:" not in partial.summary()


def test_family_penalty_is_explained():
    r = full_result(capped=True)
    assert r.speedup > 1 and r.capped
    assert "Credit is reduced: categories below 0.90×" in r.summary()


def test_invalid_policy_reports_failure_without_an_infinite_improvement():
    r = result()
    r.valid, r.speedup, r.g, r.points, r.by_family = False, 0.0, -math.inf, 0.0, {}
    r.violations = ["trace-0: target_concurrency must be in 1..16"]
    t = r.by_trace["trace-0"]
    t.valid, t.violation, t.mean_flow, t.flows, t.diag = (
        False,
        r.violations[0],
        math.inf,
        [math.inf] * 16,
        {},
    )
    text = r.summary()
    assert "Status: INVALID — 0/16 jobs completed" in text
    assert "No aggregate speedup" in text and r.violations[0] in text
    assert "Overall speedup" not in text and "inf%" not in text


def test_cli_details_and_json_keep_the_same_measurements(tmp_path, monkeypatch, capsys):
    r = full_result()
    monkeypatch.setattr("agentkv.score.score", lambda *a, **kw: r)
    path = tmp_path / "scores.json"
    assert main(["--policy", "policy.py", "--details", "--json", str(path)]) == 0
    assert json.loads(path.read_text()) == r.to_json()
    output = capsys.readouterr().out
    assert "Per-trace diagnostics:" in output
    assert f"Full results saved to {path}" in output
