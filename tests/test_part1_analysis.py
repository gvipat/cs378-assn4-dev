"""Part 1 workload analysis: critical path, prefix reuse, and KV footprint."""

import json

import pytest

from research_app.analysis import (
    Op,
    critical_path,
    kv_footprint,
    load_run,
    reuse_intervals,
    prefix_hit_rate,
)

# R00 arrives at 0. Subtopic f1 is slower than f2, and both extractions wait for a slot.
OPS = [
    # op, kind, stage, deps, start, end, slot requested/acquired
    ("plan", "llm", "plan", [], 0.0, 1.5, (0.0, 0.5)),
    ("f1/search", "search", "search", ["plan"], 1.5, 1.8, None),
    ("f2/search", "search", "search", ["plan"], 1.5, 2.0, None),
    ("f1/fetch1", "fetch", "fetch", ["f1/search"], 1.8, 2.4, None),
    ("f2/fetch1", "fetch", "fetch", ["f2/search"], 2.0, 2.2, None),
    ("f1/extract", "llm", "extract", ["f1/fetch1"], 2.4, 4.0, (2.4, 3.0)),
    ("f2/extract", "llm", "extract", ["f2/fetch1"], 2.2, 3.5, (2.2, 2.2)),
    (
        "synthesize",
        "llm",
        "synthesize",
        ["f1/fetch1", "f1/extract", "f2/fetch1", "f2/extract"],
        4.1,
        6.0,
        (4.1, 4.3),
    ),
    ("validate", "validate", "validate", ["synthesize"], 6.0, 6.0, None),
]


@pytest.fixture
def run(tmp_path):
    lines = [
        {"event": "run_start", "t": 0.0, "cap": 2, "backend": "fake"},
        {"event": "report_start", "t": 0.0, "report_id": "R00", "arrival": 0.0},
    ]
    for name, kind, stage, deps, start, end, slot in OPS:
        op_id = f"R00/{name}"
        lines.append(
            {
                "event": "op_start",
                "t": start,
                "report_id": "R00",
                "op_id": op_id,
                "kind": kind,
                "stage": stage,
                "deps": [f"R00/{d}" for d in deps],
            }
        )
        end_ev = {
            "event": "op_end",
            "t": end,
            "report_id": "R00",
            "op_id": op_id,
            "status": "ok",
            "attempts": 1,
        }
        if slot:
            end_ev.update(
                slot_requested=slot[0],
                slot_acquired=slot[1],
                prompt_tokens=600,
                output_tokens=50,
                block_hashes=["a", "b"],
            )
        lines.append(end_ev)
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return load_run(tmp_path)


def test_critical_path_follows_the_last_finishing_dependency(run):
    steps = critical_path(run, "R00")
    assert [s.op_id for s in steps] == [
        "R00/plan",
        "R00/f1/search",
        "R00/f1/fetch1",
        "R00/f1/extract",
        "R00/synthesize",
        "R00/validate",
    ]
    by_id = {s.op_id: s for s in steps}
    assert by_id["R00/plan"].wait == pytest.approx(0.5) and by_id[
        "R00/plan"
    ].service == pytest.approx(1.0)
    assert by_id["R00/f1/extract"].wait == pytest.approx(0.6)
    assert by_id["R00/synthesize"].gap == pytest.approx(0.1)
    assert by_id["R00/f1/search"].wait == 0.0


def test_critical_path_accounts_for_the_whole_latency(run):
    steps = critical_path(run, "R00")
    assert sum(s.gap + s.wait + s.service for s in steps) == pytest.approx(6.0)


def call(stage, start, end, prefix, prompt=1000, output=10, acquired=None):
    return Op(
        f"x/{stage}/{start}",
        "R",
        "llm",
        stage,
        [],
        start,
        end,
        "ok",
        1,
        start,
        start if acquired is None else acquired,
        prompt,
        output,
        list(prefix),
    )


def test_prefix_hit_rate_counts_blocks_seen_earlier():
    calls = [
        call("plan", 0, 1, ["s"], prompt=300),
        call("extract", 1, 2, ["s", "d"], prompt=600),
        call("extract", 1.5, 3, ["s", "d", "e"], prompt=800),
    ]
    assert prefix_hit_rate(calls) == pytest.approx((256 + 2 * 256) / 1700)
    assert prefix_hit_rate([]) == 0.0


def test_reuse_intervals_measure_time_since_the_last_holder_finished():
    calls = [
        call("plan", 0.0, 1.0, ["s"]),
        call("extract", 2.0, 3.0, ["s", "d"]),
        call("extract", 2.5, 4.0, ["s", "d"]),
        call("synthesize", 6.0, 7.0, ["s", "d", "d"]),
    ]
    intervals = [
        (r.block, r.stage, r.source, r.interval) for r in reuse_intervals(calls)
    ]
    assert intervals == [
        ("s", "extract", "plan", 1.0),
        ("s", "extract", "extract", 0.0),
        ("d", "extract", "extract", 0.0),
        ("s", "synthesize", "extract", 2.0),
        ("d", "synthesize", "extract", 2.0),
    ]


def test_kv_footprint_shares_identical_full_blocks():
    calls = [
        call("extract", 0.0, 2.0, ["s", "a"], prompt=600, output=40),
        call("extract", 1.0, 3.0, ["s", "b"], prompt=520, output=20, acquired=1.5),
    ]
    fp = kv_footprint(calls, bytes_per_token=1, block_size=256)
    assert [f.t for f in fp] == [0.0, 1.5, 2.0, 3.0]
    assert [f.calls for f in fp] == [1, 2, 1, 0]
    assert fp[1].private == 640 + 540
    assert fp[1].shared == 3 * 256 + (600 - 512 + 40) + (520 - 512 + 20)
