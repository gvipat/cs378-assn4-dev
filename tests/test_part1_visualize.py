"""Check the measurements behind the Part 1 figures."""

import asyncio
import math
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from research_app.concurrency import ConcurrencyLimiter
from research_app.visualize import plot_cap_demo, plot_prefix_reuse, run_cap_demo


def test_cap_demo_shows_existing_calls_finishing_before_new_calls_start():
    demo = asyncio.run(run_cap_demo(ConcurrencyLimiter(3)))
    first, waiting = demo.calls[:3], demo.calls[3:]
    assert all(c["start"] < demo.change_time < c["end"] for c in first)
    assert min(c["start"] for c in waiting) >= max(c["end"] for c in first)
    assert all(a["end"] <= b["start"] for a, b in zip(waiting, waiting[1:]))
    assert demo.peak == 3 and demo.over_limit == 0
    fig = plot_cap_demo(demo)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_stage_plot_keeps_history_from_other_stages():
    # This small oracle assigns 256 reused tokens to each call after the first,
    # including the third call's reuse of content introduced by another stage.
    calls = [
        SimpleNamespace(stage=stage, prompt_tokens=tokens)
        for stage, tokens in [("plan", 300), ("extract", 600), ("plan", 400)]
    ]

    def share(prefix):
        return 256 * (len(prefix) - 1) / sum(c.prompt_tokens for c in prefix)

    fig, rows = plot_prefix_reuse(calls, share)
    assert math.isclose(rows["plan"]["shared"], 256)
    assert math.isclose(rows["extract"]["shared"], 256)
    assert rows["plan"]["tokens"] == 700
    plt.close(fig)


def test_slot_wait_measurement_excludes_prompt_preparation(tmp_path):
    import json
    from research_app.visualize import run_metrics

    events = [
        {
            "event": "op_start",
            "op_id": "R00/plan",
            "report_id": "R00",
            "kind": "llm",
            "t": 1,
        },
        {"event": "attempt", "op_id": "R00/plan", "start": 5.1, "end": 6},
        {
            "event": "op_end",
            "op_id": "R00/plan",
            "t": 6,
            "slot_requested": 3,
            "slot_acquired": 5,
        },
    ]
    (tmp_path / "trace.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    (tmp_path / "summary.json").write_text("{}")
    assert run_metrics(tmp_path)["mean_slot_wait"] == 2
