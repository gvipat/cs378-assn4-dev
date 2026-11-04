"""The notebook plotting helpers: a recorded replay matches the trace, and every plot draws."""

import pytest
from conftest import FIFO, build_chat_trace

from agentkv import viz
from agentkv.trace import TraceBuilder, chat_segment, load, output_text

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    path = tmp_path_factory.mktemp("viz") / "chat.json"
    build_chat_trace(path, jobs=6, turns=2)
    return viz.record(load(path), FIFO, params={"cap": 3})


def test_record_sees_every_call_and_tool(rec):
    llm = sum(1 for _ in rec.trace.llm_nodes())
    tools = sum(1 for j in rec.trace.jobs for n in j.nodes if n.type == "tool")
    assert len(rec.calls) == llm and len(rec.tools) == tools
    for c in rec.calls:
        assert c["ready"] <= c["dispatch"] <= c["admit"] <= c["done"]
    total = rec.trace.memory.num_blocks
    assert all(a + c + e == total for _, a, c, e in rec.memory)
    assert rec.result.valid and len(rec.flows) == len(rec.trace.jobs)


def test_plots_draw(rec):
    import matplotlib.pyplot as plt

    viz.timeline(rec)
    viz.memory(rec)
    viz.job_dag(rec.trace, rec.trace.jobs[0].id, rec=rec)
    viz.families({"D1": 1.1, "W3": 0.9})
    viz.landmarks({"mine": (1.02, 1.01)})
    plt.close("all")


def test_critical_path_follows_the_longest_chain(tmp_path):
    b = TraceBuilder("dag", "D1", "test", num_blocks=64, block_size=16)
    job = b.job(arrival=0.0)
    plan = job.llm("plan", [chat_segment("user", "plan this")], output_text("ok"))
    fast = job.tool("search", 0.1, deps=[plan])
    slow = job.tool("search", 5.0, deps=[plan])
    job.llm(
        "write",
        [chat_segment("user", "write it")],
        output_text("done"),
        deps=[fast, slow],
    )
    b.save(tmp_path / "dag.json")
    trace = load(tmp_path / "dag.json")
    assert viz.critical_path(trace, job.id) == [plan, slow, "n3"]
