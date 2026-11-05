"""The metric on hand-computed tiny DAGs, and the exact points conversion."""

import math

import pytest

from agentkv import metric
from agentkv.engine.timing import TimingModel
from agentkv.replay import Replay
from agentkv.sandbox.client import InProcessClient
from agentkv.trace import HEADER, TraceBuilder, load, output_text

# prefill: 0.01 s per new token; decode: 0.1 s per step; nothing else costs time
T = TimingModel(
    p_base=0.0, p_token=0.01, p_attn=0.0, d_base=0.1, d_seq=0.0, d_ctx=0.0, name="toy"
)

POLICY = """
from agentkv.policy import Policy
class P(Policy):
    def target_concurrency(self, view, state):
        return self.params.get("cap", 8)
    def choose_victim(self, view, candidates, state):
        return candidates[0]
"""


def words(n, tag):
    """Text of exactly n tokens (each ' wN' is one token for this tokenizer)."""
    return "".join(f" {tag}" for _ in range(n))


def run(tmp_path, build, cap=8, num_blocks=64):
    b = TraceBuilder("toy", "D1", "test", num_blocks=num_blocks, block_size=16)
    build(b)
    path = tmp_path / "toy.json"
    b.save(path)
    t = load(path)
    pol = tmp_path / "p.py"
    pol.write_text(POLICY)
    c = InProcessClient(
        str(pol), params={"cap": cap}, block_size=16, total_blocks=num_blocks
    )
    return t, Replay(t, c, timing=T).run()


def tokens(t, sid):
    return len(t.seg_tokens[sid])


def test_single_call(tmp_path):
    def build(b):
        b.job(0.0).llm("x", [words(100, "a")], words(5, "b"))

    t, r = run(tmp_path, build)
    n = next(n for _, n in t.llm_nodes())
    # prefill computes the prompt and yields token 1; then n_output - 1 decode steps
    assert r.flows[0] == pytest.approx(0.01 * n.n_prompt + 0.1 * (n.n_output - 1))


def test_tool_then_call(tmp_path):
    def build(b):
        j = b.job(1.0)
        x = j.tool("search", 2.0)
        j.llm("x", [words(100, "a")], words(5, "b"), deps=[x])

    t, r = run(tmp_path, build)
    n = next(n for _, n in t.llm_nodes())
    assert r.flows[0] == pytest.approx(2.0 + 0.01 * n.n_prompt + 0.1 * (n.n_output - 1))


def test_overlap_and_queueing(tmp_path):
    def build(b):
        j = b.job(0.0)
        j.llm("x", [words(100, "a")], words(5, "b"))
        j.llm("x", [words(100, "c")], words(5, "d"))

    t, par = run(tmp_path, build, cap=2)
    (_, n1), (_, n2) = list(t.llm_nodes())
    # cap 2: one prefill step for both prompts, then shared decode steps
    assert par.flows[0] == pytest.approx(
        0.01 * (n1.n_prompt + n2.n_prompt) + 0.1 * (max(n1.n_output, n2.n_output) - 1)
    )
    _, ser = run(tmp_path, build, cap=1)
    # cap 1: the second call queues behind the first
    assert ser.flows[0] == pytest.approx(
        0.01 * n1.n_prompt
        + 0.1 * (n1.n_output - 1)
        + 0.01 * n2.n_prompt
        + 0.1 * (n2.n_output - 1)
    )


def test_cache_hit_shortens_only_the_critical_path(tmp_path):
    shared = words(64, "s")

    def build(b):
        b.job(0.0).llm(
            "x", [shared, words(10, "p"), HEADER], output_text(words(3, "o"))
        )
        j = b.job(10.0)
        j.tool("slow", 10.0)  # the critical path
        j.llm(
            "x", [shared, words(10, "q"), HEADER], output_text(words(3, "o"))
        )  # off the path

    t, hot = run(tmp_path, build, num_blocks=64)
    assert hot.diag["cache_hit_tokens"] >= 64
    assert hot.flows[1] == pytest.approx(10.0)  # the hit changed nothing
    t, cold = run(
        tmp_path, build, num_blocks=12
    )  # too small to keep the shared prefix around
    assert cold.flows[1] == pytest.approx(10.0)


def test_cache_hit_on_the_critical_path(tmp_path):
    shared = words(64, "s")

    def build(b):
        b.job(0.0).llm(
            "x", [shared, words(10, "p"), HEADER], output_text(words(3, "o"))
        )
        b.job(10.0).llm(
            "x", [shared, words(10, "q"), HEADER], output_text(words(3, "o"))
        )

    t, r = run(tmp_path, build)
    n = list(t.llm_nodes())[1][1]
    hit = 64
    assert r.flows[1] == pytest.approx(
        0.01 * (n.n_prompt - hit) + 0.1 * (n.n_output - 1)
    )


def test_points_conversion():
    top, parity = metric.MAX_POINTS, metric.PARITY_POINTS
    assert metric.points(0.0, g_target=0.10) == parity
    assert metric.points(0.05, g_target=0.10) == pytest.approx((parity + top) / 2)
    assert metric.points(0.10, g_target=0.10) == top
    assert metric.points(0.30, g_target=0.10) == top
    assert metric.points(-0.5, g_target=0.10) == 0.0
    assert metric.points(metric.G_TARGET) == top
    assert 1 - 1 / metric.FULL_CREDIT_SPEEDUP == pytest.approx(metric.G_TARGET)


def rows(ratios):
    return [
        {"family": f, "ratio": r, "mean_flow": 10.0 / r, "b0_mean_flow": 10.0}
        for f, r in ratios
    ]


def test_family_balanced_speedup():
    # one family with 3 traces, one with 1: each family weighs half
    m = metric.summarize(rows([("D1", 1.2), ("D1", 1.2), ("D1", 1.2), ("W1", 1.0)]))
    assert m["speedup"] == pytest.approx(math.sqrt(1.2))
    assert m["by_family"] == {"D1": pytest.approx(1.2), "W1": pytest.approx(1.0)}


def test_family_below_the_floor_costs_its_shortfall():
    m = metric.summarize(rows([("D1", 1.2), ("W1", 0.85)]))
    assert m["capped"] and m["regressed_families"] == ["W1"]
    assert m["g_eff"] == pytest.approx(m["g"] - (metric.FAMILY_FLOOR - 0.85))
    m = metric.summarize(rows([("D1", 1.2), ("W1", 0.95)]))
    assert not m["capped"] and m["g_eff"] == m["g"]
