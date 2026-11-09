"""Plots for the notebooks: what happened during one replay, and how a score
breaks down by family.

    from agentkv import viz
    rec = viz.record("data/traces/public_dev/dev-W3-0.json", "policies/reference.py", params={"cap": 4})
    viz.timeline(rec)              # one row per job: tools, waits, and LLM calls
    viz.memory(rec)                # active, cached and empty blocks over time
    viz.job_dag(rec.trace, "J03")  # one job's DAG with its critical path
    viz.families(score_result)     # speedup over the reference by family
    viz.speedup_grid(grid, rows, cols)      # a speedup table as a heatmap
    viz.landmarks({"c041": (1.03, 1.01)})   # your candidates among the published policies
    viz.run_timeline("runs/c8/trace.jsonl") # a Part 1 run, one row per report
    viz.critical_path(trace, "J03")         # node ids on a job's longest path

`record` replays in this process without the sandbox, so use it on policies
you trust, such as your own implementations. Use `agentkv.score` for
sandboxed evaluation. The plotting functions require matplotlib
(`pip install -e '.[notebook]'`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from .replay import Replay, RunResult
from .sandbox.client import InProcessClient
from .trace import LLMNode, Memory, Trace, load

EDGE = {
    "edgecolor": "white",
    "linewidth": 0.6,
}  # separates consecutive operations in one lane

# Colors shared by every plot so the notebooks read the same way.
COLORS = {
    "slot": "#f2d16b",
    "memory": "#e8833a",
    "running": "#3b75af",
    "tool": "#8c8c8c",
    "think": "#dcdcdc",
    "active": "#3b75af",
    "cached": "#8fbf6f",
    "empty": "#eeeeee",
    "critical": "#c0392b",
}


@dataclass
class Record:
    """Store a replay's calls, tool timings, and memory history using the original ids."""

    trace: Trace
    result: RunResult
    calls: list[dict] = field(
        default_factory=list
    )  # job, node, stage, ready, dispatch, admit, done,
    # prompt, output, cached (token counts)
    tools: list[dict] = field(default_factory=list)  # job, stage, start, end
    memory: list[tuple[float, int, int, int]] = field(
        default_factory=list
    )  # t, active, cached, empty

    @property
    def flows(self) -> dict[str, float]:
        return dict(zip((j.id for j in self.trace.jobs), self.result.flows))


class _Tap:
    """Wraps a policy client: forwards every event and keeps a copy for plotting."""

    def __init__(self, client):
        self.client, self.log = client, []
        self.replay: Replay | None = None

    def event(self, ev):
        assert self.replay is not None
        if hasattr(self.replay, "memory_counts"):
            counts = self.replay.memory_counts()
        else:
            kv = self.replay.kv
            counts = kv.num_active, len(kv.cached), len(kv.empty)
        self.log.append((ev, counts))
        self.client.event(ev)

    def __getattr__(self, name):
        return getattr(self.client, name)


def record(
    trace,
    policy,
    *,
    params: dict | None = None,
    seed: int = 0,
    replicas: int = 1,
    kv_tokens: int | None = None,
) -> Record:
    """Replay `trace` (a path or a Trace) under `policy` (a path or a Policy class).

    Uses one replica by default; pass replicas=2 to match Part 3 scoring.
    kv_tokens overrides each replica's KV capacity, rounded down to 16-token
    blocks. Raises ValueError if the policy fails or returns an invalid decision.
    """
    if isinstance(trace, (str, Path)):
        trace = load(trace)
    if kv_tokens is not None:
        trace = replace(trace, memory=Memory(kv_tokens // 16, 16))
    from .vidur_replay import VidurReplay, prepare_trace

    trace = prepare_trace(trace)
    m = trace.memory
    tap = _Tap(
        InProcessClient(
            str(policy) if isinstance(policy, Path) else policy,
            params=params,
            seed=seed,
            block_size=m.block_size,
            total_blocks=m.num_blocks * replicas,
        )
    )
    replay = VidurReplay(trace, tap, seed=seed, replicas=replicas)
    tap.replay = replay
    result = replay.run()
    if not result.valid:
        raise ValueError(
            f"{getattr(policy, '__name__', policy)} is invalid on {trace.trace_id}: {result.violation}"
        )

    job_of = {replay.rekey.job(j.id): j.id for j in trace.jobs}
    call_of = {
        replay.rekey.call(j.id, n.id): (j.id, n)
        for j in trace.jobs
        for n in j.nodes
        if isinstance(n, LLMNode)
    }
    stage_of = {
        replay.rekey.stage(n.stage): n.stage for j in trace.jobs for n in j.nodes
    }
    rec = Record(trace, result)
    calls, open_tools = {}, {}
    for ev, (active, cached, empty) in tap.log:
        kind, t = ev[0], ev[1]
        rec.memory.append((t, active, cached, empty))
        if kind == "ready":
            job, node = call_of[ev[2]]
            calls[ev[2]] = {
                "job": job,
                "node": node.id,
                "stage": node.stage,
                "ready": t,
                "dispatch": None,
                "admit": None,
                "done": None,
                "prompt": node.n_prompt,
                "output": node.n_output,
                "cached": 0,
            }
        elif kind == "dispatch":
            calls[ev[2]]["dispatch"] = t
        elif kind == "admit":
            calls[ev[2]]["admit"] = t
            calls[ev[2]]["cached"] = ev[3]
        elif kind == "complete":
            calls[ev[2]]["done"] = t
        elif kind == "tool_start":
            open_tools.setdefault((ev[2], ev[3]), []).append(t)
        elif kind == "tool_done":
            start = open_tools[(ev[2], ev[3])].pop(0)
            rec.tools.append(
                {
                    "job": job_of[ev[2]],
                    "stage": stage_of[ev[3]],
                    "start": start,
                    "end": t,
                }
            )
    rec.calls = list(calls.values())
    return rec


def _axes(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def timeline(
    rec: Record, ax=None, *, jobs: list[str] | None = None, title: str | None = None
):
    """One row per job. Dark gray: tool call. Light gray: think time. Yellow:
    ready but waiting for a concurrency slot. Orange: dispatched, waiting for the
    engine to admit it (for memory, or behind other calls). Blue: admitted
    (prefill and decode). Parallel calls of
    a job share its row."""
    from matplotlib.patches import Patch

    order = jobs or [j.id for j in rec.trace.jobs]
    ax = _axes(ax, (12, 0.42 * len(order) + 1.2))
    row = {j: i for i, j in enumerate(order)}
    arrival = {j.id: j.arrival for j in rec.trace.jobs}
    lanes: dict[str, list[float]] = {}
    for c in sorted(rec.calls, key=lambda c: c["ready"]):
        if c["job"] not in row or c["done"] is None:
            continue
        free = lanes.setdefault(c["job"], [])
        lane = next((i for i, end in enumerate(free) if end <= c["ready"]), None)
        if lane is None:
            free.append(0.0)
            lane = len(free) - 1
        free[lane] = c["done"]
        c["_lane"] = lane
    used = set()
    for t in rec.tools:
        if t["job"] in row:
            kind = "think" if t["stage"] == "think" else "tool"
            used.add(kind)
            ax.barh(
                row[t["job"]],
                t["end"] - t["start"],
                left=t["start"],
                height=0.8,
                color=COLORS[kind],
                linewidth=0,
            )
    for c in rec.calls:
        if c["job"] not in row or c["done"] is None:
            continue
        n = len(lanes[c["job"]])
        h = 0.8 / n
        y = row[c["job"]] - 0.4 + h * c["_lane"] + h / 2
        admit = c["admit"] if c["admit"] is not None else c["done"]
        for start, end, kind in (
            (c["ready"], c["dispatch"], "slot"),
            (c["dispatch"], admit, "memory"),
            (admit, c["done"], "running"),
        ):
            if end > start:
                used.add(kind)
                ax.barh(
                    y,
                    end - start,
                    left=start,
                    height=h * 0.9,
                    color=COLORS[kind],
                    linewidth=0,
                )
    for j in order:
        ax.plot([arrival[j]], [row[j]], marker="|", color="black", markersize=10)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlim(0, _end(rec))
    ax.set_xlabel("time (s)")
    flows = rec.result.flows
    ax.set_title(
        title
        or f"{rec.trace.trace_id}: mean job latency {sum(flows) / len(flows):.2f} s"
    )
    ax.legend(
        handles=[
            Patch(color=COLORS[k], label=v)
            for k, v in (
                ("slot", "waiting for a slot"),
                ("memory", "waiting to be admitted"),
                ("running", "admitted"),
                ("tool", "tool"),
                ("think", "think time"),
            )
            if k in used
        ],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=8,
        frameon=False,
    )
    return ax


def _end(rec: Record) -> float:
    """Last event time of a replay, so stacked timeline and memory plots share an x range."""
    ends = [c["done"] for c in rec.calls if c["done"] is not None] + [
        t["end"] for t in rec.tools
    ]
    return max(ends + [m[0] for m in rec.memory] + [1e-9])


def memory(rec: Record, ax=None, *, title: str | None = None):
    """Stacked memory use over time: blocks held by running calls (active),
    zero-reference prefixes kept for reuse (cached), and empty blocks."""
    ax = _axes(ax, (12, 3))
    ts = [m[0] for m in rec.memory]
    total = rec.trace.memory.num_blocks * rec.result.diag.get("replicas", 1)
    ax.stackplot(
        ts,
        [m[1] for m in rec.memory],
        [m[2] for m in rec.memory],
        [m[3] for m in rec.memory],
        colors=[COLORS["active"], COLORS["cached"], COLORS["empty"]],
        labels=["active", "cached", "empty"],
        step="post",
    )
    ax.set_ylim(0, total)
    ax.set_xlim(0, _end(rec))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("blocks")
    ax.set_title(
        title
        or f"{rec.trace.trace_id}: {total} blocks of {rec.trace.memory.block_size} tokens"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    return ax


def _node_seconds(node) -> float:
    """Predict a call alone on Vidur, with a cold cache and 2048-token chunks."""
    if isinstance(node, LLMNode):
        from .engine.vidur_profile import predictor
        from ._vendor.vidur.entities.execution_time_predictor_request import (
            ExecutionTimePredictorRequest,
        )

        model = predictor()
        elapsed, processed = 0.0, 0
        while processed < node.n_prompt:
            count = min(2048, node.n_prompt - processed)
            elapsed += model.get_execution_time(
                [ExecutionTimePredictorRequest(processed, count, False)], 0
            ).total_time
            processed += count
        processed += 1  # the last prefill step also produces the first output token
        for _ in range(node.n_output - 1):
            elapsed += model.get_execution_time(
                [ExecutionTimePredictorRequest(processed, 1, True)], 0
            ).total_time
            processed += 1
        return elapsed
    return node.delay


def critical_path(
    trace: Trace, job_id: str, durations: dict[str, float] | None = None
) -> list[str]:
    """Node ids on the longest path through the job, by `durations` (node id ->
    seconds) or by a rough standalone estimate when none are given."""
    job = next(j for j in trace.jobs if j.id == job_id)
    nodes = {n.id: n for n in job.nodes}
    dur = durations or {nid: _node_seconds(n) for nid, n in nodes.items()}
    finish, via = {}, {}

    def f(nid):
        if nid not in finish:
            best = max(nodes[nid].deps, key=f, default=None)
            finish[nid] = dur[nid] + (f(best) if best else 0.0)
            via[nid] = best
        return finish[nid]

    sinks = [nid for nid in nodes if not any(nid in n.deps for n in nodes.values())]
    end = max(sinks, key=f)
    path = [end]
    while via[path[-1]]:
        path.append(via[path[-1]])
    return path[::-1]


def job_dag(trace: Trace, job_id: str, ax=None, *, rec: Record | None = None):
    """The job's DAG left to right by depth. Red edges mark the critical path
    (measured from `rec` when given, otherwise estimated)."""
    job = next(j for j in trace.jobs if j.id == job_id)
    nodes = {n.id: n for n in job.nodes}
    durations = None
    if rec is not None:  # measured LLM call times; tool delays are fixed by the trace
        durations = {
            c["node"]: c["done"] - c["ready"]
            for c in rec.calls
            if c["job"] == job_id and c["done"]
        }
        for nid, n in nodes.items():
            durations.setdefault(nid, _node_seconds(n))
    crit = critical_path(trace, job_id, durations)
    depth: dict[str, int] = {}

    def d(nid):
        if nid not in depth:
            depth[nid] = 1 + max((d(x) for x in nodes[nid].deps), default=-1)
        return depth[nid]

    columns: dict[int, list[str]] = {}
    for nid in nodes:
        columns.setdefault(d(nid), []).append(nid)
    pos = {
        nid: (x, -(i - (len(col) - 1) / 2))
        for x, col in columns.items()
        for i, nid in enumerate(col)
    }
    ax = _axes(
        ax, (1.6 * len(columns) + 2, 0.7 * max(len(c) for c in columns.values()) + 1.5)
    )
    on_path = set(zip(crit, crit[1:]))
    for nid, n in nodes.items():
        for dep in n.deps:
            red = (dep, nid) in on_path
            ax.annotate(
                "",
                xy=pos[nid],
                xytext=pos[dep],
                arrowprops={
                    "arrowstyle": "->",
                    "color": COLORS["critical"] if red else "#999999",
                    "lw": 2 if red else 1,
                    "shrinkA": 12,
                    "shrinkB": 12,
                },
            )
    for nid, (x, y) in pos.items():
        n = nodes[nid]
        llm = isinstance(n, LLMNode)
        label = (
            f"{n.stage}\n{n.n_prompt}+{n.n_output} tok"
            if llm
            else f"{n.stage}\n{n.delay:.2f} s"
        )
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=7,
            bbox={
                "boxstyle": "round",
                "fc": "#dbe8f5" if llm else "#eeeeee",
                "ec": COLORS["critical"] if nid in crit else "#777777",
            },
        )
    ax.set_xlim(-0.6, max(columns) + 0.6)
    ax.set_ylim(
        min(y for _, y in pos.values()) - 0.8, max(y for _, y in pos.values()) + 0.8
    )
    ax.axis("off")
    ax.set_title(f"{trace.trace_id} / {job_id}: critical path in red")
    return ax


def families(result, ax=None, *, title: str | None = None):
    """Speedup over the reference per family (a ScoreResult or a dict), drawn from 1.0
    on a log scale, with the per-family penalty threshold and the full-credit speedup.
    """
    from . import metric

    by_family = result if isinstance(result, dict) else result.by_family
    ax = _axes(ax, (7, 3))
    names = list(by_family)
    vals = [by_family[f] for f in names]
    ax.bar(
        names,
        [v - 1.0 for v in vals],
        bottom=1.0,
        color=[COLORS["running"] if v >= 1 else COLORS["memory"] for v in vals],
    )
    ax.set_yscale("log")
    ax.axhline(1.0, color="black", lw=1)
    floor, full = metric.FAMILY_FLOOR, metric.FULL_CREDIT_SPEEDUP
    ax.axhline(
        floor,
        color=COLORS["critical"],
        lw=0.8,
        ls="--",
        label=f"family penalty below {floor:.2f}",
    )
    ax.axhline(
        full, color="#555555", lw=0.8, ls=":", label=f"full credit at S = {full:.3f}"
    )
    ax.set_ylabel("Speedup over reference")
    lo, hi = min(vals + [floor]) / 1.02, max(vals + [full]) * 1.02
    ax.set_ylim(lo, hi)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ticks = [
        t / 20 for t in range(int(lo * 20), int(hi * 20) + 1) if lo <= t / 20 <= hi
    ]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}×"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    if title is None and not isinstance(result, dict):
        title = f"S = {result.speedup:.3f} on {result.split}"
    ax.set_title(title or "")
    return ax


def speedup_grid(grid, rows, cols, ax=None, *, xlabel: str = "", title: str = ""):
    """Heatmap of speedups on a log2 color scale centered at 1 (blue faster, red slower)."""
    import math
    from matplotlib.colors import TwoSlopeNorm

    ax = _axes(ax, (0.75 * len(cols) + 2.5, 0.42 * len(rows) + 1.3))
    logs = [[math.log2(v) for v in row] for row in grid]
    im = ax.imshow(logs, cmap="RdBu", norm=TwoSlopeNorm(0.0, -1.0, 1.0), aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(math.log2(v)) > 0.55 else "black",
            )
    bar = ax.figure.colorbar(im, ax=ax, ticks=[-1, -0.5, 0, 0.5, 1])
    bar.ax.set_yticklabels(["≤0.5×", "0.71×", "1×", "1.41×", "≥2×"])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return ax


def landmarks(candidates: dict[str, tuple[float, float]] | None = None, ax=None):
    """Compare candidate scores on development and validation traces.

    Pass candidates as name -> (dev_S, val_S). Published policies from
    agentkv/data/vidur_benchmarks.json provide reference points. Points above the
    diagonal scored higher on validation; points below scored higher on development.
    """
    import json

    board = json.loads(
        (Path(__file__).parent / "data" / "vidur_benchmarks.json").read_text()
    )["rows"]
    ax = _axes(ax, (6.5, 5))
    labeled = {row["name"]: row["name"] for row in board}
    for row in board:
        x, y = row["public_dev"]["S"], row["public_val"]["S"]
        ax.scatter(x, y, color="#999999", s=18)
        if row["name"] in labeled:
            ax.annotate(
                labeled[row["name"]],
                (x, y),
                fontsize=8,
                color="#555555",
                xytext=(4, -10),
                textcoords="offset points",
            )
    for name, (x, y) in (candidates or {}).items():
        ax.scatter(x, y, color=COLORS["critical"], s=36, zorder=3)
        ax.annotate(name, (x, y), fontsize=8, xytext=(3, 2), textcoords="offset points")
    xs = [r["public_dev"]["S"] for r in board] + [
        c[0] for c in (candidates or {}).values()
    ]
    ys = [r["public_val"]["S"] for r in board] + [
        c[1] for c in (candidates or {}).values()
    ]
    lo, hi = min(xs + ys) - 0.01, max(xs + ys) + 0.01
    ax.plot([lo, hi], [lo, hi], color="#cccccc", lw=1, zorder=0)
    ax.axhline(1.0, color="black", lw=0.8)
    ax.axvline(1.0, color="black", lw=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("S on public_dev")
    ax.set_ylabel("S on public_val")
    ax.set_title("Your candidates (red) and the published policies (gray)")
    return ax


def run_timeline(trace_jsonl, ax=None, *, title: str | None = None):
    """Gantt chart of a Part 1 run (`research_app.run` writes trace.jsonl): one
    row per report, one bar per operation, packed into lanes. Yellow: an LLM call
    waiting for a slot under the cap C. Blue: an LLM call running. Gray: tool calls."""
    import json
    from matplotlib.patches import Patch

    ops, attempts, reports, cap = {}, {}, {}, None
    for line in Path(trace_jsonl).read_text().splitlines():
        ev = json.loads(line)
        kind = ev["event"]
        if kind == "run_start":
            cap = ev.get("cap")
        elif kind == "report_start":
            reports[ev["report_id"]] = ev["arrival"]
        elif kind == "op_start":
            ops[ev["op_id"]] = {
                "report": ev["report_id"],
                "kind": ev["kind"],
                "start": ev["t"],
                "end": None,
            }
        elif kind == "attempt":
            attempts.setdefault(ev["op_id"], []).append((ev["start"], ev["end"]))
        elif kind == "op_end":
            ops[ev["op_id"]].update(
                end=ev["t"],
                slot_requested=ev.get("slot_requested"),
                slot_acquired=ev.get("slot_acquired"),
            )
    order = sorted(reports)
    ax = _axes(ax, (12, 0.42 * len(order) + 1.2))
    row = {r: i for i, r in enumerate(order)}
    lanes: dict[str, list[float]] = {}
    for op_id, op in sorted(ops.items(), key=lambda kv: kv[1]["start"]):
        if op["end"] is None:
            continue
        free = lanes.setdefault(op["report"], [])
        lane = next((i for i, end in enumerate(free) if end <= op["start"]), None)
        if lane is None:
            free.append(0.0)
            lane = len(free) - 1
        free[lane] = op["end"]
        op["lane"] = lane
    for op_id, op in ops.items():
        if op["end"] is None:
            continue
        n = len(lanes[op["report"]])
        h = 0.8 / n
        y = row[op["report"]] - 0.4 + h * op["lane"] + h / 2
        if op["kind"] == "llm":
            ready = (
                op.get("slot_requested")
                if op.get("slot_requested") is not None
                else op["start"]
            )
            first = op.get("slot_acquired")
            if first is None:
                first = min((a[0] for a in attempts.get(op_id, [])), default=ready)
            ax.barh(
                y,
                first - ready,
                left=ready,
                height=h * 0.9,
                color=COLORS["slot"],
                **EDGE,
            )
            ax.barh(
                y,
                op["end"] - first,
                left=first,
                height=h * 0.9,
                color=COLORS["running"],
                **EDGE,
            )
        else:
            ax.barh(
                y,
                op["end"] - op["start"],
                left=op["start"],
                height=h * 0.9,
                color=COLORS["tool"],
                **EDGE,
            )
    for r in order:
        ax.plot([reports[r]], [row[r]], marker="|", color="black", markersize=10)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel("time (s)")
    ax.set_title(title or f"{Path(trace_jsonl).parent.name}: C = {cap}")
    ax.legend(
        handles=[
            Patch(color=COLORS["slot"], label="LLM call waiting for a slot"),
            Patch(color=COLORS["running"], label="LLM call running"),
            Patch(color=COLORS["tool"], label="tool call"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=8,
        frameon=False,
    )
    return ax
