"""Plots for the Part 1 experiments, using recorded events and explicit units."""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RUNNING = "#2563a6"
WAITING = "#e6a33d"
TOOL = "#a5adb5"
CRITICAL = "#b73c35"
STAGES = {
    "plan": "Plan",
    "extract": "Extract evidence",
    "synthesize": "Write report",
    "repair": "Revise report",
    "search": "Search",
    "fetch": "Read source",
    "validate": "Check citations",
}


@dataclass
class CapDemo:
    calls: list[dict]
    change_time: float
    initial_limit: int
    final_limit: int
    peak: int
    over_limit: int


async def run_cap_demo(cap) -> CapDemo:
    """Run six timed tasks; lower a cap of three while the first three are running."""
    if cap.limit != 3:
        raise ValueError("start this demonstration with ConcurrencyLimiter(3)")
    clock = time.perf_counter
    start = clock()
    calls = []
    three_started = asyncio.Event()

    async def call(index, duration):
        row = {"call": index + 1, "ready": clock() - start}
        async with cap.slot():
            row["start"] = clock() - start
            calls.append(row)
            if len(calls) == 3:
                three_started.set()
            await asyncio.sleep(duration)
            row["end"] = clock() - start

    async def lower_limit():
        await three_started.wait()
        await asyncio.sleep(0.15)
        await cap.set_limit(1)
        return clock() - start

    durations = [0.45, 0.65, 0.85, 0.30, 0.30, 0.30]
    changed, *_ = await asyncio.wait_for(
        asyncio.gather(lower_limit(), *(call(i, d) for i, d in enumerate(durations))),
        timeout=8,
    )
    return CapDemo(
        sorted(calls, key=lambda row: row["call"]),
        changed,
        3,
        1,
        cap.peak,
        cap.over_limit,
    )


def plot_cap_demo(demo: CapDemo):
    """Show each call's wait and execution, aligned with the cap and active count."""
    fig, (tasks, count) = plt.subplots(
        2,
        1,
        figsize=(11, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
        layout="constrained",
    )
    end = max(c["end"] for c in demo.calls)
    for i, call in enumerate(demo.calls):
        tasks.barh(
            i,
            call["start"] - call["ready"],
            left=call["ready"],
            height=0.55,
            color=WAITING,
        )
        tasks.barh(
            i,
            call["end"] - call["start"],
            left=call["start"],
            height=0.55,
            color=RUNNING,
        )
    tasks.set_yticks(range(len(demo.calls)), [f"Call {c['call']}" for c in demo.calls])
    tasks.invert_yaxis()
    tasks.set_title(
        "Running calls finish; waiting calls follow the new limit", loc="left", pad=14
    )
    tasks.legend(
        handles=[
            Patch(color=RUNNING, label="Running"),
            Patch(color=WAITING, label="Waiting for a slot"),
        ],
        loc="upper right",
        frameon=False,
    )
    # Round only the count panel to 10 ms. Sub-millisecond handoffs otherwise
    # look like long drops to zero; the call bars above retain exact timestamps.
    changes = defaultdict(int)
    for call in demo.calls:
        changes[round(call["start"], 2)] += 1
        changes[round(call["end"], 2)] -= 1
    times, active, n = [], [], 0
    for t, delta in sorted(changes.items()):
        n += delta
        times.append(t)
        active.append(n)
    times.append(end + 0.1)
    active.append(0)
    count.step(times, active, where="post", color=RUNNING, lw=2, label="Running calls")
    count.step(
        [0, demo.change_time, end + 0.1],
        [3, 1, 1],
        where="post",
        color="#30363b",
        ls="--",
        lw=1.8,
        label="Concurrency limit",
    )
    for ax in (tasks, count):
        ax.axvline(demo.change_time, color=CRITICAL, lw=1.2, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.15)
    count.annotate(
        "Limit changes from 3 to 1",
        xy=(demo.change_time, 3),
        xytext=(demo.change_time + 0.15, 3.7),
        color=CRITICAL,
        fontsize=10,
        arrowprops={"arrowstyle": "-", "color": CRITICAL},
    )
    count.set(
        yticks=[0, 1, 2, 3],
        ylim=(-0.1, 4.1),
        xlim=(0, end + 0.12),
        xlabel="Time since the calls were submitted (seconds)",
        ylabel="Number of calls",
    )
    count.legend(loc="upper right", frameon=False, fontsize=9)
    return fig


def read_run(path):
    """Read completed operations with their attempts, preserving report and operation ids."""
    path = Path(path)
    if path.is_dir():
        path = path / "trace.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    operations = {}
    for event in events:
        if event["event"] == "op_start":
            operations[event["op_id"]] = {
                **event,
                "ready": event["t"],
                "attempts_detail": [],
            }
        elif event["event"] == "attempt":
            operations[event["op_id"]]["attempts_detail"].append(event)
        elif event["event"] == "op_end":
            operations[event["op_id"]].update(event, end=event["t"])
    return events, operations


def plot_prefix_reuse(calls, share_function):
    """Plot stage totals using the student's prefix_hit_rate implementation.

    Differences between cumulative totals assign reused tokens to each call.
    This preserves earlier calls from every stage when computing a stage's share.
    """
    stages = {}
    total_tokens, previous_shared = 0, 0.0
    for i, call in enumerate(calls):
        total_tokens += call.prompt_tokens
        shared = share_function(calls[: i + 1]) * total_tokens
        row = stages.setdefault(call.stage, {"tokens": 0, "shared": 0.0, "calls": 0})
        row["tokens"] += call.prompt_tokens
        row["shared"] += shared - previous_shared
        row["calls"] += 1
        previous_shared = shared
    rows = [(STAGES.get(k, k), v) for k, v in stages.items()]
    rows.append(
        (
            "All calls",
            {"tokens": total_tokens, "shared": previous_shared, "calls": len(calls)},
        )
    )
    fig, ax = plt.subplots(figsize=(10, 0.65 * len(rows) + 1.8), layout="constrained")
    for i, (name, row) in enumerate(rows):
        fraction = row["shared"] / row["tokens"] if row["tokens"] else 0.0
        ax.barh(i, fraction * 100, color=RUNNING, height=0.55)
        ax.barh(
            i, (1 - fraction) * 100, left=fraction * 100, color="#dce4ec", height=0.55
        )
        ax.text(
            102,
            i,
            f"{fraction:.1%} shared · {row['calls']} calls",
            va="center",
            fontsize=10,
        )
    ax.set(
        yticks=range(len(rows)),
        yticklabels=[name for name, _ in rows],
        xlim=(0, 145),
        xticks=[0, 25, 50, 75, 100],
        xlabel="Share of prompt tokens (%)",
    )
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Prompt content that appeared in earlier calls", loc="left", pad=14)
    ax.legend(
        handles=[
            Patch(color=RUNNING, label="Previously seen full prefix blocks"),
            Patch(color="#dce4ec", label="New content or partial blocks"),
        ],
        loc="upper left",
        bbox_to_anchor=(0, -0.15),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    return fig, stages


def run_metrics(directory):
    """Return the comparison metrics, including mean LLM-slot wait from the event log."""
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    _, ops = read_run(directory)
    llm = [op for op in ops.values() if op["kind"] == "llm" and op["attempts_detail"]]
    # Older traces lack slot markers; their estimate also includes prompt preparation.
    waits = [
        (
            (op["slot_acquired"] - op["slot_requested"])
            if op.get("slot_acquired") is not None
            and op.get("slot_requested") is not None
            else op["attempts_detail"][0]["start"] - op["ready"]
        )
        for op in llm
    ]
    return {
        **summary,
        "mean_slot_wait": sum(waits) / len(waits) if waits else 0.0,
        "llm_calls": len(llm),
        "total_slot_wait": sum(waits),
    }


def plot_run_comparison(directories):
    """Compare latency, makespan, and per-call slot waiting with units and bar labels."""
    rows = [run_metrics(path) for path in directories]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for ax, (key, title) in zip(
        axes,
        [
            ("mean_latency", "Mean report latency"),
            ("makespan", "Time to finish all reports"),
            ("mean_slot_wait", "Mean wait for an LLM slot"),
        ],
    ):
        bars = ax.bar(
            [f"C = {r['cap']}" for r in rows],
            [r[key] for r in rows],
            color=[RUNNING, "#23856b"][: len(rows)],
            width=0.55,
        )
        ax.bar_label(
            bars,
            labels=[
                f"{r[key]:.3f} s" if r[key] < 0.1 else f"{r[key]:.1f} s" for r in rows
            ],
            padding=4,
        )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Seconds")
        ax.set_ylim(0, max((r[key] for r in rows), default=1) * 1.2 + 0.1)
        ax.spines[["top", "right"]].set_visible(False)
    return fig, rows


def plot_run_timelines(directories):
    """Compare report timelines on the same time axis, with one shared legend."""
    from agentkv.viz import COLORS, run_timeline

    fig, axes = plt.subplots(
        1, len(directories), figsize=(15, 7), sharex=True, sharey=True, squeeze=False
    )
    end = 0
    for ax, directory in zip(axes[0], directories):
        metrics = run_metrics(directory)
        run_timeline(
            Path(directory) / "trace.jsonl",
            ax=ax,
            title=f"Concurrency limit C = {metrics['cap']}",
        )
        ax.get_legend().remove()
        ax.set_xlabel("Time since the first report arrived (seconds)")
        ax.set_ylabel("Report")
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        end = max(end, metrics["makespan"])
    for ax in axes[0]:
        ax.set_xlim(0, end * 1.03)
        ax.set_ylim(len(ax.get_yticks()) - 0.5, -0.5)
    fig.legend(
        handles=[
            Patch(color=COLORS["slot"], label="Waiting for an LLM slot"),
            Patch(color=COLORS["running"], label="LLM call in progress"),
            Patch(color=COLORS["tool"], label="Search or fetch"),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_research_dag(run_path, report_id="R00"):
    """Draw one report's DAG and highlight the critical path from analysis.critical_path."""
    from .analysis import critical_path, load_run

    _, all_ops = read_run(run_path)
    nodes = {
        key: op
        for key, op in all_ops.items()
        if op["report_id"] == report_id and "end" in op
    }
    if not nodes:
        raise ValueError(f"no completed operations for report {report_id}")
    run = load_run(run_path)
    steps = critical_path(run, report_id)
    path = [step.op_id for step in steps]
    wait = {key: run.ops[key].slot_wait for key in nodes}
    service = {key: run.ops[key].end - run.ops[key].start - wait[key] for key in nodes}
    depth = {}

    def level(key):
        if key not in depth:
            depth[key] = 1 + max(
                (level(d) for d in nodes[key]["deps"] if d in nodes), default=-1
            )
        return depth[key]

    for key in nodes:
        level(key)
    # Remove redundant display edges, such as plan -> write when plan already
    # reaches write through an extraction. This does not change the trace DAG.
    ancestors = {}

    def prior(key):
        if key not in ancestors:
            ancestors[key] = set(nodes[key]["deps"])
            for dep in nodes[key]["deps"]:
                ancestors[key].update(prior(dep))
        return ancestors[key]

    edges = {
        key: [
            dep
            for dep in op["deps"]
            if not any(dep in prior(other) for other in op["deps"] if other != dep)
        ]
        for key, op in nodes.items()
    }
    tags = sorted(
        {key.split("/")[1] for key in nodes if key.split("/")[1].startswith("f")}
    )
    centers = {tag: -4.2 * i for i, tag in enumerate(tags)}
    middle = sum(centers.values()) / max(len(centers), 1)
    positions = {}
    for key, op in nodes.items():
        tag = key.split("/")[1]
        y = centers.get(tag, middle)
        if op["stage"] == "fetch":
            peers = sorted(
                k
                for k in nodes
                if k.split("/")[1] == tag and nodes[k]["stage"] == "fetch"
            )
            y += ((len(peers) - 1) / 2 - peers.index(key)) * 1.2
        positions[key] = (depth[key], y)
    fig, ax = plt.subplots(
        figsize=(max(12, 2.1 * (max(depth.values()) + 1)), max(4, 2.5 * len(tags))),
        layout="constrained",
    )
    for tag, center in centers.items():
        group = [positions[key][0] for key in nodes if key.split("/")[1] == tag]
        title = "Follow-up" if tag == "followup" else "Subtopic " + tag[1:]
        ax.text(
            (min(group) + max(group)) / 2,
            center + 1.85,
            title,
            ha="center",
            fontsize=10,
            color="#5a626a",
        )
    boxes = {}
    for key, (x, y) in positions.items():
        node = nodes[key]
        label = STAGES.get(node["stage"], node["stage"])
        if node["stage"] == "fetch":
            label += " " + key.rsplit("fetch", 1)[-1]
        label += f"\n{service[key]:.2f} s" + (
            f" + {wait[key]:.1f} s wait" if wait[key] >= 0.05 else ""
        )
        boxes[key] = ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=10,
            zorder=3,
            bbox={
                "boxstyle": "round,pad=.45",
                "fc": "#edf3fa" if node["kind"] == "llm" else "#f4f4f4",
                "ec": CRITICAL if key in path else "#b8bec5",
                "lw": 1.6 if key in path else 0.8,
            },
        )
    critical_edges = set(zip(path, path[1:]))
    for key, deps in edges.items():
        for dep in deps:
            critical = (dep, key) in critical_edges
            ax.annotate(
                "",
                xy=positions[key],
                xytext=positions[dep],
                zorder=1,
                arrowprops={
                    "arrowstyle": "->",
                    "color": CRITICAL if critical else "#aeb5bc",
                    "lw": 2 if critical else 1,
                    "shrinkA": 3,
                    "shrinkB": 3,
                    "patchA": boxes[dep].get_bbox_patch(),
                    "patchB": boxes[key].get_bbox_patch(),
                },
            )
    ax.set_xlim(-0.6, max(depth.values()) + 0.6)
    ax.set_ylim(
        min(y for _, y in positions.values()) - 1,
        max(y for _, y in positions.values()) + 1.6,
    )
    ax.axis("off")
    ax.set_title(f"Dependencies for report {report_id}", loc="left", pad=12)
    total = {
        k: sum(getattr(step, k) for step in steps) for k in ("gap", "wait", "service")
    }
    fig.text(
        0.5,
        0.01,
        f"Red: critical path. Latency {sum(total.values()):.2f} s = service {total['service']:.2f} s "
        f"+ slot wait {total['wait']:.2f} s + gaps {total['gap']:.2f} s.",
        ha="center",
        fontsize=10,
        color=CRITICAL,
    )
    return fig, steps


def plot_critical_paths(run_dirs):
    """Stacked bars per report: where each report's latency went along its critical path."""
    from .analysis import critical_path, load_run

    parts = [
        ("wait", "Waiting for an LLM slot", WAITING),
        ("llm", "LLM service", RUNNING),
        ("tool", "Tool service", TOOL),
        ("gap", "Gaps between operations", "#d9dde1"),
    ]
    fig, axes = plt.subplots(
        1,
        len(run_dirs),
        figsize=(6 * len(run_dirs), 5.5),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    totals = []
    for ax, directory in zip(axes[0], run_dirs):
        run = load_run(directory)
        reports = sorted(run.arrivals)
        total = dict.fromkeys(("wait", "llm", "tool", "gap"), 0.0)
        for i, report in enumerate(reports):
            steps = critical_path(run, report)
            v = {
                "wait": sum(s.wait for s in steps),
                "gap": sum(s.gap for s in steps),
                "llm": sum(s.service for s in steps if s.kind == "llm"),
                "tool": sum(s.service for s in steps if s.kind != "llm"),
            }
            left = 0.0
            for key, _, color in parts:
                ax.barh(i, v[key], left=left, color=color, height=0.7)
                left += v[key]
                total[key] += v[key]
        name = f"{Path(directory).name} (C = {run.cap})"
        totals.append(
            {
                "run": name,
                "cap": run.cap,
                **{k: v / len(reports) for k, v in total.items()},
            }
        )
        ax.set_yticks(range(len(reports)), reports)
        ax.invert_yaxis()
        ax.set_title(name, loc="left")
        ax.set_xlabel("Seconds on the critical path")
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(
        handles=[Patch(color=c, label=label) for _, label, c in parts],
        loc="outside upper center",
        ncol=4,
        frameon=False,
    )
    return fig, totals


def plot_reuse_intervals(run_dirs):
    """Cumulative distribution of how long reused prefix blocks sat unused, by the reusing stage."""
    from .analysis import load_run, reuse_intervals

    fig, axes = plt.subplots(
        1,
        len(run_dirs),
        figsize=(6 * len(run_dirs), 3.8),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for ax, directory in zip(axes[0], run_dirs):
        run = load_run(directory)
        by_stage = defaultdict(list)
        for r in reuse_intervals(run.llm_calls()):
            by_stage[r.stage].append(r.interval)
        for stage, gaps in sorted(by_stage.items()):
            gaps.sort()
            ax.step(
                gaps,
                [(i + 1) / len(gaps) for i in range(len(gaps))],
                where="post",
                label=f"{STAGES.get(stage, stage)} ({len(gaps)} reuses)",
            )
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set(
            xlabel="Seconds since the block was last held",
            ylabel="Fraction of reuses",
            ylim=(0, 1.02),
            title=f"{Path(directory).name} (C = {run.cap})",
        )
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_kv_footprint(run_dirs):
    """KV-cache memory held by running LLM calls, with and without sharing identical prefix blocks."""
    from .analysis import kv_footprint, load_run

    fig, ax = plt.subplots(figsize=(11, 3.8), layout="constrained")
    peaks = []
    for color, directory in zip([RUNNING, "#23856b", "#8e5ea2", WAITING], run_dirs):
        run = load_run(directory)
        fp = kv_footprint(run.llm_calls())
        t = [f.t for f in fp]
        name = Path(directory).name
        ax.step(
            t,
            [f.private / 1e9 for f in fp],
            where="post",
            color=color,
            lw=1,
            alpha=0.45,
            label=f"{name}, every call stores its own prompt",
        )
        ax.step(
            t,
            [f.shared / 1e9 for f in fp],
            where="post",
            color=color,
            lw=1.8,
            label=f"{name}, identical prefix blocks stored once",
        )
        peaks.append(
            {
                "run": name,
                "cap": run.cap,
                "calls": max((f.calls for f in fp), default=0),
                "private_gb": max((f.private for f in fp), default=0) / 1e9,
                "shared_gb": max((f.shared for f in fp), default=0) / 1e9,
            }
        )
    ax.set(
        xlabel="Time since the run started (s)", ylabel="KV cache of running calls (GB)"
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1, 1))
    ax.spines[["top", "right"]].set_visible(False)
    return fig, peaks
