"""Readable summaries of policy measurements for the CLI and notebooks."""

from collections import Counter
import math
from pathlib import Path

from . import metric

FAMILY_NAMES = {
    "D": "Deep research",
    "W1": "Long-context chat",
    "W2": "Coding agent",
    "W3": "Document QA",
    "W4": "Customer-service agents",
    "W5": "Reasoning chat",
    "M1": "Document QA + chat",
    "M2": "Document QA + reasoning",
}


def _table(headers, rows):
    rows = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    return [
        "  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip()
        for row in [headers, ["-" * width for width in widths], *rows]
    ]


def _seconds(value):
    return f"{value:.4f}" if math.isfinite(value) else "incomplete"


def _count(number, noun):
    return f"{number} {noun}" + ("" if number == 1 else "s")


def score_summary(result, *, details=False):
    """Explain a ScoreResult without changing its measurements or grading metric."""
    traces = list(result.by_trace.values())
    families = Counter(t.family for t in traces)
    jobs = sum(len(t.flows) for t in traces)
    completed = sum(math.isfinite(value) for t in traces for value in t.flows)
    policy = Path(result.policy)
    try:
        policy = policy.relative_to(Path(__file__).resolve().parents[1])
    except ValueError:
        pass
    lines = [
        f"Policy: {policy} (SHA-256: {result.policy_sha256[:12]})",
        f"Workload: {result.split}",
        f"Coverage: {_count(len(traces), 'trace')}, {_count(jobs, 'job')}, "
        f"{len(families)} of 8 workload categories",
    ]
    settings = next((t.diag for t in traces if "replicas" in t.diag), {})
    replicas = settings.get("replicas")
    profile = settings.get("profile", {})
    if result.backend == "sim":
        lines.append(
            "Engine: Vidur CPU simulation; GPU times are predictions from measured profiles"
        )
        if profile and replicas is not None:
            lines.append(
                f"Configuration: {profile['model'].split('/')[-1]}, {_count(replicas, 'independent replica')}, "
                f"one {profile['device']} per replica"
            )
        quotas = sorted(
            {
                t.diag["kv_tokens_per_replica"]
                for t in traces
                if "kv_tokens_per_replica" in t.diag
            }
        )
        if quotas:
            quota = (
                f"{quotas[0]:,}"
                if len(quotas) == 1
                else f"{quotas[0]:,}–{quotas[-1]:,}"
            )
            lines.append(
                f"KV cache: {settings['block_size']}-token blocks; {quota} tokens per replica"
                + (" (quota varies by trace)" if len(quotas) > 1 else "")
            )
        lines.append(
            "Reference: fixed staff LRU, concurrency limit 8, same workload and hardware"
        )
    else:
        lines.extend(
            [f"Engine: {result.backend}", "Reference: fixed baseline for this backend"]
        )
    lines.extend(
        [
            "",
            f"Status: {'VALID' if result.valid else 'INVALID'} — {completed}/{jobs} jobs completed",
            "",
            "Mean job latency: arrival to completion, including queueing and tool waits.",
        ]
    )
    rows = []
    for t in traces:
        valid = t.valid and math.isfinite(t.mean_flow)
        rows.append(
            [
                t.trace_id,
                t.family,
                _seconds(t.mean_flow),
                _seconds(t.b0_mean_flow),
                f"{t.ratio:.4f}×" if valid else "INVALID",
            ]
        )
    if rows:
        lines.extend(
            _table(
                ["Trace", "Category", "Policy (s)", "Reference (s)", "Speedup"], rows
            )
        )
    if result.valid:
        lines.extend(["", "Speedup by workload category:"])
        for family, speedup in result.by_family.items():
            lines.append(
                f"  {FAMILY_NAMES.get(family, family)} ({family}): {speedup:.4f}× "
                f"across {_count(families[family], 'trace')}"
            )
        lines.append(f"Overall speedup (S): {result.speedup:.4f}×")
        direction = "lower" if result.g >= 0 else "higher"
        lines.append(
            f"Equivalent latency change: {abs(100 * result.g):.2f}% {direction}"
        )
        lines.append(
            "Speedup = reference / policy job latency; 1.0000× matches the reference, higher is better."
        )
        if len(traces) > 1:
            lines.append(
                "Category scores use geometric means; the overall score weights each selected category equally."
            )
    else:
        lines.extend(
            ["", "No aggregate speedup: every trace must finish with valid results."]
        )
        lines.extend(f"  {message}" for message in result.violations)

    diagnostics = [t.diag for t in traces if "prompt_tokens" in t.diag]
    if diagnostics:
        total = lambda key: sum(d.get(key, 0) for d in diagnostics)
        prompt, hits = total("prompt_tokens"), total("cache_hit_tokens")
        reuse = f"{100 * hits / prompt:.1f}%" if prompt else "n/a"
        lines.extend(
            [
                "",
                "Cache and scheduling totals"
                + (":" if result.valid else " (partial results):"),
                f"  Prompt tokens reused from cache: {hits:,} / {prompt:,} ({reuse})",
                f"  Prompt tokens computed: {total('prefill_tokens'):,} (includes recomputation)",
                f"  Tokens recomputed after preemption: {total('recomputed_tokens'):,}",
                f"  Preemptions: {total('preemptions'):,} (calls interrupted to free KV memory)",
                f"  Cached blocks evicted: {total('evictions'):,}",
                f"  Policy callback CPU time: {1000 * total('policy_cpu_s'):.1f} ms",
                "  CPU time is measured on the host and is excluded from simulated job latency.",
            ]
        )
        if details and len(traces) > 1:
            rows = []
            for t in traces:
                d = t.diag
                if "prompt_tokens" not in d:
                    rows.append([t.trace_id, "unavailable", "—", "—", "—", "—"])
                    continue
                hit = 100 * d.get("cache_hit_tokens", 0) / max(d["prompt_tokens"], 1)
                rows.append(
                    [
                        t.trace_id,
                        f"{hit:.1f}%",
                        f"{d.get('recomputed_tokens', 0):,}",
                        f"{d.get('preemptions', 0):,}",
                        f"{d.get('evictions', 0):,}",
                        f"{1000 * d.get('policy_cpu_s', 0):.1f}",
                    ]
                )
            lines.extend(["", "Per-trace diagnostics:"])
            lines.extend(
                _table(
                    [
                        "Trace",
                        "Prompt reuse",
                        "Recomputed tokens",
                        "Preemptions",
                        "Block evictions",
                        "Policy CPU (ms)",
                    ],
                    rows,
                )
            )

    lines.append("")
    per_family = {"public_dev": 4, "public_val": 2}.get(result.split)
    full_split = (
        per_family is not None
        and set(families) == set(metric.FAMILY_ORDER)
        and all(count == per_family for count in families.values())
    )
    if result.valid and full_split and result.backend == "sim" and replicas == 2:
        top = metric.MAX_POINTS
        lines.append(
            f"Projected performance credit on this split: {result.points:.1f}% of {top:g}%"
        )
        lines.append(
            f"The formula gives {metric.PARITY_POINTS:g}% at parity and {top:g}% for a "
            f"{100 * result.g_target:g}% reduction in job latency (S = {1 / (1 - result.g_target):.4f})."
        )
        if result.capped:
            lines.append(
                f"Credit is reduced: categories below {metric.FAMILY_FLOOR:.2f}× subtract their shortfall from g."
            )
        lines.append("Final performance grading uses the hidden traces.")
    elif not result.valid:
        lines.append("Invalid or incomplete evaluations receive no performance credit.")
    elif replicas == 1:
        lines.append(
            "This is a one-replica experiment. Part 3 scoring uses two replicas and all eight categories."
        )
    else:
        lines.append(
            "Scope: this result covers only the selected traces; projected credit is not shown."
        )
        lines.append(
            "Evaluate the full public_dev or public_val split to compare all eight categories."
        )
    return "\n".join(lines)
