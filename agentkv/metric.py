"""The grading metric.

    L(P, i)  = mean job latency of policy P on trace i
    S_f      = exp( mean over traces i in family f ( log( L(baseline, i) / L(P, i) ) ) )
    S        = exp( mean over families ( log S_f ) )
    g        = 1 - 1/S                                   # fractional latency reduction
    g_eff    = g - sum over families ( max(0, FAMILY_FLOOR - S_f) )
    points   = clip( PARITY_POINTS + (MAX_POINTS - PARITY_POINTS) * g_eff / G_TARGET, 0, MAX_POINTS )

An invalid or incomplete trace gives 0 points. A family slower than
FAMILY_FLOOR costs points in proportion to how far it falls below.
"""

from __future__ import annotations

import math
import statistics

G_TARGET = 0.09  # g that earns full performance points
FULL_CREDIT_SPEEDUP = 1 / (1 - G_TARGET)
FAMILY_FLOOR = 0.90
MAX_POINTS = 30.0
PARITY_POINTS = 10.0  # a valid policy that matches the reference
FAMILY_ORDER = ("D", "W1", "W2", "W3", "W4", "W5", "M1", "M2")


def family_key(f: str) -> tuple:
    return (FAMILY_ORDER.index(f), f) if f in FAMILY_ORDER else (len(FAMILY_ORDER), f)


def speedup(log_ratios: dict[str, list[float]]) -> float:
    """Compute family-balanced speedup from per-trace log ratios grouped by family."""
    fams = [statistics.fmean(v) for v in log_ratios.values() if v]
    return math.exp(statistics.fmean(fams)) if fams else 1.0


def points(g_eff: float, g_target: float = G_TARGET) -> float:
    return min(
        max(PARITY_POINTS + (MAX_POINTS - PARITY_POINTS) * g_eff / g_target, 0.0),
        MAX_POINTS,
    )


def summarize(rows: list[dict], *, g_target: float = G_TARGET) -> dict:
    """Compute the score and performance points from per-trace results.

    Each row supplies family and ratio (baseline latency divided by candidate latency).
    """
    logs: dict[str, list[float]] = {}
    for r in rows:
        logs.setdefault(r["family"], []).append(math.log(r["ratio"]))
    s = speedup(logs)
    g = 1.0 - 1.0 / s
    by_family = {
        f: math.exp(statistics.fmean(v))
        for f, v in sorted(logs.items(), key=lambda kv: family_key(kv[0]))
    }
    shortfall = {f: FAMILY_FLOOR - v for f, v in by_family.items() if v < FAMILY_FLOOR}
    g_eff = g - sum(shortfall.values())
    return {
        "speedup": s,
        "g": g,
        "g_eff": g_eff,
        "points": points(g_eff, g_target),
        "by_family": by_family,
        "capped": bool(shortfall),
        "regressed_families": sorted(shortfall, key=family_key),
    }
