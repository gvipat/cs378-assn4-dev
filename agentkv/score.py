"""Score a candidate policy against the baseline.

    from agentkv import score
    r = score("candidates/p17.py", split="public_dev")
    r.valid, r.violations, r.speedup, r.points, r.by_family, r.by_trace

    python -m agentkv.score --policy candidates/p17.py --split public_dev

Scoring runs Vidur locally with two independent replicas by default.
Part 2 compares one and two replicas; Part 2.3 and Part 3 use two.
Pass replicas=1 for single-replica experiments. Traces run in worker processes;
AGENTKV_WORKERS overrides the worker count. Call score() from a notebook, from
a script guarded by `if __name__ == "__main__":`, or with workers=1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from . import metric
from .replay import RunResult
from .sandbox.loader import policy_sha256
from .tokenizer import single_threaded
from .trace import Trace, load

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
VIDUR_REFERENCE_PATH = PACKAGE / "data" / "vidur_reference.json"
VIDUR_POLICY_PATH = REPO / "policies" / "reference.py"
PUBLIC_SPLITS = ("public_dev", "public_val")
STAFF_SPLITS = ("hidden_feedback", "sealed")


class ScoreError(RuntimeError):
    pass


# ------------------------------------------------------------------ results


@dataclass
class TraceResult:
    trace_id: str
    family: str
    valid: bool
    violation: str | None
    mean_flow: float
    b0_mean_flow: float
    ratio: float  # L(baseline) / L(policy); > 1 is faster than the baseline
    flows: list[float]
    b0_flows: list[float]
    diag: dict = field(default_factory=dict)


@dataclass
class ScoreResult:
    policy: str
    policy_sha256: str
    split: str
    backend: str
    valid: bool
    violations: list[str]
    speedup: float
    g: float
    g_eff: float
    points: float
    capped: bool
    by_family: dict[str, float]
    by_trace: dict[str, TraceResult]
    g_target: float
    sim_fingerprint: str
    b0_sha256: str | None
    reps: int = 1

    def to_json(self) -> dict:
        d = asdict(self)
        return json.loads(json.dumps(d, default=str, allow_nan=True))

    def summary(self, *, details: bool = False) -> str:
        """Explain the results; details=True also lists cache statistics for each trace."""
        from .reporting import score_summary

        return score_summary(self, details=details)


# ------------------------------------------------------------------ splits


def split_paths(split) -> list[Path]:
    """Resolve a split name, trace directory, trace file, or file list into paths to evaluate.

    "screen" is one public_dev trace per family, for quick checks before a full evaluation.
    """
    if isinstance(split, (list, tuple)):
        return [Path(p) for p in split]
    p = Path(split)
    if p.suffix == ".json" and p.exists():
        return [p]
    if p.is_dir():
        return sorted(p.glob("*.json"))
    base = Path(os.environ.get("AGENTKV_TRACES", REPO / "data" / "traces"))
    if split == "screen":
        return sorted((base / "public_dev").glob("dev-*-0.json"))
    if split in PUBLIC_SPLITS:
        d = base / split
    elif split in STAFF_SPLITS:
        d = (
            Path(os.environ.get("AGENTKV_STAFF_TRACES", REPO / "grader" / "traces"))
            / split
        )
    else:
        raise ScoreError(
            f"unknown split {split!r}; use screen, {', '.join(PUBLIC_SPLITS)}, or a path"
        )
    paths = sorted(d.glob("*.json"))
    if not paths:
        raise ScoreError(f"no traces found in {d}")
    return paths


@lru_cache(maxsize=64)
def _load(path: str) -> Trace:
    return load(path)


# ------------------------------------------------------------------ running


def _run(task: dict) -> RunResult:
    from .vidur_replay import run_policy

    return run_policy(
        _load(task["trace"]),
        task["policy"],
        params=task["params"],
        seed=task["seed"],
        isolate=task["isolate"],
        replicas=task["replicas"],
    )


def default_workers(n_tasks: int) -> int:
    """Worker processes for scoring: AGENTKV_WORKERS, or all but one CPU (at most 12)."""
    env = os.environ.get("AGENTKV_WORKERS")
    if env:
        return max(1, min(int(env), n_tasks))
    return max(1, min(n_tasks, (os.cpu_count() or 2) - 1, 12))


def _spawn_safe() -> bool:
    """Spawned workers re-import the caller's main module; that fails for `python - <<EOF`."""
    path = getattr(sys.modules.get("__main__"), "__file__", None)
    return path is None or os.path.exists(path)


def _run_all(tasks: list[dict], workers: int | None) -> list[RunResult]:
    workers = workers or default_workers(len(tasks))
    if workers > 1 and len(tasks) > 1 and not _spawn_safe():
        print(
            "agentkv.score: the main script is not a file (e.g. stdin), so traces run in one process; "
            "save the script to a .py file to score in parallel",
            file=sys.stderr,
        )
        workers = 1
    if workers <= 1 or len(tasks) == 1:
        return [_run(t) for t in tasks]
    import multiprocessing as mp

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=single_threaded,
    ) as ex:
        return list(ex.map(_run, tasks))


def load_reference(path: Path = VIDUR_REFERENCE_PATH) -> dict:
    if not path.exists():
        return {"traces": {}}
    return json.loads(path.read_text())


def evaluate(
    policy_path: str,
    trace_paths: list,
    backend: str = "sim",
    *,
    b0: str = "reference",
    reps: int | None = None,
    params: dict | None = None,
    seed: int = 0,
    isolate: bool = True,
    workers: int | None = None,
    g_target: float | None = None,
    split: str = "",
    replicas: int = 2,
) -> ScoreResult:
    """Score a policy against reference job latencies.

    b0="reference" uses frozen staff results. b0="live" reruns the local
    policies/reference.py, which must be implemented first.
    """
    from .engine.vidur_profile import fingerprint as vidur_fingerprint

    if backend != "sim":
        raise ScoreError(
            f"unknown backend {backend!r}; the evaluator is backend='sim' (Vidur)"
        )
    if replicas not in (1, 2):
        raise ScoreError("replicas must be 1 or 2")
    policy_path = str(Path(policy_path).resolve())
    paths = [str(Path(p).resolve()) for p in trace_paths]
    fingerprint = vidur_fingerprint(replicas)
    baseline_path = VIDUR_POLICY_PATH
    g_target = g_target if g_target is not None else metric.G_TARGET
    reps = reps or 1
    traces = {p: _load(p) for p in paths}

    ref = load_reference().get("configurations", {}).get(str(replicas), {})
    b0_runs: dict[str, list[dict]] = {}
    if b0 == "reference":
        if ref.get("sim") != fingerprint:
            raise ScoreError(
                "Baseline reference was computed with a different simulator; update agentkv"
            )
        for p, t in traces.items():
            r = ref["traces"].get(t.sha256)
            if r is None:
                raise ScoreError(
                    f"no baseline reference for {t.trace_id} ({Path(p).name}); baseline reference "
                    f"job latencies exist only for the published splits"
                )
            b0_runs[p] = [
                {"flows": r["flows"], "violation": None, "diag": r.get("diag", {})}
            ]
        b0_sha = ref.get("b0_sha256")
    else:
        if not baseline_path.exists():
            raise ScoreError(
                "Live baseline evaluation requires staff code; use b0='reference'"
            )
        b0_sha = policy_sha256(baseline_path)

    tasks, owners = [], []
    for p in paths:
        for rep in range(reps):
            pair = [("cand", policy_path, params)]
            if b0 == "live":
                pair.append(("b0", str(baseline_path), None))
            for who, pol, prm in pair:
                tasks.append(
                    {
                        "trace": p,
                        "policy": pol,
                        "params": prm,
                        "seed": seed + rep,
                        "isolate": isolate,
                        "replicas": replicas,
                    }
                )
                owners.append((p, who))
    results = _run_all(tasks, workers)

    cand_runs: dict[str, list[RunResult]] = {}
    for (p, who), r in zip(owners, results):
        if who == "cand":
            cand_runs.setdefault(p, []).append(r)
        else:
            b0_runs.setdefault(p, []).append(
                {"flows": r.flows, "violation": r.violation, "diag": r.diag}
            )

    by_trace: dict[str, TraceResult] = {}
    violations: list[str] = []
    rows = []
    for p in paths:
        t = traces[p]
        runs, bruns = cand_runs[p], b0_runs[p]
        for b in bruns:
            if b["violation"]:
                raise ScoreError(f"Baseline failed on {t.trace_id}: {b['violation']}")
        bad = next((r.violation for r in runs if r.violation), None)
        means = [r.mean_flow for r in runs]
        bmeans = [statistics.fmean(b["flows"]) for b in bruns]
        if bad is None and all(math.isfinite(m) for m in means):
            ratio = statistics.median(
                bm / m for bm, m in zip(bmeans * (len(means) // len(bmeans)), means)
            )
        else:
            ratio = 0.0
            violations.append(f"{t.trace_id}: {bad or 'incomplete jobs'}")
        by_trace[t.trace_id] = TraceResult(
            t.trace_id,
            t.family,
            bad is None,
            bad,
            statistics.median(means),
            statistics.median(bmeans),
            ratio,
            runs[0].flows,
            bruns[0]["flows"],
            runs[0].diag,
        )
        if bad is None:
            rows.append(
                {
                    "family": t.family,
                    "ratio": ratio,
                    "mean_flow": statistics.median(means),
                    "b0_mean_flow": statistics.median(bmeans),
                }
            )

    valid = not violations
    if valid:
        m = metric.summarize(rows, g_target=g_target)
    else:
        m = {
            "speedup": 0.0,
            "g": -math.inf,
            "g_eff": -math.inf,
            "points": 0.0,
            "by_family": {},
            "capped": False,
        }
    by_trace = dict(
        sorted(
            by_trace.items(), key=lambda kv: (metric.family_key(kv[1].family), kv[0])
        )
    )
    return ScoreResult(
        policy=policy_path,
        policy_sha256=policy_sha256(policy_path),
        split=split or "custom",
        backend=backend,
        valid=valid,
        violations=violations,
        speedup=m["speedup"],
        g=m["g"],
        g_eff=m["g_eff"],
        points=m["points"] if valid else 0.0,
        capped=m["capped"],
        by_family=m["by_family"],
        by_trace=by_trace,
        g_target=g_target,
        sim_fingerprint=fingerprint,
        b0_sha256=b0_sha,
        reps=reps,
    )


def score(
    policy,
    split="public_dev",
    backend: str = "sim",
    *,
    params: dict | None = None,
    seed: int = 0,
    workers: int | None = None,
    isolate: bool = True,
    b0: str = "reference",
    g_target: float | None = None,
    replicas: int = 2,
) -> ScoreResult:
    """Check validity and score a candidate against the baseline using the grading metric."""
    if isinstance(policy, type):
        import inspect

        policy = inspect.getsourcefile(policy)
    if backend != "sim":
        raise ScoreError(
            f"unknown backend {backend!r}; the assignment evaluator is backend='sim' (Vidur)"
        )
    name = split if isinstance(split, str) else "custom"
    return evaluate(
        str(policy),
        split_paths(split),
        backend,
        b0=b0,
        params=params,
        seed=seed,
        isolate=isolate,
        workers=workers,
        g_target=g_target,
        split=name,
        replicas=replicas,
    )


def _from_json(d: dict) -> ScoreResult:
    d = dict(d)
    d["by_trace"] = {k: TraceResult(**v) for k, v in d["by_trace"].items()}
    return ScoreResult(**d)


# ------------------------------------------------------------------ CLI


def parse_params(text: str | None) -> dict | None:
    """Parse parameters from a JSON object or a path to a JSON file.

    Try inline JSON first so a long JSON string is not treated as a filename.
    """
    if not text:
        return None
    if not text.lstrip().startswith("{"):
        text = Path(text).read_text()
    return json.loads(text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m agentkv.score", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("--policy", required=True)
    ap.add_argument(
        "--split",
        default="public_dev",
        help="screen, public_dev, public_val, a directory, or a trace file",
    )
    ap.add_argument("--replicas", type=int, choices=[1, 2], default=2)
    ap.add_argument(
        "--params",
        help="JSON object or path to a JSON file passed to the policy constructor",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--json", help="write the full result as JSON to this path")
    ap.add_argument(
        "--details",
        action="store_true",
        help="also print cache and scheduling statistics for each trace",
    )
    ap.add_argument(
        "--no-isolate",
        action="store_true",
        help="run the policy in-process (debugging only)",
    )
    ap.add_argument(
        "--baseline",
        default="reference",
        choices=["reference", "live"],
        help="reference: saved reference job latencies; live: rerun the reference",
    )
    args = ap.parse_args(argv)
    params = parse_params(args.params)
    r = score(
        args.policy,
        args.split,
        params=params,
        seed=args.seed,
        workers=args.workers,
        isolate=not args.no_isolate,
        b0=args.baseline,
        replicas=args.replicas,
    )
    print(r.summary(details=args.details))
    if args.json:
        Path(args.json).write_text(json.dumps(r.to_json(), indent=1) + "\n")
        print(f"\nFull results saved to {args.json}")
    return 0 if r.valid else 1


class _CallableModule(type(sys)):
    """`from agentkv import score` may bind this module (once it is imported)
    instead of the function; make both spellings work."""

    def __call__(self, *args, **kwargs) -> ScoreResult:
        return score(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule

if __name__ == "__main__":
    sys.exit(main())
