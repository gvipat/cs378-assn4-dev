"""A starter agentic research system that generates policies, evaluates them, and
retains the best candidate. See Part 3 in README.md.

    export AGENTKV_RESEARCH_MODEL=<the course's approved Bedrock model id>
    python examples/research_system.py --iters 10

    python examples/research_system.py --iters 3 --offline     # check execution without model calls
    python examples/research_system.py --evaluator research/evaluator.py --iters 10

Complete policies/reference.py in Part 2 before starting the campaign. For a
plumbing check before then, use --offline --start policies/fifo.py.

Every candidate is written to --out and logged as one JSON line in log.jsonl
(id, parent, hypothesis, file, sha256, objective, dev_S, by_family, valid, tokens), the
format the Part 3 notebook plots. Per-call usage, prompts, and replies are
retained beside it. Extend this example with your own budget, selection,
and failure-handling rules, and record those rules before a campaign.
"""

import argparse, hashlib, json, math, os, pathlib, random, re, runpy, time

from agentkv import score


def bedrock_client():
    import boto3
    from botocore.config import Config

    # Long replies can take minutes; adaptive retries absorb throttling.
    return boto3.client(
        "bedrock-runtime",
        config=Config(
            read_timeout=900, retries={"max_attempts": 8, "mode": "adaptive"}
        ),
    )


def ask(bedrock, model, prompt):
    """Return (reply text, usage). A failed request returns an empty reply and records the error."""
    try:
        r = bedrock.converse(
            modelId=model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 32000},
        )
    except Exception as e:
        return "", {
            "input_tokens": 0,
            "output_tokens": 0,
            "model": model,
            "error": f"{type(e).__name__}: {e}",
        }
    usage = r["usage"]
    text = "".join(b.get("text", "") for b in r["output"]["message"]["content"])
    return text, {
        "input_tokens": usage["inputTokens"],
        "output_tokens": usage["outputTokens"],
        "model": model,
        "request_id": r.get("ResponseMetadata", {}).get("RequestId"),
    }


def ask_offline(best_code, rng):
    """Generate a candidate with a fixed concurrency cap to test the example without API calls."""
    cap = rng.randint(1, 16)
    code, n = re.subn(
        r"(    def target_concurrency\(self, view, state\):\n).*?(?=\n    (?:def |@)|\nPOLICY)",
        rf"\g<1>        return {cap}\n",
        best_code,
        count=1,
        flags=re.S,
    )
    hypothesis = (
        f"a fixed cap of {cap} suits this mix better"
        if n
        else "no target_concurrency found"
    )
    return f"{hypothesis}\n```python\n{code}```", {
        "input_tokens": 0,
        "output_tokens": 0,
        "model": "offline",
    }


def brief(r):
    fams = ", ".join(f"{f} {s:.3f}" for f, s in r["by_family"].items())
    objective = f"objective={r['objective']} " if "objective" in r else ""
    return f"valid={r['valid']} {objective}S={r['speedup']:.4f} by_family=[{fams}] violations={r['violations'][:2]}"


def hypothesis_of(reply):
    """The first non-empty line of the reply, without Markdown emphasis or a 'Hypothesis:' label."""
    for line in reply.splitlines():
        line = re.sub(
            r"^hypothesis\b\s*:?\s*", "", line.strip().strip("*#> ").strip(), flags=re.I
        ).strip("* ")
        if line:
            return line[:200]
    return "(empty reply)"


def load_evaluator(path):
    """Load the student's evaluate(policy_path, *, split) function."""
    evaluate = runpy.run_path(str(path)).get("evaluate")
    if not callable(evaluate):
        raise ValueError(f"{path} must define evaluate(policy_path, *, split)")
    return evaluate


def evaluate_candidate(evaluate, policy_path):
    """Run the development evaluator and check the result needed for selection."""
    result = evaluate(str(policy_path), split="public_dev")
    required = {"valid", "violations", "speedup", "by_family", "objective"}
    if not isinstance(result, dict) or not required <= result.keys():
        raise ValueError(
            "The evaluator must return the score dictionary plus an objective field"
        )
    if type(result["valid"]) is not bool:
        raise ValueError("The evaluator's valid field must be a boolean")
    if result["valid"]:
        objective = result["objective"]
        if (
            isinstance(objective, bool)
            or not isinstance(objective, (int, float))
            or not math.isfinite(objective)
        ):
            raise ValueError("A valid candidate must have a finite numeric objective")
    else:
        result["objective"] = None
    return result


def main():  # score() runs traces in a process pool, so everything lives under main()
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument(
        "--start",
        default="policies/reference.py",
        help="initial policy; your completed Part 2 reference by default",
    )
    ap.add_argument(
        "--evaluator",
        default="examples/evaluator.py",
        help="Python file defining evaluate(policy_path, *, split); see examples/evaluator.py",
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="no LLM calls: a stand-in that only changes the cap proposes candidates",
    )
    ap.add_argument("--out", default=f"runs/example-{time.strftime('%Y%m%d-%H%M%S')}")
    args = ap.parse_args()
    evaluator_source = pathlib.Path(args.evaluator).read_text()
    evaluate = load_evaluator(args.evaluator)

    rng, bedrock, model = random.Random(0), None, None
    if not args.offline:
        model = os.environ["AGENTKV_RESEARCH_MODEL"]
        bedrock = bedrock_client()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=False)  # do not mix two campaigns' records
    (out / "evaluator.py").write_text(evaluator_source)
    (out / "campaign.json").write_text(
        json.dumps(
            {
                "started_at": time.time(),
                "model": model if model else "offline",
                "iterations": args.iters,
                "starting_policy": args.start,
                "starting_sha256": hashlib.sha256(
                    pathlib.Path(args.start).read_bytes()
                ).hexdigest(),
                "evaluator": args.evaluator,
                "evaluator_sha256": hashlib.sha256(
                    evaluator_source.encode()
                ).hexdigest(),
                "selection": "highest valid development objective among generated candidates; score all validation categories at the end",
                "stopping": "fixed iteration count",
                "offline": args.offline,
            },
            indent=2,
        )
        + "\n"
    )
    api = pathlib.Path("agentkv/policy.py").read_text()

    best_code, best_id = pathlib.Path(args.start).read_text(), "start"
    best = evaluate_candidate(evaluate, args.start)
    (out / "start.scores.json").write_text(json.dumps(best, indent=2) + "\n")
    if not best["valid"]:
        raise ValueError(
            "Starting policy is invalid. Complete Part 2 or use --start policies/fifo.py. "
            + "; ".join(best["violations"][:2])
        )
    history = [f"start ({args.start}, its default parameters): {brief(best)}"]
    selected, selected_id = None, None
    print(history[0])
    for i in range(args.iters):
        prompt = (
            "You are improving a KV-cache serving policy: eviction, concurrency, and the scheduling hooks.\n"
            f"The API:\n```python\n{api}\n```\n"
            "The research evaluator defines the objective to maximize. Invalid candidates cannot be selected.\n"
            f"```python\n{evaluator_source}\n```\n"
            f"Current best policy:\n```python\n{best_code}\n```\n"
            "History (most recent last):\n" + "\n".join(history[-12:]) + "\n"
            "State one hypothesis in one line, then give the complete new policy file "
            "in one ```python block."
        )
        (out / f"p{i:02d}.prompt.txt").write_text(prompt)
        reply, usage = (
            ask_offline(best_code, rng) if args.offline else ask(bedrock, model, prompt)
        )
        (out / f"p{i:02d}.reply.txt").write_text(reply)
        with (out / "calls.jsonl").open("a") as f:
            f.write(
                json.dumps({"candidate": f"p{i:02d}", "time": time.time(), **usage})
                + "\n"
            )
        tokens = usage["input_tokens"] + usage["output_tokens"]
        m = re.search(r"```python\n(.*?)```", reply, re.S)
        path = out / f"p{i:02d}.py"
        path.write_text(m.group(1) if m else "# no code returned\n")
        r = evaluate_candidate(evaluate, path)
        scores_path = out / f"p{i:02d}.scores.json"
        scores_path.write_text(json.dumps(r, indent=2) + "\n")
        hypothesis = (
            hypothesis_of(reply)
            if reply
            else f"(model call failed: {usage.get('error', '')[:150]})"
        )
        history.append(f"p{i:02d}: {hypothesis} -> {brief(r)}")
        row = {
            "id": f"p{i:02d}",
            "parent": best_id,
            "hypothesis": hypothesis,
            "file": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "dev_S": r["speedup"],
            "objective": r["objective"],
            "scores_file": str(scores_path),
            "by_family": r["by_family"],
            "valid": r["valid"],
            "tokens": tokens,
            **usage,
        }
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        if r["valid"] and (not best["valid"] or r["objective"] > best["objective"]):
            best, best_code, best_id = r, path.read_text(), f"p{i:02d}"
        if r["valid"] and (selected is None or r["objective"] > selected["objective"]):
            selected, selected_id = r, f"p{i:02d}"
            (out / "best.py").write_text(path.read_text())
        print(history[-1])

    if selected_id is None:
        print(
            "No valid generated candidate; inspect the failures and revise the system before another campaign."
        )
        return
    final = out / "best.py"
    validation = score(str(final), split="public_val")
    (out / "validation.scores.json").write_text(
        json.dumps(validation.to_json(), indent=2) + "\n"
    )
    (out / "selection.json").write_text(
        json.dumps(
            {
                "candidate": selected_id,
                "sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
                "dev_objective": selected["objective"],
                "dev_S": selected["speedup"],
                "val_S": validation.speedup,
                "valid": validation.valid,
                "by_family": validation.by_family,
            },
            indent=2,
        )
        + "\n"
    )
    print("validation of the selected candidate:", brief(validation.to_json()))


if __name__ == "__main__":
    main()
