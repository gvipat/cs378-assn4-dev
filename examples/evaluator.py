"""Example evaluator for a Part 3 agentic research system.

Copy this file to research/evaluator.py and choose an objective for your search.
The simulator supplies the measurements; your evaluator decides how to rank
candidates and what feedback to give the agents. Use split="screen" (one
public_dev trace per family) to discard weak candidates quickly.
"""

from agentkv import score


def evaluate(policy_path, *, split="public_dev", params=None, workers=None):
    """Return the simulator results and a search objective to maximize.

    Keep the measured fields unchanged so campaigns can report both their
    own objective and the assignment's overall and per-category speedups.
    An invalid candidate has no objective and must not be selected.
    """
    result = score(policy_path, split=split, params=params, workers=workers)
    feedback = result.to_json()
    feedback["diagnostics"] = result.summary(
        details=True
    )  # per-trace cache, preemption, and CPU statistics
    feedback["objective"] = None
    if result.valid:
        # Choose the value your system should maximize. This example uses the
        # assignment's overall speedup. To study document QA, use
        # result.by_family["W3"]; to favor the worst category, use
        # min(result.by_family.values()). Explain your choice in the report.
        feedback["objective"] = result.speedup
    return feedback
