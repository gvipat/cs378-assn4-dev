"""The optional scheduling hooks: dispatch_order, admission_order, and
choose_preempted.

A policy that leaves them out gets nano-vLLM's rules without being called; one
that spells those rules out gets exactly the same run. A policy that uses them
changes what the engine does, and a bad answer fails the run like any other
rule violation."""

import textwrap

import pytest
from conftest import write_policy

from agentkv.policy import Policy, overridden_hooks
from agentkv.replay import Replay, run_policy
from agentkv.sandbox.client import InProcessClient

BASE = """
from agentkv.policy import Policy

class P(Policy):
    def target_concurrency(self, view, state):
        return 6
    def choose_victim(self, view, candidates, state):
        return candidates[0]
"""

SPELLED_OUT_DEFAULTS = (
    BASE
    + """
    def dispatch_order(self, view, ready, state):
        return list(ready)
    def admission_order(self, view, waiting, state):
        return list(waiting)
    def choose_preempted(self, view, running, state):
        return running[-1]
"""
)


def run(trace, tmp_path, source, isolate=True):
    return run_policy(
        trace, write_policy(tmp_path, textwrap.dedent(source)), isolate=isolate
    )


def test_only_defined_hooks_are_reported():
    class Plain(Policy):
        pass

    class Picky(Policy):
        def choose_preempted(self, view, running, state):
            return running[0]

    assert overridden_hooks(Plain()) == ()
    assert overridden_hooks(Picky()) == ("choose_preempted",)


@pytest.mark.parametrize("isolate", [True, False], ids=["sandbox", "in-process"])
def test_spelled_out_defaults_change_nothing(pressure_trace, tmp_path, isolate):
    plain = run(pressure_trace, tmp_path, BASE, isolate)
    spelled = run(pressure_trace, tmp_path, SPELLED_OUT_DEFAULTS, isolate)
    assert plain.valid and spelled.valid, (plain.violation, spelled.violation)
    assert plain.diag["preemptions"] > 0, "the trace should exercise preemption"
    assert spelled.flows == plain.flows
    assert spelled.diag["preemptions"] == plain.diag["preemptions"]


def test_dispatch_follows_the_policy_order(pressure_trace):
    class NewestFirst(Policy):
        def __init__(self, params=None):
            super().__init__(params)
            self.order, self.mismatches, self.dispatches = [], 0, 0

        def on_event(self, event, state):
            if event.kind == "dispatch":
                self.dispatches += 1
                self.mismatches += event.call != self.order.pop(0)

        def target_concurrency(self, view, state):
            return 2

        def choose_victim(self, view, candidates, state):
            return candidates[0]

        def dispatch_order(self, view, ready, state):
            self.order = list(reversed(ready))
            return self.order

    policy = NewestFirst()
    client = InProcessClient(
        policy,
        block_size=pressure_trace.memory.block_size,
        total_blocks=pressure_trace.memory.num_blocks,
    )
    result = Replay(pressure_trace, client).run()
    assert result.valid, result.violation
    assert policy.dispatches == sum(1 for _ in pressure_trace.llm_nodes())
    assert policy.mismatches == 0


def test_hooks_change_the_run(pressure_trace, tmp_path):
    plain = run(pressure_trace, tmp_path, BASE)
    for body in [
        "def dispatch_order(self, view, ready, state):\n    return list(reversed(ready))",
        "def admission_order(self, view, waiting, state):\n    return list(reversed(waiting))",
        "def choose_preempted(self, view, running, state):\n    return running[0]",
    ]:
        r = run(pressure_trace, tmp_path, BASE + textwrap.indent(body, "    "))
        assert r.valid, (body, r.violation)
        assert r.flows != plain.flows, body


@pytest.mark.parametrize(
    "body",
    [
        "def dispatch_order(self, view, ready, state):\n    return ['not-a-call']",
        "def dispatch_order(self, view, ready, state):\n    return list(ready) + list(ready)",
        "def dispatch_order(self, view, ready, state):\n    return ready[0] if ready else []",
        "def dispatch_order(self, view, ready, state):\n    return []",
        "def admission_order(self, view, waiting, state):\n    return list(waiting)[1:]",
        "def admission_order(self, view, waiting, state):\n    return list(waiting) + ['x']",
        "def choose_preempted(self, view, running, state):\n    return 'not-running'",
        "def choose_preempted(self, view, running, state):\n    return None",
    ],
    ids=[
        "dispatch-unknown",
        "dispatch-duplicate",
        "dispatch-not-a-list",
        "dispatch-stalls",
        "admission-drops",
        "admission-adds",
        "preempt-unknown",
        "preempt-none",
    ],
)
def test_bad_answers_fail_the_run(pressure_trace, tmp_path, body):
    r = run(pressure_trace, tmp_path, BASE + textwrap.indent(body, "    "))
    assert not r.valid
    assert r.violation.startswith("invalid"), r.violation
