"""Policy sandbox: bad return values, state overflow, runaway CPU, exceptions,
and attempts at network, file, subprocess, thread or wall-clock access. No
silent repairs: every one of these fails the evaluation."""

import textwrap

import pytest
from conftest import FIFO, write_policy

from agentkv.replay import Replay, run_policy
from agentkv.sandbox.client import WorkerClient

BASE = """
from agentkv.policy import Policy

class P(Policy):
    def on_event(self, event, state):
        pass
    def target_concurrency(self, view, state):
        return 4
    def choose_victim(self, view, candidates, state):
        return candidates[0]
"""


def run(trace, tmp_path, source):
    path = write_policy(tmp_path, textwrap.dedent(source))
    return run_policy(trace, path)


def variant(**overrides):
    """BASE with some methods replaced, e.g. variant(target_concurrency="return 9")."""
    src = BASE
    for hook, body in overrides.items():
        marker = f"    def {hook}(self"
        start = src.index(marker)
        nxt = src.find("\n    def ", start + 1)
        sig = src[start : src.index("\n", start)]
        body = textwrap.indent(textwrap.dedent(body).strip(), " " * 8)
        src = src[:start] + sig + "\n" + body + "\n" + (src[nxt:] if nxt != -1 else "")
    return src


def test_base_policy_is_valid(pressure_trace, tmp_path):
    r = run(pressure_trace, tmp_path, BASE)
    assert r.valid, r.violation


@pytest.mark.parametrize("ret", ["0", "17", "'4'", "True", "4.0", "None"])
def test_bad_cap(chat_trace, tmp_path, ret):
    r = run(chat_trace, tmp_path, variant(target_concurrency=f"return {ret}"))
    assert not r.valid
    assert r.violation.startswith("invalid")


@pytest.mark.parametrize(
    "body",
    [
        "return 'not-a-block'",
        "return next(iter(view.blocks.values() if False else [b for b in view.blocks if view.blocks[b].ref_cnt > 0] or ['x']))",
        "return 7",
    ],
)
def test_bad_victim(pressure_trace, tmp_path, body):
    r = run(pressure_trace, tmp_path, variant(choose_victim=body))
    assert not r.valid
    assert r.violation.startswith("invalid")


def test_exception(chat_trace, tmp_path):
    r = run(
        chat_trace, tmp_path, variant(target_concurrency="raise ValueError('boom')")
    )
    assert r.violation.startswith("exception") and "boom" in r.violation


def test_exception_in_on_event(chat_trace, tmp_path):
    r = run(
        chat_trace, tmp_path, variant(on_event="if event.kind == 'complete': 1 / 0")
    )
    assert r.violation.startswith("exception")


def test_state_overflow(chat_trace, tmp_path):
    r = run(
        chat_trace,
        tmp_path,
        variant(on_event="state.setdefault('junk', []).append(repr(event) * 50)"),
    )
    assert r.violation.startswith("state")


def test_unserializable_state(chat_trace, tmp_path):
    r = run(chat_trace, tmp_path, variant(on_event="state['f'] = (lambda: 0)"))
    assert r.violation.startswith("state")


def test_runaway_cpu(chat_trace, tmp_path):
    body = """
    x = 0
    for i in range(10 ** 8):
        x += i
        if i % 100000 == 0 and view.rng.random() < 0:
            break
    return 4
    """
    r = run(chat_trace, tmp_path, variant(target_concurrency=body))
    assert r.violation.startswith("timeout")


@pytest.mark.slow
def test_infinite_loop_is_killed(chat_trace, tmp_path):
    r = run(chat_trace, tmp_path, variant(target_concurrency="while True: pass"))
    assert r.violation.startswith("timeout")


@pytest.mark.parametrize(
    "body",
    [
        "open('/etc/hosts').read()",
        "import socket",
        "import subprocess",
        "import os; os.system('true')",
        "import os; os.listdir('/')",
        "import threading; threading.Thread(target=print).start()",
        "import time; time.time()",
        "import ctypes",
    ],
)
def test_forbidden_operations(chat_trace, tmp_path, body):
    r = run(chat_trace, tmp_path, variant(target_concurrency=f"{body}\nreturn 4"))
    assert r.violation is not None and r.violation.startswith("sandbox"), r.violation


def test_caught_violation_still_fails(chat_trace, tmp_path):
    body = """
    try:
        open('/etc/hosts')
    except Exception:
        pass
    return 4
    """
    r = run(chat_trace, tmp_path, variant(target_concurrency=body))
    assert r.violation.startswith("sandbox")


def test_module_level_code_is_sandboxed(chat_trace, tmp_path):
    src = (
        "import os\nSECRET = open(os.path.expanduser('~/.ssh/id_rsa')).read()\n" + BASE
    )
    r = run(chat_trace, tmp_path, src)
    assert r.violation.startswith("sandbox")


def test_malicious_policy_cannot_read_trace_files(chat_trace, tmp_path):
    """A policy that tries to read the trace it is being scored on."""
    body = f"""
    try:
        state['leak'] = open({chat_trace.path!r}).read()[:100]
    except Exception:
        pass
    return 4
    """
    r = run(chat_trace, tmp_path, variant(target_concurrency=body))
    assert r.violation.startswith("sandbox")


def test_print_does_not_break_the_protocol(chat_trace, tmp_path):
    r = run(
        chat_trace, tmp_path, variant(target_concurrency="print('hello'); return 4")
    )
    assert r.valid, r.violation


def test_load_errors(chat_trace, tmp_path):
    assert run(chat_trace, tmp_path, "this is not python").violation.startswith("load")
    assert run(chat_trace, tmp_path, "x = 1").violation.startswith("load")


def test_file_policy_runs_in_sandbox(chat_trace):
    m = chat_trace.memory
    r = Replay(
        chat_trace,
        WorkerClient(FIFO, block_size=m.block_size, total_blocks=m.num_blocks),
    ).run()
    assert r.valid


def test_preloaded_numpy_does_not_open_ctypes(chat_trace, tmp_path):
    """numpy is preloaded for policies and itself imports ctypes; a policy's own
    `import ctypes` must still be refused."""
    pytest.importorskip("numpy")
    r = run(
        chat_trace,
        tmp_path,
        variant(target_concurrency="import numpy\nimport ctypes\nreturn 4"),
    )
    assert r.violation is not None and r.violation.startswith("sandbox"), r.violation
