import copy
import json

import pytest
from conftest import build_chat_trace

from agentkv.trace import TraceError, from_dict, validate


@pytest.fixture
def d(tmp_path):
    build_chat_trace(tmp_path / "t.json")
    return json.loads((tmp_path / "t.json").read_text())


def broken(d, f):
    d = copy.deepcopy(d)
    f(d)
    return d


def test_valid(d):
    assert validate(d, require_jobs=16) == []
    t = from_dict(d, require_jobs=16)
    assert len(t.jobs) == 16


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(schema="x"), "schema"),
        (lambda d: d["jobs"].pop(), "expected 16 jobs"),
        (lambda d: d["jobs"][0]["nodes"][0].update(deps=["n0"]), "cycle"),
        (lambda d: d["jobs"][0]["nodes"][1].update(deps=["zz"]), "unknown dep"),
        (lambda d: d["jobs"][0]["nodes"][0].update(prompt=["nope"]), "unknown segment"),
        (lambda d: d["jobs"][0]["nodes"][1].update(delay=-1), "delay"),
        (lambda d: d["jobs"][3].update(arrival=-2), "arrival"),
        (lambda d: d["jobs"][3].update(arrival=10**6), "sorted"),
    ],
)
def test_structural_errors(d, mutate, message):
    errs = validate(broken(d, mutate), require_jobs=16)
    assert any(message in e for e in errs), errs


def test_token_count_mismatch_rejected(d):
    sid = next(iter(d["segments"]))
    bad = broken(
        d,
        lambda x: x["segments"][sid].update(
            n_tokens=x["segments"][sid]["n_tokens"] + 1
        ),
    )
    with pytest.raises(TraceError, match="n_tokens"):
        from_dict(bad)


def test_call_must_fit_in_half_of_memory(d):
    bad = broken(d, lambda x: x["memory"].update(num_blocks=10))
    with pytest.raises(TraceError, match="half of memory"):
        from_dict(bad)


def test_shared_segments_share_block_hashes(d):
    t = from_dict(d)
    first = [n.hashes[0] for _, n in t.llm_nodes() if n.hashes]
    assert len(set(first)) == 1  # every call starts with the same system prompt block
