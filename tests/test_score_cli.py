"""The scorer's command line: --params takes inline JSON or a JSON file."""

import json

from agentkv.score import parse_params


def test_inline_params_of_any_length():
    big = {"knob_" + str(i): i for i in range(60)}  # well over 255 characters
    assert parse_params(json.dumps(big)) == big


def test_params_from_a_file(tmp_path):
    f = tmp_path / "p.json"
    f.write_text('{"cap": 4}')
    assert parse_params(str(f)) == {"cap": 4}
    assert parse_params(None) is None
