"""Small helpers the notebooks share.

from agentkv.notebook import check, repo_root
check("tests/test_part2_lru.py")     # run tests; print PASS or FAIL, then any failing tests
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    """Return the repository directory containing the agentkv package."""
    return Path(__file__).resolve().parents[1]


def check(*tests: str, verbose: bool = False) -> None:
    """Run pytest on `tests` in a fresh process and print PASS or FAIL.

    A fresh process means your latest edits are always what gets tested. With
    verbose=False, only failing tests are shown, with their last error line."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--no-header",
        "--color=no",
        "-p",
        "no:cacheprovider",
        *tests,
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root(),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "COLUMNS": "200"},
    )
    out = proc.stdout + proc.stderr
    summary = next(
        (
            line
            for line in reversed(out.splitlines())
            if re.search(r"passed|failed|error", line)
        ),
        out,
    )
    ok = proc.returncode == 0
    print(("PASS  " if ok else "FAIL  ") + summary.strip("= "))
    if verbose or not ok:
        failures = re.findall(r"^(?:FAILED|ERROR) (\S+)(?: - (.*))?$", out, re.M)
        for name, why in failures[:15]:
            print(f"  {name}" + (f"\n      {why[:160]}" if why else ""))
        if len(failures) > 15:
            print(f"  ... and {len(failures) - 15} more")
        if verbose:
            print(out)
