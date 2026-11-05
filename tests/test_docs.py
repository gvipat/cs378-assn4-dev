"""The README, trace format, and notebooks only point at things that exist, and are
formatted consistently: every referenced file and dotted name resolves, every
code fence names its language, and notebooks ship valid and without outputs."""

import importlib
import json
import re

import pytest
from conftest import REPO

DOCS = [REPO / "README.md", REPO / "docs" / "trace_format.md"]
NOTEBOOKS = sorted((REPO / "notebooks").glob("*.ipynb"))
# Paths students create themselves.
CREATED = ("runs/", "candidates/", "research/", "out.json")
TOP = "agentkv|research_app|docs|notebooks|policies|examples|tests|data|tools|runs|candidates|research"
PATH = re.compile(rf"`((?:{TOP})/[\w./*-]*)`")
DOTTED = re.compile(r"`((?:agentkv|research_app)(?:\.\w+)+)")


def texts():
    for doc in DOCS:
        yield doc.name, doc.read_text()
    for nb in NOTEBOOKS:
        cells = json.loads(nb.read_text())["cells"]
        yield nb.name, "\n".join(
            "".join(c["source"]) for c in cells if c["cell_type"] == "markdown"
        )


@pytest.mark.parametrize(
    "name,text",
    list(texts()),
    ids=lambda v: v if isinstance(v, str) and len(v) < 60 else "",
)
def test_referenced_paths_exist(name, text):
    missing = []
    for path in PATH.findall(text):
        if path.startswith(CREATED) or "*" in path or "<" in path:
            continue
        if not (REPO / path).exists():
            missing.append(path)
    assert not missing, f"{name} mentions paths that do not exist: {missing}"


@pytest.mark.parametrize(
    "name,text",
    list(texts()),
    ids=lambda v: v if isinstance(v, str) and len(v) < 60 else "",
)
def test_referenced_names_resolve(name, text):
    unresolved = []
    for dotted in set(DOTTED.findall(text)):
        parts = dotted.split(".")
        for i in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:i]))
            except ImportError:
                continue
            try:
                for attr in parts[i:]:
                    obj = getattr(obj, attr)
            except AttributeError:
                unresolved.append(dotted)
            break
        else:
            unresolved.append(dotted)
    assert not unresolved, f"{name} mentions names that do not resolve: {unresolved}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_code_fences_name_a_language(doc):
    inside, bare = False, []
    for n, line in enumerate(doc.read_text().splitlines(), 1):
        if line.strip().startswith("```"):
            if not inside and line.strip() == "```":
                bare.append(n)
            inside = not inside
    assert not bare, f"{doc.name}: code fences without a language on lines {bare}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebooks_are_valid_clean_and_readable(path):
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert (
                not cell.outputs and cell.execution_count is None
            ), "notebooks ship without outputs"
            long = [line for line in cell.source.splitlines() if len(line) > 110]
            assert not long, f"code lines over 110 characters: {long}"


@pytest.mark.parametrize(
    "module",
    [
        "agentkv.viz",
        "agentkv.notebook",
        "agentkv.harness",
        "research_app.agent",
        "research_app.concurrency",
    ],
)
def test_public_functions_have_docstrings(module):
    """Check that student-facing functions have docstrings."""
    import inspect

    mod = importlib.import_module(module)
    missing = [
        name
        for name, obj in vars(mod).items()
        if inspect.isfunction(obj)
        and obj.__module__ == module
        and not name.startswith("_")
        and not inspect.getdoc(obj)
    ]
    assert not missing, f"{module}: no docstring on {missing}"
