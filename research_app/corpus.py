"""Search the saved Wikipedia corpus and retrieve article revisions.

The corpus contains 48 articles on six systems topics, with two revisions of
each article in data/corpus/docs.json. The tool wrapper assigns reproducible
delays and failures. Timed runs wait for the delays; --record logs them
without waiting. Exporting replay.json with --replay does not change timing.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable

CORPUS_PATH = Path(
    os.environ.get(
        "AGENTKV_CORPUS",
        Path(__file__).resolve().parents[1] / "data" / "corpus" / "docs.json",
    )
)

STOPWORDS = frozenset(
    """
a an and are as at be been but by can do does for from has have how in into is it its of on or that the
their them then there these this those to was were what when where which while who why will with would
than also may such more most other some each between both use used using via about over under not no
""".split()
)


def terms(text: str) -> list[str]:
    return [
        w
        for w in re.findall(r"[a-z0-9][a-z0-9+\-]*", text.lower())
        if w not in STOPWORDS and len(w) > 1
    ]


@dataclass(frozen=True)
class Doc:
    """The exact source text and metadata passed to the research agent."""

    doc_id: str
    topic: str
    title: str
    uri: str
    snapshot: str  # local revision name, or retrieval time for a web source
    sha256: str  # snapshot hash; evidence cites this
    rev_id: int
    permalink: str
    text: str


@dataclass(frozen=True)
class SearchHit:
    """A search result, including its document id, relevance score, and excerpt."""

    doc_id: str
    title: str
    uri: str
    snapshot: str  # sha256 of the snapshot text (Doc.sha256), not "v1"/"v2"
    score: float
    snippet: str


class Corpus:
    """Search and fetch a selected version of the assignment's documents.

    Each article has two saved revisions. Corpus("v1") uses the earlier
    revision of each article; Corpus("v2") uses the later revision. Document
    ids stay the same across versions, but the article text and its hash may
    differ. Search results and fetched documents use the selected version.
    """

    _cache: dict[str, dict] = {}

    def __init__(self, snapshot: str = "v2", path: str | Path | None = None):
        path = str(path or CORPUS_PATH)
        if path not in Corpus._cache:
            Corpus._cache[path] = json.loads(Path(path).read_text())
        self.raw = Corpus._cache[path]
        self.path = path
        self.snapshot = snapshot
        self._docs = {d["doc_id"]: d for d in self.raw["docs"]}
        if snapshot not in self.raw["docs"][0]["snapshots"]:
            raise ValueError(f"unknown snapshot {snapshot!r}")

    def with_snapshot(self, snapshot: str) -> "Corpus":
        """Return a new Corpus for the requested version, using the same source file."""
        return Corpus(snapshot, self.path)

    @property
    def doc_ids(self) -> list[str]:
        """Return the document ids in sorted order."""
        return sorted(self._docs)

    def topics(self) -> dict[str, list[str]]:
        """Group document ids by their systems topic."""
        out: dict[str, list[str]] = {}
        for d in self.raw["docs"]:
            out.setdefault(d["topic"], []).append(d["doc_id"])
        return out

    def doc(self, doc_id: str, snapshot: str | None = None) -> Doc:
        """Return an article's selected revision, or the named revision if provided."""
        d = self._docs[doc_id]
        name = snapshot or self.snapshot
        s = d["snapshots"][name]
        return Doc(
            doc_id,
            d["topic"],
            d["title"],
            d["uri"],
            name,
            s["sha256"],
            s["rev_id"],
            s["permalink"],
            s["text"],
        )

    def has(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def snapshot_of(self, doc_id: str, sha256: str) -> str | None:
        """Find which saved revision of this article matches the given text hash.

        Return its version name ("v1" or "v2"), or None if neither matches.
        """
        for name, s in self._docs[doc_id]["snapshots"].items():
            if s["sha256"] == sha256:
                return name
        return None

    # ------------------------------------------------------------ BM25

    @cached_property
    def _index(self) -> tuple[dict[str, Counter], dict[str, int], float]:
        tfs: dict[str, Counter] = {}
        for doc_id in self.doc_ids:
            doc = self.doc(doc_id)
            tfs[doc_id] = Counter(terms(doc.title) * 3 + terms(doc.text))
        df: Counter = Counter()
        for tf in tfs.values():
            df.update(tf.keys())
        avg = sum(sum(tf.values()) for tf in tfs.values()) / len(tfs)
        return tfs, dict(df), avg

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        """Return up to k matching articles, ranked by their BM25 relevance score."""
        tfs, df, avg = self._index
        n = len(tfs)
        q = terms(query)
        scored = []
        for doc_id, tf in tfs.items():
            length = sum(tf.values())
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * length / avg))
            if s > 0:
                scored.append((-s, doc_id))
        scored.sort()
        hits = []
        for neg, doc_id in scored[:k]:
            doc = self.doc(doc_id)
            hits.append(
                SearchHit(
                    doc_id,
                    doc.title,
                    doc.uri,
                    doc.sha256,
                    round(-neg, 4),
                    snippet(doc.text, q),
                )
            )
        return hits

    def fetch(self, doc_id: str) -> Doc:
        """Return the full article text and metadata; raise KeyError for an unknown id."""
        if doc_id not in self._docs:
            raise KeyError(doc_id)
        return self.doc(doc_id)


def snippet(text: str, query_terms: list[str], width: int = 200) -> str:
    best, best_score = "", -1
    qs = set(query_terms)
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        score = len(qs & set(terms(sent)))
        if score > best_score:
            best, best_score = sent, score
    return best[:width]


# ---------------------------------------------------------------- tools


class TransientToolError(Exception):
    pass


class PermanentToolError(Exception):
    pass


# fault(kind, op_id, attempt) -> None | "transient" | "permanent"
FaultFn = Callable[[str, str, int], str | None]


@dataclass
class ToolLatency:
    """Assign reproducible delays and transient failures to tool operations.

    The seed and operation key determine each outcome. Changing the order
    in which calls run therefore does not change their assigned outcomes.
    """

    seed: int = 0
    search_range: tuple[float, float] = (0.3, 1.2)
    fetch_median: float = 0.6
    fetch_sigma: float = 0.6
    fetch_range: tuple[float, float] = (0.2, 2.5)
    straggler_rate: float = 0.03  # heavy tail: occasional 2-4x slow fetches
    failure_rate: float = 0.0  # transient failure probability per attempt
    scale: float = 1.0

    def _rng(self, kind: str, key: str) -> random.Random:
        return random.Random(f"{self.seed}|{kind}|{key}")

    def duration(self, kind: str, key: str) -> float:
        rng = self._rng(kind, key)
        if kind == "search":
            d = rng.uniform(*self.search_range)
        else:
            d = min(
                max(
                    rng.lognormvariate(math.log(self.fetch_median), self.fetch_sigma),
                    self.fetch_range[0],
                ),
                self.fetch_range[1],
            )
            if rng.random() < self.straggler_rate:
                d *= rng.uniform(2.0, 4.0)
        return round(d * self.scale, 4)

    def fails(self, kind: str, key: str) -> bool:
        return (
            self.failure_rate > 0
            and self._rng(kind + "/fail", key).random() < self.failure_rate
        )


class Tools:
    """Wrap corpus search and fetch with failure checks and delay estimates.

    The tracing client calls duration() and handles the wait. The search()
    and fetch() methods return local results or raise a tool error.
    """

    def __init__(
        self,
        corpus: Corpus,
        latency: ToolLatency | None = None,
        faults: FaultFn | None = None,
    ):
        self.corpus = corpus
        self.latency = latency or ToolLatency()
        self.faults = faults

    def _check(self, kind: str, op_id: str, attempt: int) -> None:
        fault = self.faults(kind, op_id, attempt) if self.faults else None
        if fault == "permanent":
            raise PermanentToolError(f"{kind} {op_id}: permanent failure")
        if fault == "transient" or self.latency.fails(kind, f"{op_id}#{attempt}"):
            raise TransientToolError(f"{kind} {op_id}: transient failure")

    def duration(self, kind: str, op_id: str, attempt: int) -> float:
        return self.latency.duration(kind, f"{op_id}#{attempt}")

    def search(
        self, query: str, *, op_id: str, attempt: int = 1, k: int = 5
    ) -> list[SearchHit]:
        self._check("search", op_id, attempt)
        return self.corpus.search(query, k)

    def fetch(self, doc_id: str, *, op_id: str, attempt: int = 1) -> Doc:
        self._check("fetch", op_id, attempt)
        if not self.corpus.has(doc_id):
            raise PermanentToolError(f"fetch {op_id}: unknown document {doc_id}")
        return self.corpus.fetch(doc_id)
