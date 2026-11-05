"""Part 1: the supplied citation checker."""

import copy

import pytest

from research_app.artifacts import Claim, Evidence, Report, locate
from research_app.citations import check_report, is_valid
from research_app.corpus import Corpus


@pytest.fixture(scope="module")
def corpus():
    return Corpus("v2")


def make_report(corpus, doc_id="raft", snapshot="v2"):
    doc = corpus.doc(doc_id, snapshot)
    quote = doc.text.split(". ")[0] + "."
    start, end = locate(quote, doc.text)
    ev = Evidence(
        "E1", doc_id, doc.sha256, start, end, quote, "f1", doc.uri, "R00/f1/extract"
    )
    return Report(
        "R00",
        "q",
        "What is Raft?",
        "Raft",
        [Claim("Raft is a consensus algorithm.", ["E1"])],
        [ev],
    )


def codes(report, corpus, **kw):
    return [e.code for e in check_report(report, corpus, **kw)]


def test_valid_report(corpus):
    r = make_report(corpus)
    assert is_valid(r, corpus)
    assert is_valid(r.to_dict(), corpus)  # dict form too
    assert is_valid(r, corpus, snapshot="v2")


def test_wrong_snapshot(corpus):
    r = make_report(corpus, snapshot="v1")
    assert is_valid(r, corpus)  # a real snapshot of the doc
    assert codes(r, corpus, snapshot="v2") == ["wrong_snapshot"]
    bogus = copy.deepcopy(r)
    bogus.evidence[0].snapshot = "0" * 64
    assert codes(bogus, corpus) == ["wrong_snapshot"]


def test_bad_span(corpus):
    r = make_report(corpus)
    r.evidence[0].end = 10**7
    assert codes(r, corpus) == ["span_out_of_range"]
    r = make_report(corpus)
    r.evidence[0].start += 1
    assert codes(r, corpus) == ["quote_mismatch"]


def test_unknown_evidence_and_doc(corpus):
    r = make_report(corpus)
    r.claims.append(Claim("Unsupported.", ["E9"]))
    assert codes(r, corpus) == ["unknown_evidence"]
    r = make_report(corpus)
    r.evidence[0].doc_id = "no-such-doc"
    assert codes(r, corpus) == ["unknown_doc"]


def test_uncited_and_empty(corpus):
    r = make_report(corpus)
    r.claims.append(Claim("No citation.", []))
    assert codes(r, corpus) == ["uncited_claim"]
    r.claims = []
    assert codes(r, corpus) == ["no_claims"]


def test_locate_is_whitespace_tolerant(corpus):
    text = corpus.doc("paxos").text
    sent = text.split(". ")[1]
    squashed = "  " + sent.replace(" ", "\n ", 1) + " "
    span = locate(squashed, text)
    assert span is not None and text[span[0] : span[1]].split() == sent.split()
    assert locate("this sentence is not in the document at all", text) is None
