"""Tests for the pure functions. These are what let you refactor retrieval
without re-running the whole app to find out you broke it."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from index import extract_references, rrf, tokenize


def test_tokenize_keeps_hyphenated_sections():
    assert "489-f" in tokenize("Section 489-F PPC")
    assert "10a" in tokenize("Article 10A")


def test_reference_article():
    assert extract_references("Article 199") == [("199", "Constitution")]


def test_reference_bare_act_suffix():
    assert ("302", "PPC") in extract_references("302 PPC")


def test_reference_abbreviated_section():
    assert ("497", "CrPC") in extract_references("u/s 497 CrPC")


def test_reference_hyphenated():
    assert ("489-F", "PPC") in extract_references("Section 489-F PPC")


def test_no_reference_in_plain_question():
    assert extract_references("can I get bail") == []


def test_rrf_rewards_agreement():
    """A doc ranked mid by both channels should beat one ranked top by only one."""
    a, b = ["x", "y"], ["z", "y"]
    scores = dict(rrf([a, b], k=1))
    assert scores["y"] > scores["x"]


def test_rrf_weights_applied():
    s = dict(rrf([["a"], ["b"]], k=60, weights=[2.0, 1.0]))
    assert s["a"] == 2 * s["b"]


def test_rrf_empty():
    assert rrf([]) == []
