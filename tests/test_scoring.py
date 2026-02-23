"""Unit tests for scoring functions in scoring/score.py."""
import pandas as pd
import pytest
from scoring.score import syntactic_score, doc_score, composite_score, security_score


# ── Helpers ────────────────────────────────────────────────────────────────

def make_row(**kwargs) -> pd.Series:
    defaults = {
        "is_parseable": True,
        "num_lines": 20,
        "avg_line_length": 40.0,
        "alphanum_fraction": 0.60,
        "num_functions": 2,
        "num_classes": 0,
        "has_docstring": False,
        "comment_ratio": 0.0,
        "syntactic_score": 0.9,
        "doc_score": 0.5,
        "security_score": 1.0,
        "llm_score": None,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


# ── syntactic_score ────────────────────────────────────────────────────────

def test_syntactic_unparseable_returns_zero():
    row = make_row(is_parseable=False)
    assert syntactic_score(row) == 0.0


def test_syntactic_valid_code_high_score():
    row = make_row(is_parseable=True, num_lines=30, avg_line_length=50, alphanum_fraction=0.65)
    score = syntactic_score(row)
    assert score >= 0.8


def test_syntactic_penalises_short_files():
    long_row = make_row(num_lines=30)
    short_row = make_row(num_lines=5)
    assert syntactic_score(long_row) > syntactic_score(short_row)


def test_syntactic_penalises_long_lines():
    normal = make_row(avg_line_length=60)
    very_long = make_row(avg_line_length=130)
    assert syntactic_score(normal) > syntactic_score(very_long)


def test_syntactic_penalises_low_alphanum():
    # alphanum=0.10 triggers -0.20, but num_functions=2 gives +0.05 bonus → 0.85
    # Confirm it's penalised vs a normal row (0.90 baseline)
    normal = make_row(alphanum_fraction=0.60)
    penalised = make_row(alphanum_fraction=0.10)
    assert syntactic_score(penalised) < syntactic_score(normal)


def test_syntactic_bounds():
    for row in [make_row(), make_row(is_parseable=False), make_row(alphanum_fraction=0.01, num_lines=2)]:
        score = syntactic_score(row)
        assert 0.0 <= score <= 1.0


# ── doc_score ─────────────────────────────────────────────────────────────

def test_doc_no_docs_returns_zero():
    row = make_row(has_docstring=False, comment_ratio=0.0, num_functions=1)
    assert doc_score(row) == 0.0


def test_doc_with_docstring_over_half():
    row = make_row(has_docstring=True, comment_ratio=0.0)
    assert doc_score(row) >= 0.5


def test_doc_high_comment_ratio_bonus():
    low = make_row(has_docstring=False, comment_ratio=0.03)
    high = make_row(has_docstring=False, comment_ratio=0.12)
    assert doc_score(high) > doc_score(low)


def test_doc_penalises_undocumented_large_files():
    small = make_row(has_docstring=False, num_functions=1)
    large_undoc = make_row(has_docstring=False, num_functions=5)
    assert doc_score(small) >= doc_score(large_undoc)


def test_doc_bounds():
    for row in [make_row(), make_row(has_docstring=True, comment_ratio=0.15), make_row(num_functions=10)]:
        score = doc_score(row)
        assert 0.0 <= score <= 1.0


# ── security_score ─────────────────────────────────────────────────────────

def test_security_clean_code_returns_high_score():
    import shutil
    clean_code = "def add(a, b):\n    return a + b\n"
    result = security_score(clean_code)
    if shutil.which("bandit"):
        # bandit on PATH: clean code should have no findings and score >= 0.9
        assert result["security_score"] >= 0.9
        assert result["security_finding_count"] == 0
    else:
        # bandit not on PATH: fallback score of 0.5, no findings recorded
        assert result["security_score"] == 0.5


def test_security_returns_required_keys():
    result = security_score("x = 1")
    assert {"security_score", "security_findings", "security_finding_count", "security_high_count"} <= result.keys()


def test_security_score_bounds():
    result = security_score("import os\nos.system('ls')")
    assert 0.0 <= result["security_score"] <= 1.0


# ── composite_score ────────────────────────────────────────────────────────

def test_composite_without_llm_redistributes_weight():
    row = make_row(syntactic_score=1.0, doc_score=1.0, security_score=1.0, llm_score=None)
    score = composite_score(row)
    assert score == pytest.approx(1.0, abs=0.01)


def test_composite_with_llm_included():
    row = make_row(syntactic_score=1.0, doc_score=1.0, security_score=1.0, llm_score=5.0)
    score = composite_score(row)
    assert score == pytest.approx(1.0, abs=0.01)


def test_composite_low_scores_below_threshold():
    row = make_row(syntactic_score=0.0, doc_score=0.0, security_score=0.0, llm_score=1.0)
    score = composite_score(row)
    assert score < 0.10


def test_composite_bounds():
    for llm in [None, 1.0, 3.0, 5.0]:
        row = make_row(syntactic_score=0.8, doc_score=0.6, security_score=0.9, llm_score=llm)
        score = composite_score(row)
        assert 0.0 <= score <= 1.0
