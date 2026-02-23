"""Unit tests for transform/clean.py."""
import polars as pl
from transform.clean import deduplicate, near_deduplicate, apply_hard_filters, add_text_features


def make_df(contents: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"content": contents})


# ── deduplicate (exact) ────────────────────────────────────────────────────

def test_exact_dedup_removes_duplicates():
    df = make_df(["def foo(): pass", "def foo(): pass", "def bar(): pass"])
    result = deduplicate(df)
    assert len(result) == 2


def test_exact_dedup_keeps_all_unique():
    df = make_df(["abc", "def", "ghi"])
    result = deduplicate(df)
    assert len(result) == 3


def test_exact_dedup_whitespace_normalised():
    # Leading/trailing whitespace should be stripped before hashing
    df = make_df(["def foo(): pass", "  def foo(): pass  "])
    result = deduplicate(df)
    assert len(result) == 1


# ── near_deduplicate ───────────────────────────────────────────────────────

def test_near_dedup_removes_near_duplicates():
    # Build a long document so a tiny change gives Jaccard well above 0.8
    base = "\n".join(f"result_{i} = compute(x={i}, y={i*2})  # step {i}" for i in range(200))
    # Change only one token — Jaccard ≈ 0.997, clearly above the 0.8 threshold
    similar = base.replace("result_0", "output_0", 1)
    distinct = "\n".join(f"class Model_{i}:\n    def fit(self): pass" for i in range(200))

    df = make_df([base, similar, distinct])
    result = near_deduplicate(df)
    assert len(result) == 2


def test_near_dedup_keeps_distinct_docs():
    docs = [
        "def add(a, b):\n    return a + b\n" * 8,
        "class Dog:\n    def bark(self):\n        print('woof')\n" * 8,
        "import json\ndef load(path):\n    with open(path) as f:\n        return json.load(f)\n" * 8,
    ]
    df = make_df(docs)
    result = near_deduplicate(df)
    assert len(result) == 3


def test_near_dedup_single_row():
    df = make_df(["only one document here " * 20])
    result = near_deduplicate(df)
    assert len(result) == 1


# ── apply_hard_filters ─────────────────────────────────────────────────────

def test_hard_filters_remove_short_files():
    df = add_text_features(make_df(["x = 1\n" * 3, "x = 1\n" * 20]))
    result = apply_hard_filters(df)
    assert len(result) == 1


def test_hard_filters_keep_valid_files():
    df = add_text_features(make_df(["def foo():\n    pass\n" * 10]))
    result = apply_hard_filters(df)
    assert len(result) == 1


def test_hard_filters_remove_long_lines():
    long_line = "x" * 250 + "\n"
    df = add_text_features(make_df([long_line * 10]))
    result = apply_hard_filters(df)
    assert len(result) == 0
