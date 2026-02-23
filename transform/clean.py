"""
Stage 3: Transform & Clean
Deduplication, syntax validation, schema normalisation.
"""

import ast
import hashlib
import logging
from pathlib import Path

import polars as pl
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DIR,
    MIN_LINE_COUNT, MAX_LINE_COUNT,
    MIN_ALPHANUM_FRACTION, MAX_AVG_LINE_LENGTH,
    DEDUP_NEAR, MINHASH_THRESHOLD, MINHASH_NUM_PERM,
)

logger = logging.getLogger(__name__)


def deduplicate(df: pl.DataFrame, content_col: str = "content") -> pl.DataFrame:
    """Exact deduplication via SHA256 hash of stripped content."""
    original_len = len(df)

    df = df.with_columns(
        pl.col(content_col)
        .map_elements(lambda x: hashlib.sha256(x.strip().encode()).hexdigest(), return_dtype=pl.Utf8)
        .alias("content_hash")
    )
    df = df.unique(subset=["content_hash"])

    removed = original_len - len(df)
    logger.info(f"Dedup: removed {removed} duplicates ({removed/original_len:.1%} of dataset)")
    return df


def validate_python_syntax(code: str) -> dict:
    """
    Attempts to parse Python code via AST.
    Returns validity flag and basic structural metrics.
    """
    try:
        tree = ast.parse(code)
        num_functions = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        num_classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
        has_docstring = ast.get_docstring(tree) is not None
        return {
            "is_valid": True,
            "num_functions": num_functions,
            "num_classes": num_classes,
            "has_module_docstring": has_docstring,
            "parse_error": None,
        }
    except SyntaxError as e:
        return {
            "is_valid": False,
            "num_functions": 0,
            "num_classes": 0,
            "has_module_docstring": False,
            "parse_error": str(e),
        }


def normalise_schema(df: pl.DataFrame, source: str) -> pl.DataFrame:
    """
    Flattens dataset-specific fields into a common schema.
    Add source-specific mappings here as new datasets are added.
    """
    # TODO: map source-specific column names to common schema
    # Common target schema:
    # id, source, language, content, num_lines, has_docstring,
    # is_syntactically_valid, content_hash
    df = df.with_columns(pl.lit(source).alias("source"))
    logger.info(f"Normalised schema for source={source} | rows={len(df)}")
    return df


def add_text_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute text-level metrics used by the scoring stage."""
    return df.with_columns([
        pl.col("content")
            .map_elements(lambda x: len(x.splitlines()), return_dtype=pl.Int32)
            .alias("num_lines"),
        pl.col("content")
            .map_elements(
                lambda x: sum(len(l) for l in x.splitlines()) / max(len(x.splitlines()), 1),
                return_dtype=pl.Float64,
            )
            .alias("avg_line_length"),
        pl.col("content")
            .map_elements(
                lambda x: sum(c.isalnum() for c in x) / max(len(x), 1),
                return_dtype=pl.Float64,
            )
            .alias("alphanum_fraction"),
        pl.col("content")
            .map_elements(
                lambda x: sum(1 for l in x.splitlines() if l.strip().startswith("#")) / max(len(x.splitlines()), 1),
                return_dtype=pl.Float64,
            )
            .alias("comment_ratio"),
    ])


def add_ast_features(df: pl.DataFrame, language: str) -> pl.DataFrame:
    """Run AST parsing for Python files and add structural metrics as columns."""
    empty_cols = [
        pl.lit(None).cast(pl.Boolean).alias("is_parseable"),
        pl.lit(0).alias("num_functions"),
        pl.lit(0).alias("num_classes"),
        pl.lit(False).alias("has_docstring"),
        pl.lit(None).cast(pl.Utf8).alias("parse_error"),
    ]

    if language != "python" or len(df) == 0:
        return df.with_columns(empty_cols)

    ast_records = [validate_python_syntax(code) for code in df["content"].to_list()]
    ast_series = pl.DataFrame(ast_records)

    return df.with_columns([
        ast_series["is_valid"].alias("is_parseable"),
        ast_series["num_functions"],
        ast_series["num_classes"],
        ast_series["has_module_docstring"].alias("has_docstring"),
        ast_series["parse_error"],
    ])


def near_deduplicate(df: pl.DataFrame, content_col: str = "content") -> pl.DataFrame:
    """
    Near-deduplication using MinHash LSH (Jaccard similarity over 5-char shingles).
    Streams through the dataset; keeps the first of any near-duplicate cluster.
    Controlled by DEDUP_NEAR and MINHASH_THRESHOLD in config.
    """
    from datasketch import MinHash, MinHashLSH

    before = len(df)
    contents = df[content_col].to_list()

    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_NUM_PERM)
    keep_indices = []

    for i, text in enumerate(contents):
        m = MinHash(num_perm=MINHASH_NUM_PERM)
        shingles = {text[j:j + 5].encode("utf-8") for j in range(max(1, len(text) - 4))}
        for shingle in shingles:
            m.update(shingle)

        # If no near-duplicate already in the index, keep this doc
        if not lsh.query(m):
            lsh.insert(str(i), m)
            keep_indices.append(i)

    df = df[keep_indices]
    removed = before - len(df)
    logger.info(
        f"Near-dedup (threshold={MINHASH_THRESHOLD}): removed {removed} rows "
        f"({removed / max(before, 1):.1%}), {len(df)} remaining"
    )
    return df


def apply_hard_filters(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows that fail hard quality cutoffs defined in config."""
    before = len(df)
    df = df.filter(
        (pl.col("num_lines") >= MIN_LINE_COUNT) &
        (pl.col("num_lines") <= MAX_LINE_COUNT) &
        (pl.col("alphanum_fraction") >= MIN_ALPHANUM_FRACTION) &
        (pl.col("avg_line_length") <= MAX_AVG_LINE_LENGTH)
    )
    removed = before - len(df)
    logger.info(f"Hard filters: removed {removed} rows ({removed/max(before,1):.1%}), {len(df)} remaining")
    return df


def run_transforms(input_path: Path, source: str, language: str = "python") -> Path:
    """Full transform stage: load → deduplicate → validate → normalise → filter → save."""
    logger.info(f"Running transforms on {input_path}")

    df = pl.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} rows")

    df = deduplicate(df)
    if DEDUP_NEAR:
        df = near_deduplicate(df)
    df = normalise_schema(df, source=source)
    df = add_text_features(df)
    df = apply_hard_filters(df)
    df = add_ast_features(df, language=language)

    n_valid = df["is_parseable"].sum() if "is_parseable" in df.columns else "n/a"
    logger.info(f"AST valid: {n_valid}/{len(df)}")

    output_path = PROCESSED_DIR / f"{source}_{language}_processed.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    logger.info(f"Saved processed data to {output_path}")
    return output_path
