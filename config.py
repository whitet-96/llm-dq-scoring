"""
Central configuration for the LLM Data Quality Scoring Pipeline.
Adjust thresholds, weights, and settings here.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SCORED_DIR = DATA_DIR / "scored"
DOCS_DIR = BASE_DIR / "docs"

# ── Ingestion ──────────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = ["python", "javascript", "go", "java", "ruby"]
DEFAULT_SAMPLE_SIZE = 1000          # Start small; scale up when confident
DEFAULT_LANGUAGE = "python"

# ── Deduplication ──────────────────────────────────────────────────────────
DEDUP_EXACT = True                  # SHA256 hash dedup (always run)
DEDUP_NEAR = False                  # MinHash LSH (slower, run on larger datasets)
MINHASH_THRESHOLD = 0.8             # Jaccard similarity threshold for near-dedup
MINHASH_NUM_PERM = 128              # Number of permutations for MinHash

# ── Quality Filters (hard cutoffs before scoring) ─────────────────────────
MIN_LINE_COUNT = 5                  # Discard files shorter than this
MAX_LINE_COUNT = 1000               # Discard files longer than this
MIN_ALPHANUM_FRACTION = 0.25        # Discard if less than 25% alphanumeric chars
MAX_AVG_LINE_LENGTH = 200           # Discard if avg line length exceeds this

# ── Scoring Weights (must sum to 1.0) ─────────────────────────────────────
SCORE_WEIGHTS = {
    "syntactic": 0.35,
    "documentation": 0.25,
    "security": 0.20,
    "llm": 0.20,
}

# ── LLM Scoring ────────────────────────────────────────────────────────────
LLM_SCORE_SAMPLE_SIZE = 500         # Max samples to send to Claude API
LLM_MODEL = "claude-haiku-4-5-20251001"  # Use Haiku for cost efficiency
LLM_MAX_TOKENS = 256

# ── Composite Score Threshold ──────────────────────────────────────────────
QUALITY_THRESHOLD = 0.70            # Samples below this are flagged as low quality

# ── Security ───────────────────────────────────────────────────────────────
SECURITY_SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH"]
SECURITY_HIGH_PENALISES_SCORE = True  # HIGH severity findings reduce composite score

# ── Output ─────────────────────────────────────────────────────────────────
PIPELINE_VERSION = "v0.1.0"
OUTPUT_FORMAT = "jsonl"             # Training-ready output format
