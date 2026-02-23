# Data Card — LLM Code Quality Dataset

> **Version:** v0.1.0
> **Created:** 2026-02-23
> **Pipeline version:** v0.1.0

---

## Dataset Summary

A curated, quality-scored subset of open-source Python code, processed through a multi-stage quality pipeline designed to produce training-ready data for code-focused LLMs. Sources include The Stack v1 (real-world OSS code) and HumanEval (canonical Python benchmark problems).

---

## Source Datasets

| Dataset | Source | License | Samples used |
|---|---|---|---|
| The Stack v1 | BigCode / HuggingFace | Various (permissive only) | 500 |
| HumanEval | OpenAI / HuggingFace | MIT | 164 (full test set) |

---

## Pipeline Overview

1. **Ingestion** — Streamed from HuggingFace using `data_files` partitioning, saved as Parquet
2. **Exact deduplication** — SHA256 content hash dedup
3. **Near-deduplication** — MinHash LSH (Jaccard threshold 0.80, 128 permutations) — optional, off by default for small runs
4. **Hard filters** — Line count (5–1000), alphanum fraction (≥0.25), avg line length (≤200)
5. **AST validation** — Python `ast.parse` for syntax correctness, function/class counts, docstring detection
6. **Quality scoring** — Composite score from 4 dimensions (see below)
7. **Output** — JSONL (training-ready) + Parquet (queryable)

---

## Quality Scoring Methodology

| Dimension | Weight | Method |
|---|---|---|
| Syntactic | 35% | AST validity, line length, alphanum ratio |
| Documentation | 25% | Docstring presence, comment density |
| Security | 20% | Bandit static analysis |
| LLM (sampled) | 20% | Claude API rating (1–5) on up to 500 samples |

**Composite score threshold:** Samples with score ≥ 0.70 are considered training-ready.

### Weight rationale

Syntactic quality is weighted highest (35%) because unparseable or malformed code has no value as training data regardless of other signals. Documentation (25%) is weighted second as it strongly correlates with code that is well-structured and human-reviewed. Security (20%) penalises samples with high-severity Bandit findings. LLM scoring (20%) captures holistic readability and idiom quality that rule-based signals miss, but is capped at 500 samples per run due to API cost.

---

## Dataset Statistics

| Metric | Value |
|---|---|
| Total raw samples ingested | 664 (500 The Stack + 164 HumanEval) |
| After exact deduplication | 664 |
| After hard filters | 512 |
| Syntactically valid (AST) | 501 / 512 (97.9%) |
| Pass quality threshold (≥0.7) | 76 / 512 (14.8%) |
| Security flagged | 0 / 512 (0.0%) |
| LLM scored | 500 / 512 |

---

## Known Limitations

- LLM scoring is capped at 500 samples per run due to API cost — the remaining 12 samples use a 3-dimension composite score
- Near-deduplication (MinHash LSH) is implemented but disabled by default (`DEDUP_NEAR = False` in `config.py`) — enable for larger runs
- Security scoring uses Bandit (Python only) — no multi-language static analysis
- Dataset skews toward popular open-source repos — may underrepresent niche domains
- The Stack v1 is used rather than v2, as v2 does not embed file content directly in Parquet (content lives in external blob storage)

---

## Intended Use

✅ **Suitable for:** Experimenting with training data quality pipelines, understanding quality signal distribution, demonstrating data operations methodology

⚠️ **Not suitable for:** Direct use in production model training without further review, generalising to languages beyond Python in current form

---

## Versioning

| Version | Date | Changes |
|---|---|---|
| v0.1.0 | 2026-02-23 | Initial pipeline — Python only, exact + optional near-dedup, 4-score composite, HumanEval + The Stack v1 |

---

## Reproducing This Dataset

```bash
git clone https://github.com/whitet-96/llm-dq-scoring.git
cd llm-dq-scoring
pip install -r requirements.txt
# Create .env with ANTHROPIC_API_KEY and HF_TOKEN
python main.py --stage all --sample 500 --language python
```
