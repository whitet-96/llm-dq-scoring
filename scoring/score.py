"""
scoring/score.py

Applies multi-dimensional quality scoring to cleaned data.
Outputs scored JSONL — the training-ready format.

Scores computed:
  - syntactic_score  (0-1): validity, complexity, line quality
  - doc_score        (0-1): docstring/comment coverage
  - security_score   (0-1): 1.0 = clean, penalised per finding severity
  - llm_score        (1-5): Claude API rating (sampled subset only)
  - composite_score  (0-1): weighted average of all signals

Usage:
    python -m scoring.score --input data/processed/ --output data/scored/
"""

import json
import subprocess
import tempfile
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from rich.console import Console
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    SCORED_DIR, SCORE_WEIGHTS, PIPELINE_VERSION,
    LLM_SCORE_SAMPLE_SIZE, LLM_MODEL, LLM_MAX_TOKENS,
    QUALITY_THRESHOLD
)

console = Console()


# ── Syntactic Score ────────────────────────────────────────────────────────

def syntactic_score(row: pd.Series) -> float:
    """
    Score based on structural code quality signals.
    Returns 0.0–1.0.
    """
    score = 1.0

    # Hard fail: unparseable
    if not row.get("is_parseable", False):
        return 0.0

    # Penalise very short files
    if row.get("num_lines", 0) < 10:
        score -= 0.15

    # Penalise long average line length
    avg_ll = row.get("avg_line_length", 0)
    if avg_ll > 120:
        score -= 0.10
    elif avg_ll > 80:
        score -= 0.05

    # Penalise very low alphanum fraction (likely binary/generated garbage)
    alpha = row.get("alphanum_fraction", 1.0)
    if alpha < 0.35:
        score -= 0.20

    # Bonus for having functions/classes (substantive code)
    if row.get("num_functions", 0) > 0 or row.get("num_classes", 0) > 0:
        score += 0.05

    return max(0.0, min(1.0, score))


# ── Documentation Score ────────────────────────────────────────────────────

def doc_score(row: pd.Series) -> float:
    """
    Score based on documentation quality.
    Returns 0.0–1.0.
    """
    score = 0.0

    if row.get("has_docstring", False):
        score += 0.50

    comment_ratio = row.get("comment_ratio", 0.0)
    if comment_ratio >= 0.10:
        score += 0.30
    elif comment_ratio >= 0.05:
        score += 0.15

    # Penalise zero documentation on substantial files
    num_functions = row.get("num_functions", 0)
    if num_functions >= 3 and not row.get("has_docstring", False):
        score -= 0.10

    return max(0.0, min(1.0, score))


# ── Security Score ─────────────────────────────────────────────────────────

def security_score(code: str) -> dict:
    """
    Run Bandit static analysis. Returns score + list of findings.
    Score: 1.0 = clean, penalised per HIGH/MEDIUM/LOW finding.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            ["bandit", "-q", "-f", "json", tmp_path],
            capture_output=True, text=True, timeout=10
        )
        findings = []
        score = 1.0

        if result.stdout:
            data = json.loads(result.stdout)
            for issue in data.get("results", []):
                severity = issue.get("issue_severity", "LOW")
                findings.append({
                    "severity": severity,
                    "text": issue.get("issue_text", ""),
                    "line": issue.get("line_number", 0),
                })
                # Penalise by severity
                if severity == "HIGH":
                    score -= 0.30
                elif severity == "MEDIUM":
                    score -= 0.15
                elif severity == "LOW":
                    score -= 0.05

        return {
            "security_score": max(0.0, score),
            "security_findings": findings,
            "security_finding_count": len(findings),
            "security_high_count": sum(1 for f in findings if f["severity"] == "HIGH"),
        }

    except Exception as e:
        return {
            "security_score": 0.5,   # Unknown — don't penalise fully
            "security_findings": [],
            "security_finding_count": 0,
            "security_high_count": 0,
        }


# ── LLM Score ──────────────────────────────────────────────────────────────

def llm_score_batch(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """
    Sample up to sample_size rows and score via Claude API.
    Adds llm_score column (1-5, None if not sampled).
    """
    try:
        import anthropic
    except ImportError:
        console.print("[yellow]  anthropic package not installed — skipping LLM scoring[/yellow]")
        df["llm_score"] = None
        return df

    client = anthropic.Anthropic()
    sample_idx = df.sample(min(sample_size, len(df)), random_state=42).index
    df["llm_score"] = None

    console.print(f"  [cyan]LLM scoring[/cyan] {len(sample_idx)} samples via Claude API...")

    for idx in tqdm(sample_idx, desc="  Claude scoring"):
        code = df.loc[idx, "content"][:2000]  # Truncate to save tokens
        prompt = f"""Rate this code snippet for AI model training utility on a scale of 1-5.

Criteria:
- 5: Excellent — clear, well-documented, instructive, demonstrates good practices
- 4: Good — readable and correct, some documentation
- 3: Average — functional but limited learning value
- 2: Poor — hard to read, minimal value for training
- 1: Bad — buggy, obfuscated, or harmful patterns

Respond ONLY with valid JSON, no preamble:
{{"score": <1-5>, "reason": "<one sentence>", "flags": []}}

Code:
```
{code}
```"""

        try:
            response = client.messages.create(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("```")[1].lstrip("json").strip()
            result = json.loads(text)
            df.loc[idx, "llm_score"] = result.get("score")
            df.loc[idx, "llm_reason"] = result.get("reason", "")
        except Exception as e:
            console.print(f"[yellow]  LLM score failed for idx {idx}: {e}[/yellow]")

    return df


# ── Composite Score ────────────────────────────────────────────────────────

def composite_score(row: pd.Series) -> float:
    """
    Weighted average. If llm_score is None (not sampled), redistribute weight.
    """
    weights = SCORE_WEIGHTS.copy()
    scores = {
        "syntactic": row.get("syntactic_score", 0),
        "documentation": row.get("doc_score", 0),
        "security": row.get("security_score", 0),
    }

    llm = row.get("llm_score")
    if llm is not None:
        scores["llm"] = (llm - 1) / 4  # Normalise 1-5 to 0-1
    else:
        # Redistribute LLM weight across other scores
        llm_w = weights.pop("llm")
        total_w = sum(weights.values())
        weights = {k: v + (v / total_w) * llm_w for k, v in weights.items()}

    total = sum(scores[k] * weights.get(k, 0) for k in scores)
    return round(total, 4)


# ── Main scoring pipeline ──────────────────────────────────────────────────

def score(input_dir: Path, output_dir: Path):
    """Full scoring pipeline."""
    console.print(f"\n[bold cyan]Scoring[/bold cyan] reading from {input_dir}")

    parquet_files = list(input_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files in {input_dir}")

    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    console.print(f"  Loaded {len(df)} samples")

    # Rule-based scores
    console.print("  [cyan]Computing syntactic scores...[/cyan]")
    df["syntactic_score"] = df.apply(syntactic_score, axis=1)

    console.print("  [cyan]Computing documentation scores...[/cyan]")
    df["doc_score"] = df.apply(doc_score, axis=1)

    console.print("  [cyan]Running security analysis (Bandit)...[/cyan]")
    sec_results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Bandit"):
        sec_results.append(security_score(row["content"]))
    sec_df = pd.DataFrame(sec_results, index=df.index)
    df = pd.concat([df, sec_df], axis=1)

    # LLM scoring (sampled)
    df = llm_score_batch(df, LLM_SCORE_SAMPLE_SIZE)

    # Composite
    console.print("  [cyan]Computing composite scores...[/cyan]")
    df["composite_score"] = df.apply(composite_score, axis=1)
    df["above_threshold"] = df["composite_score"] >= QUALITY_THRESHOLD
    df["pipeline_version"] = PIPELINE_VERSION
    df["scored_at"] = datetime.utcnow().isoformat()

    # Save as JSONL (training-ready format)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scored.jsonl"
    df.to_json(output_path, orient="records", lines=True)

    # Save as Parquet (for dashboard)
    parquet_path = output_dir / "scored.parquet"
    df.to_parquet(parquet_path, index=False)

    # Summary
    n_pass = df["above_threshold"].sum()
    n_security = (df["security_finding_count"] > 0).sum()
    console.print(f"\n[bold green]Scoring complete[/bold green]")
    console.print(f"  Total scored:          {len(df)}")
    console.print(f"  Pass threshold (≥{QUALITY_THRESHOLD}): {n_pass} ({n_pass/len(df)*100:.1f}%)")
    console.print(f"  Security findings:     {n_security} ({n_security/len(df)*100:.1f}%)")
    console.print(f"  Output: {output_path}")
    console.print(f"  Output: {parquet_path}")

    return output_path


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed")
    parser.add_argument("--output", default="data/scored")
    args = parser.parse_args()
    score(Path(args.input), Path(args.output))
