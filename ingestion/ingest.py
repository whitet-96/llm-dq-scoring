"""
ingestion/ingest.py

Pulls samples from HuggingFace datasets and saves as raw Parquet files.
Supports: The Stack v1, CodeSearchNet, HumanEval.

Usage:
    python -m ingestion.ingest --source the-stack --language python --n 1000
"""

import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm
from rich.console import Console
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DIR, DEFAULT_SAMPLE_SIZE, DEFAULT_LANGUAGE

console = Console()

# ── Source configs ─────────────────────────────────────────────────────────

SOURCES = {
    "the-stack": {
        "dataset_name": "bigcode/the-stack",
        "content_field": "content",
        "language_field": "lang",
        "extra_fields": ["size", "avg_line_length", "max_line_length", "alphanum_fraction"],
    },
    "codesearchnet": {
        "dataset_name": "code_search_net",
        "content_field": "func_code_string",
        "language_field": "language",
        "extra_fields": ["func_documentation_string", "repository_name"],
    },
    "humaneval": {
        "dataset_name": "openai_humaneval",
        "content_field": "prompt",
        "language_field": None,   # always Python
        "extra_fields": ["canonical_solution", "test", "entry_point"],
    },
}

# ── Core ingestion ─────────────────────────────────────────────────────────

def ingest(source: str, language: str, n: int) -> Path:
    """
    Stream n samples from source, save as Parquet.
    Returns path to saved file.
    """
    if source not in SOURCES:
        raise ValueError(f"Unknown source: {source}. Choose from: {list(SOURCES.keys())}")

    cfg = SOURCES[source]
    console.print(f"\n[bold cyan]Ingesting[/bold cyan] {n} samples from [yellow]{source}[/yellow] ({language})")

    # Load dataset in streaming mode — avoids downloading entire dataset
    try:
        if source == "the-stack":
            # The Stack v1 stores data in data/{language}/ folders.
            # Use data_files to load only the target language partition.
            ds = load_dataset(
                cfg["dataset_name"],
                data_files=f"data/{language}/train-*.parquet",
                split="train",
                streaming=True,
            )
        elif source == "codesearchnet":
            ds = load_dataset(
                cfg["dataset_name"],
                language,
                split="train",
                streaming=True,
            )
        else:
            ds = load_dataset(cfg["dataset_name"], split="test")
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        raise

    # Collect samples
    records = []
    fields = [cfg["content_field"]] + (cfg["extra_fields"] or [])
    if cfg["language_field"]:
        fields.append(cfg["language_field"])

    for i, sample in enumerate(tqdm(ds, total=n, desc="Streaming samples")):
        if i >= n:
            break
        record = {
            "id": f"{source}_{language}_{i:06d}",
            "source": source,
            "language": language,
            "content": sample.get(cfg["content_field"], ""),
            "ingested_at": datetime.utcnow().isoformat(),
        }
        # Add extra fields where present
        for field in cfg["extra_fields"] or []:
            record[field] = sample.get(field, None)
        records.append(record)

    console.print(f"[green]✓[/green] Collected {len(records)} samples")

    # Save as Parquet
    output_path = _save_parquet(records, source, language)
    console.print(f"[green]✓[/green] Saved to {output_path}")
    return output_path


def _save_parquet(records: list, source: str, language: str) -> Path:
    """Save records as a partitioned Parquet file."""
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    output_dir = RAW_DIR / f"source={source}" / f"language={language}" / f"date={date_str}"
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(records)
    output_path = output_dir / "data.parquet"
    pq.write_table(table, output_path)
    return output_path


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest code samples from public datasets")
    parser.add_argument("--source", default="the-stack", choices=list(SOURCES.keys()))
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--n", type=int, default=DEFAULT_SAMPLE_SIZE)
    args = parser.parse_args()

    ingest(args.source, args.language, args.n)
