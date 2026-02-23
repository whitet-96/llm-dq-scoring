"""
LLM Data Quality Scoring Pipeline
Main entrypoint — runs the full pipeline end to end.

Usage:
    python main.py --sample 1000 --language python
    python main.py --stage ingest
    python main.py --stage score
    python main.py --stage all
"""

import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(stage: str, sample_size: int, language: str):
    logger.info(f"Starting pipeline | stage={stage} | sample={sample_size} | lang={language}")

    if stage in ("ingest", "all"):
        logger.info("--- Stage 1-2: Ingestion ---")
        from ingestion.ingest import ingest
        ingest(source="the-stack", language=language, n=sample_size)

    if stage in ("transform", "all"):
        logger.info("--- Stage 3: Transform & Clean ---")
        from transform.clean import run_transforms
        from config import RAW_DIR
        input_path = RAW_DIR / f"source=the-stack/language={language}"
        run_transforms(input_path=next(input_path.rglob("*.parquet")), source="the-stack", language=language)

    if stage in ("score", "all"):
        logger.info("--- Stage 4: Quality Scoring ---")
        from scoring.score import score
        from config import PROCESSED_DIR, SCORED_DIR
        score(input_dir=PROCESSED_DIR, output_dir=SCORED_DIR)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Data Quality Pipeline")
    parser.add_argument("--stage", default="all", choices=["ingest", "transform", "score", "all"])
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--language", default="python")
    args = parser.parse_args()

    run_pipeline(stage=args.stage, sample_size=args.sample, language=args.language)
