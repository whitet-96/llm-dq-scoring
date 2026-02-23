# LLM Data Quality Scoring Pipeline

Multi-stage pipeline for ingesting, deduplicating, and quality-scoring code samples from public HuggingFace datasets. Outputs training-ready JSONL with a composite quality score across four dimensions.

## Setup

```bash
pip install -r requirements.txt
# Create .env with ANTHROPIC_API_KEY and HF_TOKEN
python main.py --stage all --language python
```

## Stages

| Stage | Module | Description |
|---|---|---|
| ingest | `ingestion/ingest.py` | Stream samples from HuggingFace |
| transform | `transform/clean.py` | Dedup, validate, normalise |
| score | `scoring/score.py` | Compute 4-dimension quality score |
| dashboard | `dashboard/app.py` | Visualise results |
