Data folder and usage
=====================

This folder contains data-related notes and small demo inputs. Do NOT commit
large raw datasets or full downloaded shards. Use the streaming helpers in
`src/preprocessing.py` to fetch the Amazon Reviews Polarity dataset directly
from the Hugging Face Hub at runtime.

Contents
--------
- `raw/` : Small demo files checked into the repo (e.g., `demo_amazon_reviews.csv`).

Guidelines
----------
- Full dataset: do not store the full Amazon Polarity data inside the repo. Use
  the HF Hub streaming API (`datasets.load_dataset(..., streaming=True)`) as
  demonstrated in the code.
- Large raw files (parquet, tar.gz) should be downloaded to a machine-local
  data directory outside the repo (e.g., `~/data/amazon_polarity/`) or stored
  in cloud storage. Add them to `.gitignore`.
- Small curated examples and test fixtures (like `demo_amazon_reviews.csv`)
  are OK to keep in `data/raw/` for reproducible quick checks and CI.

Reproducibility
---------------
Document any preprocessing steps here that are required to reproduce the
materialized validation/test subsets used in experiments.
