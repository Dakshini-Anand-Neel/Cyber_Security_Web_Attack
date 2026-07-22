# Raw Data Directory

This directory stores the **original, unmodified** data files downloaded from the source.

## Contents

| File | Description |
|------|-------------|
| `../raw_train.csv` | Cached copy of the full Hugging Face training split |

## Data Source

- **Dataset:** `shengqin/web-attacks-long`
- **URL:** https://huggingface.co/datasets/shengqin/web-attacks-long
- **Download method:** `datasets` library or direct CSV URL

## Notes

- Do **not** modify files in this directory manually.
- Files are excluded from Git via `.gitignore` (use DVC or Git LFS for large files).
- To re-download: delete `raw_train.csv` and re-run `notebooks/01_eda.ipynb` Step 1.
