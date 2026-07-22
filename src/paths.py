"""Centralized project paths — no hardcoded drive letters."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
RAW_CSV = DATA_DIR / "raw_train.csv"
