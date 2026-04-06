from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_FILINGS_DIR = REPO_ROOT / "RAG_test" / "data" / "raw_filings"
CHUNKED_FILINGS_DIR = REPO_ROOT / "RAG_test" / "data" / "chunked_filings"
RESULTS_DIR = REPO_ROOT / "RAG_test" / "results"
BENCHMARK_DATASET_PATH = REPO_ROOT / "RAG_test" / "benchmark_dataset.json"

COMPANIES: List[Dict[str, str]] = [
    {"company_name": "NVIDIA", "ticker": "NVDA"},
    {"company_name": "Alphabet", "ticker": "GOOG"},
    {"company_name": "Tesla", "ticker": "TSLA"},
    {"company_name": "Apple", "ticker": "AAPL"},
    {"company_name": "Meta Platforms", "ticker": "META"},
]


def ensure_repo_on_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def stub_optional_market_data_dependencies() -> None:
    """Allow SEC-only scripts to run even if Yahoo Finance deps are missing."""
    sys.modules.setdefault("yfinance", SimpleNamespace())
    sys.modules.setdefault(
        "pandas",
        SimpleNamespace(MultiIndex=type("MultiIndex", (), {})),
    )


def ensure_data_dirs() -> None:
    RAW_FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKED_FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def raw_filings_path(ticker: str) -> Path:
    return RAW_FILINGS_DIR / f"{ticker.lower()}_10q_filings.json"


def chunked_filings_path(ticker: str) -> Path:
    return CHUNKED_FILINGS_DIR / f"{ticker.lower()}_10q_chunks.json"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
