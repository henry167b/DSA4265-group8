#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from RAG_test.common import (
    COMPANIES,
    chunked_filings_path,
    ensure_data_dirs,
    ensure_repo_on_path,
    load_json,
    raw_filings_path,
    stub_optional_market_data_dependencies,
    write_json,
)

ensure_repo_on_path()
stub_optional_market_data_dependencies()

from backend.agents.filing_chunker import prepare_filing_html_for_chunking


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chunk locally cached 10-Q filings and store the chunk artifacts on disk."
    )
    parser.add_argument("--prose-chunk-size", type=int, default=600)
    parser.add_argument("--prose-chunk-overlap", type=int, default=100)
    parser.add_argument("--table-window", type=int, default=10)
    parser.add_argument("--table-overlap", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_data_dirs()

    for company in COMPANIES:
        ticker = company["ticker"]
        source_path = raw_filings_path(ticker)
        if not source_path.exists():
            print(f"Skipping {ticker}: missing raw filing cache at {source_path}")
            continue

        cached_filings = load_json(source_path)
        prepared_filings = []
        for filing in cached_filings.get("filings", []):
            prepared_filing = dict(filing)
            prepared_filing["prepared_chunk_data"] = prepare_filing_html_for_chunking(
                filing.get("document_html") or "",
                prose_chunk_size=args.prose_chunk_size,
                prose_chunk_overlap=args.prose_chunk_overlap,
                table_window=args.table_window,
                table_overlap=args.table_overlap,
            )
            prepared_filings.append(prepared_filing)

        destination = chunked_filings_path(ticker)
        chunked_payload = {
            "company_name": company["company_name"],
            "ticker": ticker,
            "cached_at_utc": datetime.now(timezone.utc).isoformat(),
            "success": cached_filings.get("success", False),
            "error": cached_filings.get("error"),
            "chunking_config": {
                "prose_chunk_size": args.prose_chunk_size,
                "prose_chunk_overlap": args.prose_chunk_overlap,
                "table_window": args.table_window,
                "table_overlap": args.table_overlap,
            },
            "filings": prepared_filings,
        }
        write_json(destination, chunked_payload)
        total_chunks = sum(
            len(filing.get("prepared_chunk_data", {}).get("chunks", []))
            for filing in prepared_filings
        )
        print(
            f"Cached chunked filings for {ticker} at {destination} ({total_chunks} total chunks)"
        )


if __name__ == "__main__":
    main()
