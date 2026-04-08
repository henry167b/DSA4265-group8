#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from RAG_test.common import (
    COMPANIES,
    ensure_data_dirs,
    ensure_repo_on_path,
    raw_filings_path,
    stub_optional_market_data_dependencies,
    write_json,
)

ensure_repo_on_path()
stub_optional_market_data_dependencies()

from backend.agents.financial_data_agent import YahooFinanceAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch recent 10-Q filings for the benchmark companies and cache them locally."
    )
    parser.add_argument(
        "--num-filings",
        type=int,
        default=1,
        help="Number of most recent 10-Q filings to cache per company. Recommended: 1 for the first benchmark pass.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_data_dirs()
    agent = YahooFinanceAgent()

    for company in COMPANIES:
        ticker = company["ticker"]
        payload = agent.get_recent_10q_filings(
            ticker=ticker,
            num_quarters=args.num_filings,
            include_document_html=True,
        )

        cache_payload = {
            "company_name": company["company_name"],
            "ticker": ticker,
            "num_filings_requested": args.num_filings,
            "cached_at_utc": datetime.now(timezone.utc).isoformat(),
            "success": payload.get("success", False),
            "error": payload.get("error"),
            "filings": payload.get("filings", []),
        }
        destination = raw_filings_path(ticker)
        write_json(destination, cache_payload)
        print(
            f"Cached {len(cache_payload['filings'])} filing(s) for {ticker} at {destination}"
        )


if __name__ == "__main__":
    main()
