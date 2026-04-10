import os
import re
import sys
from typing import Dict, List

import pandas as pd
import yfinance as yf

# Make sure the project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.agents.filing_chunker import prepare_filing_html_for_chunking
from backend.agents.financial_data_agent import YahooFinanceAgent
from backend.agents.quarterly_sentiment_tool import QuarterlySentimentTool
from backend.agents.retrieval_pipeline import (
    FilingRetrievalPipeline,
    OpenAIChatGenerationProvider,
    OpenAIEmbeddingProvider,
    build_chunk_records_from_prepared_filings,
)


RISK_SECTION = "RISK"
MDA_SECTION = "MDA"
UNKNOWN_SECTION = "UNKNOWN"


def _canonicalize_section_name(section_name: str) -> str:
    if not section_name:
        return ""
    normalized = section_name.strip().lower()
    normalized = normalized.replace("\u2019", "'").replace("`", "'")
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _normalize_section_name(chunk: Dict) -> str:
    return (
        chunk.get("section_name")
        or chunk.get("section")
        or chunk.get("metadata", {}).get("section_name")
        or ""
    ).strip()


def _section_bucket_from_name(section_name: str) -> str:
    s = _canonicalize_section_name(section_name)
    if "risk factor" in s or "risk factors" in s or "item 1a" in s:
        return RISK_SECTION
    if (
        "management's discussion and analysis" in s
        or "managements discussion and analysis" in s
        or "management discussion and analysis" in s
        or "md&a" in s
        or "md and a" in s
        or "item 2" in s
    ):
        return MDA_SECTION
    return UNKNOWN_SECTION


def _filter_filing_to_relevant_prose_chunks(filing: Dict) -> tuple[Dict, List[str]]:
    prepared_chunk_data = filing.get("prepared_chunk_data", {}) or {}
    prose_chunk_records = prepared_chunk_data.get("prose_chunk_records", []) or []

    filtered_records: List[Dict] = []
    has_risk = False
    has_mda = False

    for record in prose_chunk_records:
        section_name = _normalize_section_name(record)
        if not section_name:
            continue

        bucket = _section_bucket_from_name(section_name)
        if bucket == RISK_SECTION:
            filtered_records.append(record)
            has_risk = True
        elif bucket == MDA_SECTION:
            filtered_records.append(record)
            has_mda = True

    selected_sections: List[str] = []
    if has_risk:
        selected_sections.append(RISK_SECTION)
    if has_mda:
        selected_sections.append(MDA_SECTION)

    filtered_prepared_chunk_data = dict(prepared_chunk_data)
    filtered_prepared_chunk_data["prose_chunk_records"] = filtered_records
    filtered_prepared_chunk_data["prose_chunks"] = []
    filtered_prepared_chunk_data["table_chunk_records"] = []
    filtered_prepared_chunk_data["table_chunks"] = []

    filtered_filing = dict(filing)
    filtered_filing["prepared_chunk_data"] = filtered_prepared_chunk_data
    filtered_filing["selected_sections"] = selected_sections
    filtered_filing["pre_filtered"] = True
    return filtered_filing, selected_sections


def label_from_return_percent(
    return_percent: float | None,
    price_label_threshold: float = 0.10,
) -> str:
    if return_percent is None:
        return "Unknown"

    threshold_percent = price_label_threshold * 100.0
    if return_percent > threshold_percent:
        return "Bullish"
    if return_percent < -threshold_percent:
        return "Bearish"
    return "Neutral"


def compute_actual_label_from_yfinance(
    ticker: str,
    filing_date: str,
    horizon_days: int = 90,
    price_label_threshold: float = 0.10,
) -> Dict:
    from datetime import datetime, timedelta

    try:
        start_dt = datetime.strptime(filing_date, "%Y-%m-%d").date()
    except Exception:
        return {
            "actual_label": "Unknown",
            "actual_return_percent": None,
            "actual_price_start": None,
            "actual_price_end": None,
            "error": f"Invalid filing_date format: {filing_date}",
        }

    end_dt = start_dt + timedelta(days=horizon_days)

    try:
        hist = yf.download(
            ticker,
            start=str(start_dt),
            end=str(end_dt + timedelta(days=1)),
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:
        return {
            "actual_label": "Unknown",
            "actual_return_percent": None,
            "actual_price_start": None,
            "actual_price_end": None,
            "error": str(exc),
        }

    if hist is None or hist.empty:
        return {
            "actual_label": "Unknown",
            "actual_return_percent": None,
            "actual_price_start": None,
            "actual_price_end": None,
            "error": "No price data available in the selected window.",
        }

    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    if "Close" not in hist.columns or hist["Close"].dropna().empty:
        return {
            "actual_label": "Unknown",
            "actual_return_percent": None,
            "actual_price_start": None,
            "actual_price_end": None,
            "error": "Close price data unavailable.",
        }

    close_prices = hist["Close"].dropna()
    if len(close_prices) < 2:
        value = float(close_prices.iloc[0]) if len(close_prices) == 1 else None
        return {
            "actual_label": "Unknown",
            "actual_return_percent": None,
            "actual_price_start": value,
            "actual_price_end": value,
            "error": "Not enough price points to compute return.",
        }

    start_price = float(close_prices.iloc[0])
    end_price = float(close_prices.iloc[-1])

    if start_price == 0:
        return {
            "actual_label": "Unknown",
            "actual_return_percent": None,
            "actual_price_start": start_price,
            "actual_price_end": end_price,
            "error": "Start price is zero.",
        }

    return_percent = ((end_price - start_price) / start_price) * 100.0
    actual_label = label_from_return_percent(
        return_percent,
        price_label_threshold=price_label_threshold,
    )

    return {
        "actual_label": actual_label,
        "actual_return_percent": round(return_percent, 2),
        "actual_price_start": round(start_price, 4),
        "actual_price_end": round(end_price, 4),
        "error": None,
    }


def _preview_text(text: str, limit: int = 240) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    return normalized[:limit]


def _prepare_recent_filings_for_test(
    yahoo_agent: YahooFinanceAgent,
    ticker: str,
    num_quarters: int,
) -> Dict:
    legacy_prepare = getattr(yahoo_agent, "prepare_recent_10q_filings_for_chunking", None)
    if callable(legacy_prepare):
        return legacy_prepare(ticker=ticker, num_quarters=num_quarters)

    filings_payload = yahoo_agent.get_recent_10q_filings(
        ticker=ticker,
        num_quarters=num_quarters,
        include_document_html=True,
    )

    if not filings_payload.get("success"):
        return {
            "success": False,
            "ticker": ticker.upper(),
            "error": filings_payload.get("error", "Failed to fetch recent filings."),
            "filings": [],
        }

    prepared_filings: List[Dict] = []
    for filing in filings_payload.get("filings", []):
        filing_copy = dict(filing)
        html_content = filing_copy.get("document_html")

        if not html_content:
            document_url = filing_copy.get("document_url")
            if document_url:
                html_content = yahoo_agent.get_full_10q_document(document_url)

        if not html_content:
            continue

        filing_copy["prepared_chunk_data"] = prepare_filing_html_for_chunking(html_content)
        prepared_filings.append(filing_copy)

    if not prepared_filings:
        return {
            "success": False,
            "ticker": ticker.upper(),
            "error": "No filings could be prepared for chunking.",
            "filings": [],
        }

    return {
        "success": True,
        "ticker": ticker.upper(),
        "filings": prepared_filings,
        "prepared_for_chunking": True,
    }


def _retrieve_chunks_for_filing(
    ticker: str,
    filing: Dict,
    openai_api_key: str | None,
    embedding_model: str = "text-embedding-3-small",
    retrieval_k: int = 20,
) -> tuple[list[dict], int]:
    filtered_filing, _ = _filter_filing_to_relevant_prose_chunks(filing)
    single_prepared_filings = {
        "ticker": ticker,
        "filings": [filtered_filing],
    }

    chunk_records = build_chunk_records_from_prepared_filings(single_prepared_filings)
    chunk_count = len(chunk_records)

    embedding_provider = OpenAIEmbeddingProvider(
        api_key=openai_api_key,
        model=embedding_model,
    )
    pipeline = FilingRetrievalPipeline(embedding_provider)
    pipeline.index_chunks(chunk_records)

    query = (
        "risk factors management's discussion and analysis "
        "results of operations financial condition liquidity outlook"
    )
    retrieval = pipeline.search(query=query, k=max(retrieval_k, 16))
    if not retrieval.get("success"):
        return [], chunk_count

    return retrieval.get("results", []), chunk_count


def main() -> None:
    ticker = "AAPL"
    num_quarters = 4
    horizon_days = 90
    price_label_threshold = 0.10
    embedding_model = "text-embedding-3-small"
    generation_model = "gpt-4o-mini"
    retrieval_k = 20
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    print(f"\nRunning modular tool integration test for {ticker} ({num_quarters} quarter)")
    print("=" * 120)

    yahoo_agent = YahooFinanceAgent()
    tool = QuarterlySentimentTool()

    prepared = _prepare_recent_filings_for_test(
        yahoo_agent=yahoo_agent,
        ticker=ticker,
        num_quarters=num_quarters,
    )
    if not prepared.get("success"):
        print(f"Failed to prepare filings: {prepared.get('error')}")
        return

    filings = prepared.get("filings", [])
    if not filings:
        print("No filings were prepared.")
        return

    generation_provider = OpenAIChatGenerationProvider(
        api_key=openai_api_key,
        model=generation_model,
    )

    for filing in filings:
        candidate_chunks, chunk_count = _retrieve_chunks_for_filing(
            ticker=ticker,
            filing=filing,
            openai_api_key=openai_api_key,
            embedding_model=embedding_model,
            retrieval_k=retrieval_k,
        )

        actual_result = compute_actual_label_from_yfinance(
            ticker=ticker,
            filing_date=filing.get("filing_date", ""),
            horizon_days=horizon_days,
            price_label_threshold=price_label_threshold,
        )

        tool_output = tool.analyze_single_filing(
            ticker=ticker,
            filing=filing,
            candidate_chunks=candidate_chunks,
            actual_result=actual_result,
            chunk_count=chunk_count,
            generation_provider=generation_provider,
        )

        print(f"Quarter: {tool_output.get('quarter')}")
        print(f"Filing Date: {tool_output.get('filing_date')}")
        print(f"Predicted Label: {tool_output.get('predicted_label')}")
        print(f"Actual Label: {tool_output.get('actual_label')}")
        print(f"Realized % Change: {tool_output.get('realized_next_quarter_pct_change')}")
        print(f"Chunks Used: {tool_output.get('chunks_used')}")
        print(f"Selected Sections: {tool_output.get('selected_sections')}")
        print(f"Model Error: {tool_output.get('model_error')}")
        print(f"Price Error: {tool_output.get('price_error')}")

        sample_chunk = (tool_output.get("retrieved_chunks") or [{}])[0]
        if sample_chunk:
            print(f"Sample Chunk Preview: {_preview_text(sample_chunk.get('text', ''))}")
        print("-" * 120)

    print("=" * 120)


if __name__ == "__main__":
    main()
