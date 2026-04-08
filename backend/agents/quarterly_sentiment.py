# quarterly_sentiment.py

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from .yahoo_finance_agent import YahooFinanceAgent
from .retrieval_pipeline import (
    FilingRetrievalPipeline,
    OpenAIChatGenerationProvider,
    OpenAIEmbeddingProvider,
    build_chunk_records_from_prepared_filings,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


RISK_SECTION = "RISK"
MDA_SECTION = "MDA"
UNKNOWN_SECTION = "UNKNOWN"

SECTION_LABELS = {RISK_SECTION, MDA_SECTION}

ITEM_1A_RE = re.compile(r"\bitem\s+1a\b", re.IGNORECASE)
ITEM_2_RE = re.compile(r"\bitem\s+2\b", re.IGNORECASE)
LABEL_PATTERN = re.compile(r"\b(bullish|neutral|bearish)\b", re.IGNORECASE)


def build_classification_prompt(ticker: str, filing: Dict) -> str:
    return (
        f"You are a financial analyst.\n\n"
        f"Task: Based ONLY on the retrieved 10-Q excerpts for {ticker} "
        f"(quarter: {filing.get('quarter', 'N/A')}, filing date: {filing.get('filing_date', 'N/A')}), "
        f"classify the stock outlook.\n\n"
        f"Return ONLY one word from this list:\n"
        f"Bullish\n"
        f"Neutral\n"
        f"Bearish\n\n"
        f"Do not explain your answer."
    )


def normalize_label(text: str) -> str:
    if not text:
        return "Neutral"

    match = LABEL_PATTERN.search(text.strip())
    if not match:
        return "Neutral"

    return match.group(1).capitalize()


def label_from_return_percent(return_percent: Optional[float]) -> str:
    if return_percent is None:
        return "Unknown"

    if return_percent > 5:
        return "Bullish"
    if return_percent < -5:
        return "Bearish"
    return "Neutral"


def compute_actual_label_from_yfinance(
    ticker: str,
    filing_date: str,
    horizon_days: int = 90,
) -> Dict:
    """
    Compute actual label from price movement after the filing date.

    Logic:
      - start price: first available close on/after the filing date
      - end price: last available close within the next `horizon_days`
    """
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
    except Exception as e:
        return {
            "actual_label": "Unknown",
            "actual_return_percent": None,
            "actual_price_start": None,
            "actual_price_end": None,
            "error": str(e),
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
    actual_label = label_from_return_percent(return_percent)

    return {
        "actual_label": actual_label,
        "actual_return_percent": round(return_percent, 2),
        "actual_price_start": round(start_price, 4),
        "actual_price_end": round(end_price, 4),
        "error": None,
    }


def _detect_section_from_chunk_text(chunk_text: str) -> str:
    """
    Detect a section label from a chunk's text.

    For 10-Q:
      Item 1A -> Risk Factors
      Item 2  -> Management's Discussion and Analysis
    """
    if not chunk_text:
        return UNKNOWN_SECTION

    text = chunk_text.strip()

    if ITEM_1A_RE.search(text):
        return RISK_SECTION

    if ITEM_2_RE.search(text):
        return MDA_SECTION

    return UNKNOWN_SECTION


def build_section_map_from_prepared_filing(filing: Dict) -> Dict[str, str]:
    """
    Build a mapping from chunk_id -> inferred section label for ONE filing.

    This walks through prose chunks in order and propagates the current
    section until a new heading is detected.
    """
    section_map: Dict[str, str] = {}

    accession_number = filing.get("accession_number", "")
    prepared_chunk_data = filing.get("prepared_chunk_data", {})
    prose_chunks = prepared_chunk_data.get("prose_chunks", [])

    current_section = UNKNOWN_SECTION

    for index, chunk_text in enumerate(prose_chunks):
        chunk_id = f"{accession_number}:prose:{index}"
        detected_section = _detect_section_from_chunk_text(chunk_text)

        if detected_section in SECTION_LABELS:
            current_section = detected_section

        section_map[chunk_id] = current_section

    return section_map


def _chunk_order_key(chunk_id: str) -> Tuple[int, str]:
    try:
        index_part = int(chunk_id.rsplit(":", 1)[-1])
    except Exception:
        index_part = 0
    return index_part, chunk_id


def retrieve_relevant_chunks_for_filing(
    pipeline: FilingRetrievalPipeline,
    section_map: Dict[str, str],
    filing: Dict,
    query: str,
    retrieval_k: int = 20,
    top_k: int = 8,
) -> List[Dict]:
    """
    Retrieve relevant chunks for one filing, then keep only prose chunks from
    MDA / Risk sections.
    """
    accession_number = filing.get("accession_number", "")
    retrieval = pipeline.search(query=query, k=retrieval_k)

    if not retrieval.get("success"):
        return []

    selected: List[Dict] = []
    seen_chunk_ids = set()

    for item in retrieval.get("results", []):
        if item.get("accession_number") != accession_number:
            continue
        if item.get("source_type") != "prose":
            continue

        chunk_id = item.get("chunk_id", "")
        section = section_map.get(chunk_id, UNKNOWN_SECTION)

        if section not in SECTION_LABELS:
            continue

        if chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.add(chunk_id)
        item = dict(item)
        item["section"] = section
        selected.append(item)

    if not selected:
        fallback: List[Dict] = []
        for chunk_record in pipeline.chunk_records:
            if chunk_record.accession_number != accession_number:
                continue
            if chunk_record.source_type != "prose":
                continue

            section = section_map.get(chunk_record.chunk_id, UNKNOWN_SECTION)
            if section not in SECTION_LABELS:
                continue

            record_dict = chunk_record.to_dict()
            record_dict["section"] = section
            fallback.append(record_dict)

        fallback.sort(
            key=lambda x: (
                _chunk_order_key(x.get("chunk_id", ""))[0],
                x.get("chunk_id", ""),
            )
        )
        selected = fallback

    selected.sort(
        key=lambda x: (
            -float(x.get("score", 0.0) or 0.0),
            _chunk_order_key(x.get("chunk_id", ""))[0],
            x.get("chunk_id", ""),
        )
    )

    return selected[:top_k]


def analyze_single_filing(
    ticker: str,
    filing: Dict,
    pipeline: FilingRetrievalPipeline,
    section_map: Dict[str, str],
    generation_provider: OpenAIChatGenerationProvider,
    retrieval_k: int = 20,
    top_k: int = 8,
    horizon_days: int = 90,
) -> Dict:
    query = "management discussion and analysis risk factors financial performance outlook"
    selected_chunks = retrieve_relevant_chunks_for_filing(
        pipeline=pipeline,
        section_map=section_map,
        filing=filing,
        query=query,
        retrieval_k=retrieval_k,
        top_k=top_k,
    )

    if not selected_chunks:
        predicted_label = "Neutral"
        model_error = "No relevant chunks retrieved."
    else:
        prompt = build_classification_prompt(ticker, filing)
        raw_prediction = generation_provider.generate_answer(
            question=prompt,
            retrieved_chunks=selected_chunks,
        )
        predicted_label = normalize_label(raw_prediction)
        model_error = None

    actual_result = compute_actual_label_from_yfinance(
        ticker=ticker,
        filing_date=filing.get("filing_date", ""),
        horizon_days=horizon_days,
    )

    return {
        "ticker": ticker.upper(),
        "quarter": filing.get("quarter", "N/A"),
        "filing_date": filing.get("filing_date", "N/A"),
        "accession_number": filing.get("accession_number", "N/A"),
        "form_type": filing.get("form_type", "10-Q"),
        "predicted_label": predicted_label,
        "actual_label": actual_result.get("actual_label", "Unknown"),
        "actual_return_percent": actual_result.get("actual_return_percent"),
        "actual_price_start": actual_result.get("actual_price_start"),
        "actual_price_end": actual_result.get("actual_price_end"),
        "chunks_used": len(selected_chunks),
        "selected_sections": sorted(
            {chunk.get("section", UNKNOWN_SECTION) for chunk in selected_chunks}
        ),
        "model_error": model_error,
        "price_error": actual_result.get("error"),
    }


def analyze_ticker_10q_sentiment(
    ticker: str,
    openai_api_key: Optional[str] = None,
    num_quarters: int = 4,
    retrieval_k: int = 20,
    top_k: int = 8,
    horizon_days: int = 90,
    embedding_model: str = "text-embedding-3-small",
    generation_model: str = "gpt-4o-mini",
) -> List[Dict]:
    """
    End-to-end pipeline for one ticker:
      1. Fetch and chunk recent 10-Q filings
      2. Build section map per filing (MDA / Risk)
      3. Index each filing separately
      4. Retrieve only relevant chunks from that filing
      5. Ask OpenAI for a sentiment label
      6. Compute actual label from post-filing price movement
      7. Return a list of result dictionaries
    """
    agent = YahooFinanceAgent()

    prepared_filings = agent.prepare_recent_10q_filings_for_chunking(
        ticker=ticker,
        num_quarters=num_quarters,
    )

    if not prepared_filings.get("success"):
        return [
            {
                "ticker": ticker.upper(),
                "predicted_label": "Error",
                "actual_label": "Unknown",
                "error": prepared_filings.get("error", "Failed to prepare filings."),
            }
        ]

    generation_provider = OpenAIChatGenerationProvider(
        api_key=openai_api_key,
        model=generation_model,
    )

    results: List[Dict] = []

    for filing in prepared_filings.get("filings", []):
        single_prepared = {
            "ticker": ticker,
            "filings": [filing],
        }

        chunk_records = build_chunk_records_from_prepared_filings(single_prepared)
        if not chunk_records:
            results.append(
                {
                    "ticker": ticker.upper(),
                    "quarter": filing.get("quarter", "N/A"),
                    "filing_date": filing.get("filing_date", "N/A"),
                    "accession_number": filing.get("accession_number", "N/A"),
                    "form_type": filing.get("form_type", "10-Q"),
                    "predicted_label": "Neutral",
                    "actual_label": "Unknown",
                    "actual_return_percent": None,
                    "actual_price_start": None,
                    "actual_price_end": None,
                    "chunks_used": 0,
                    "selected_sections": [],
                    "model_error": "No chunks were built for this filing.",
                    "price_error": None,
                }
            )
            continue

        section_map = build_section_map_from_prepared_filing(filing)

        embedding_provider = OpenAIEmbeddingProvider(
            api_key=openai_api_key,
            model=embedding_model,
        )
        pipeline = FilingRetrievalPipeline(embedding_provider)
        index_result = pipeline.index_chunks(chunk_records)

        if not index_result.get("success"):
            results.append(
                {
                    "ticker": ticker.upper(),
                    "quarter": filing.get("quarter", "N/A"),
                    "filing_date": filing.get("filing_date", "N/A"),
                    "accession_number": filing.get("accession_number", "N/A"),
                    "form_type": filing.get("form_type", "10-Q"),
                    "predicted_label": "Neutral",
                    "actual_label": "Unknown",
                    "actual_return_percent": None,
                    "actual_price_start": None,
                    "actual_price_end": None,
                    "chunks_used": 0,
                    "selected_sections": [],
                    "model_error": "Failed to index chunks.",
                    "price_error": None,
                }
            )
            continue

        result = analyze_single_filing(
            ticker=ticker,
            filing=filing,
            pipeline=pipeline,
            section_map=section_map,
            generation_provider=generation_provider,
            retrieval_k=retrieval_k,
            top_k=top_k,
            horizon_days=horizon_days,
        )
        results.append(result)

    return results


if __name__ == "__main__":
    # Example usage:
    # export OPENAI_API_KEY=...
    # results = analyze_ticker_10q_sentiment("AAPL")
    # print(results)
    pass