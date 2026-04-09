# quarterly_sentiment.py

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from .financial_data_agent import YahooFinanceAgent
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
TOP_PER_SECTION = 2

ITEM_1A_RE = re.compile(r"\bitem\s+1a\b", re.IGNORECASE)
ITEM_2_RE = re.compile(
    r"\bitem\s+2\b|\bmanagement'?s discussion\b|\bmanagement'?s discussion and analysis\b",
    re.IGNORECASE
)
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


def label_from_return_percent(
    return_percent: Optional[float],
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


def _detect_section_from_chunk_text(chunk_text: str) -> str:
    """
    Detect a section label from a chunk's text.

    For 10-Q:
      Item 1A -> Risk Factors
      Item 2 / Management's Discussion -> Management's Discussion and Analysis (MD&A)
    """
    if not chunk_text:
        return UNKNOWN_SECTION

    text = chunk_text.strip()

    if ITEM_1A_RE.search(text):
        return RISK_SECTION

    if ITEM_2_RE.search(text):
        return MDA_SECTION

    return UNKNOWN_SECTION


def _canonicalize_section_name(section_name: str) -> str:
    if not section_name:
        return ""
    normalized = section_name.strip().lower()
    normalized = normalized.replace("’", "'").replace("`", "'")
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


def _is_risk_factor_section_name(section_name: str) -> bool:
    s = _canonicalize_section_name(section_name)
    return (
        "risk factor" in s
        or "risk factors" in s
        or "item 1a" in s
    )


def _is_mda_section_name(section_name: str) -> bool:
    s = _canonicalize_section_name(section_name)
    return (
        "management's discussion and analysis" in s
        or "managements discussion and analysis" in s
        or "management discussion and analysis" in s
        or "md&a" in s
        or "md and a" in s
        or "item 2" in s
    )


def _section_bucket_from_name(section_name: str) -> str:
    if _is_risk_factor_section_name(section_name):
        return RISK_SECTION
    if _is_mda_section_name(section_name):
        return MDA_SECTION
    return UNKNOWN_SECTION


def _chunk_score(chunk: Dict) -> float:
    for key in ("score", "rerank_score", "hybrid_score", "dense_score", "sparse_score"):
        value = chunk.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _select_top_chunks_by_section(chunks: List[Dict], top_per_section: int = TOP_PER_SECTION) -> List[Dict]:
    risk_chunks: List[Dict] = []
    mda_chunks: List[Dict] = []

    # Re-rank within each section by score descending.
    ranked = sorted(chunks, key=_chunk_score, reverse=True)

    for chunk in ranked:
        section_name = _normalize_section_name(chunk)
        bucket = _section_bucket_from_name(section_name)

        if bucket == RISK_SECTION and len(risk_chunks) < top_per_section:
            risk_chunks.append(chunk)
        elif bucket == MDA_SECTION and len(mda_chunks) < top_per_section:
            mda_chunks.append(chunk)

        if len(risk_chunks) >= top_per_section and len(mda_chunks) >= top_per_section:
            break

    return risk_chunks + mda_chunks


def _filter_filing_to_relevant_prose_chunks(filing: Dict) -> Tuple[Dict, List[str]]:
    """
    Keep only prose chunks where section_name indicates:
      - Risk Factors
      - Management's Discussion and Analysis (MD&A)
    """
    prepared_chunk_data = filing.get("prepared_chunk_data", {}) or {}
    prose_chunk_records = prepared_chunk_data.get("prose_chunk_records", []) or []

    filtered_records: List[Dict] = []
    has_risk = False
    has_mda = False

    for record in prose_chunk_records:
        section_name = _normalize_section_name(record)
        if not section_name:
            continue  # strict: require section_name to exist

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

class QuarterlySentimentAnalyzer:
    """
    Analyze the sentiment of the latest 10-Q filings for a ticker and compare
    the model prediction against realized stock movement.
    """

    def __init__(
        self,
        yahoo_agent: Optional[YahooFinanceAgent] = None,
        top_k: int = 5,
        num_quarters: int = 4,
        retrieval_k: int = 20,
        price_label_threshold: float = 0.10,
        horizon_days: int = 90,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        generation_model: str = "gpt-4o-mini",
    ):
        self.yahoo_agent = yahoo_agent or YahooFinanceAgent()
        self.top_k = top_k
        self.num_quarters = num_quarters
        self.retrieval_k = retrieval_k
        self.price_label_threshold = price_label_threshold
        self.horizon_days = horizon_days
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.generation_model = generation_model

    def _build_pipeline_for_single_filing(self, ticker: str, filing: Dict) -> Tuple[FilingRetrievalPipeline, List[str], int]:
        filtered_filing, selected_sections = _filter_filing_to_relevant_prose_chunks(filing)

        single_prepared_filings = {
            "ticker": ticker,
            "filings": [filtered_filing],
        }

        chunk_records = build_chunk_records_from_prepared_filings(single_prepared_filings)
        chunk_count = len(chunk_records)

        embedding_provider = OpenAIEmbeddingProvider(
            api_key=self.openai_api_key,
            model=self.embedding_model,
        )
        pipeline = FilingRetrievalPipeline(embedding_provider)
        pipeline.index_chunks(chunk_records)

        return pipeline, selected_sections, chunk_count

    def _retrieve_chunks_for_filing(
        self,
        pipeline: FilingRetrievalPipeline,
        filing: Dict,
    ) -> List[Dict]:
        # Broad enough to rank both target sections, then section-aware top-2 selection.
        query = (
            "risk factors management's discussion and analysis "
            "results of operations financial condition liquidity outlook"
        )
        retrieval = pipeline.search(query=query, k=max(self.retrieval_k, 16))

        if not retrieval.get("success"):
            return []

        candidates = retrieval.get("results", [])
        return _select_top_chunks_by_section(candidates, top_per_section=TOP_PER_SECTION)

    def analyze_single_filing(self, ticker: str, filing: Dict) -> Dict:
        pipeline, selected_sections, chunk_count = self._build_pipeline_for_single_filing(
            ticker=ticker,
            filing=filing,
        )

        retrieved_chunks = self._retrieve_chunks_for_filing(pipeline=pipeline, filing=filing)

        model_error = None
        raw_answer = ""
        predicted_label = "Neutral"

        if retrieved_chunks:
            generation_provider = OpenAIChatGenerationProvider(
                api_key=self.openai_api_key,
                model=self.generation_model,
            )

            prompt = build_classification_prompt(ticker, filing)
            raw_answer = generation_provider.generate_answer(
                question=prompt,
                retrieved_chunks=retrieved_chunks,
            )
            predicted_label = normalize_label(raw_answer)
        else:
            model_error = "No relevant chunks retrieved."

        actual_result = compute_actual_label_from_yfinance(
            ticker=ticker,
            filing_date=filing.get("filing_date", ""),
            horizon_days=self.horizon_days,
            price_label_threshold=self.price_label_threshold,
        )

        return {
            "ticker": ticker.upper(),
            "quarter": filing.get("quarter", "N/A"),
            "filing_date": filing.get("filing_date", "N/A"),
            "accession_number": filing.get("accession_number", "N/A"),
            "form_type": filing.get("form_type", "10-Q"),
            "predicted_label": predicted_label,
            "predicted_raw_answer": raw_answer,
            "realized_next_quarter_label": actual_result.get("actual_label", "Unknown"),
            "actual_label": actual_result.get("actual_label", "Unknown"),
            "realized_next_quarter_pct_change": actual_result.get("actual_return_percent"),
            "actual_return_percent": actual_result.get("actual_return_percent"),
            "realized_next_quarter_start_price": actual_result.get("actual_price_start"),
            "actual_price_start": actual_result.get("actual_price_start"),
            "realized_next_quarter_end_price": actual_result.get("actual_price_end"),
            "actual_price_end": actual_result.get("actual_price_end"),
            "chunks_used": len(retrieved_chunks),
            "chunk_count": chunk_count,
            "selected_sections": selected_sections,
            "retrieved_chunks": retrieved_chunks,
            "model_error": model_error,
            "price_error": actual_result.get("error"),
        }

    def analyze_ticker(self, ticker: str) -> Dict:
        """
        Main entry point.

        Returns a dictionary with:
          - ticker
          - generated_at
          - filing_count
          - chunk_count
          - quarterly_results
        """
        prepared_filings = self.yahoo_agent.prepare_recent_10q_filings_for_chunking(
            ticker=ticker,
            num_quarters=self.num_quarters,
        )

        if not prepared_filings.get("success"):
            return {
                "success": False,
                "ticker": ticker.upper(),
                "error": prepared_filings.get("error", "Failed to prepare filings."),
                "generated_at": datetime.now().isoformat(),
                "filing_count": 0,
                "chunk_count": 0,
                "quarterly_results": [],
            }

        quarterly_results: List[Dict] = []
        total_chunk_count = 0

        for filing in prepared_filings.get("filings", []):
            result = self.analyze_single_filing(ticker=ticker, filing=filing)
            quarterly_results.append(result)
            total_chunk_count += int(result.get("chunk_count", 0) or 0)

        return {
            "success": True,
            "ticker": ticker.upper(),
            "generated_at": datetime.now().isoformat(),
            "filing_count": len(prepared_filings.get("filings", [])),
            "chunk_count": total_chunk_count,
            "quarterly_results": quarterly_results,
        }


if __name__ == "__main__":
    # Example:
    # analyzer = QuarterlySentimentAnalyzer()
    # result = analyzer.analyze_ticker("AAPL")
    # print(result)
    pass