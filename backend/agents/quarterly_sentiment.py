# quarterly_sentiment.py

import logging
import math
import re
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from transformers import pipeline

import pandas as pd
import torch
import yfinance as yf

from .retrieval_pipeline import (
    FilingRetrievalPipeline,
    build_chunk_records_from_prepared_filings,
    build_generation_context,
    order_retrieved_chunks_for_generation,
)
from .financial_data_agent import YahooFinanceAgent

logger = logging.getLogger(__name__)


class SentimentEmbeddingProvider:
    """
    Local embedding provider using sentence-transformers.

    Default model:
    - sentence-transformers/all-MiniLM-L6-v2
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentimentEmbeddingProvider. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> List[float]:
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()


class SentimentLLMGenerationProvider:
    """
    Local generation provider using a Hugging Face seq2seq model.

    Default model:
    - google/flan-t5-small
    """

    def __init__(self, model_name: str = "google/flan-t5-small"):
        # Use GPU if available
        device = 0 if torch.cuda.is_available() else -1

        self.generator = pipeline(
            task="text-generation",
            model=model_name,
            device=device,
        )

    def generate_answer(self, question: str, retrieved_chunks: list) -> str:
        context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
        prompt = (
            f"Using the following SEC 10-Q excerpts:\n{context}\n\n"
            f"Answer the following question and label as Bullish, Neutral, or Bearish:\n{question}"
        )
        output = self.generator(prompt, max_length=256, do_sample=False)
        return output[0]["generated_text"].strip()

class QuarterlySentimentAnalyzer:
    """
    4-quarter 10-Q sentiment analyzer.

    Reuses your existing filing fetching/chunking utilities and keeps the analyzer
    fully free by using local Hugging Face models.
    """

    def __init__(
        self,
        yahoo_agent: YahooFinanceAgent,
        embedding_provider: Optional[Any] = None,
        sentiment_llm_provider: Optional[Any] = None,
        num_quarters: int = 4,
        top_k: int = 5,
        price_label_threshold: float = 0.02,
        prose_chunk_size: int = 600,
        prose_chunk_overlap: int = 100,
        table_window: int = 10,
        table_overlap: int = 2,
    ) -> None:
        if yahoo_agent is None:
            raise ValueError("yahoo_agent must be provided. Do not initialize it inside this module.")

        self.yahoo_agent = yahoo_agent
        self.num_quarters = max(1, num_quarters)
        self.top_k = max(1, top_k)
        self.price_label_threshold = max(0.0, price_label_threshold)

        self.prose_chunk_size = prose_chunk_size
        self.prose_chunk_overlap = prose_chunk_overlap
        self.table_window = table_window
        self.table_overlap = table_overlap

        self.embedding_provider = embedding_provider or SentimentEmbeddingProvider()
        self.sentiment_llm_provider = sentiment_llm_provider or SentimentLLMGenerationProvider()

        self.pipeline = FilingRetrievalPipeline(self.embedding_provider)

    def analyze_ticker(self, ticker: str, num_quarters: Optional[int] = None) -> Dict[str, Any]:
        """
        End-to-end workflow:
        1) fetch latest 10-Q filings
        2) chunk and index them
        3) retrieve relevant chunks per filing
        4) predict Bullish / Neutral / Bearish per filing
        5) compute realized quarter-to-quarter stock movement for comparison
        """
        ticker = ticker.upper()
        quarters_to_fetch = num_quarters if num_quarters is not None else self.num_quarters

        prepared_filings = self.yahoo_agent.prepare_recent_10q_filings_for_chunking(
            ticker=ticker,
            num_quarters=quarters_to_fetch,
            prose_chunk_size=self.prose_chunk_size,
            prose_chunk_overlap=self.prose_chunk_overlap,
            table_window=self.table_window,
            table_overlap=self.table_overlap,
        )

        if not prepared_filings.get("success"):
            return {
                "ticker": ticker,
                "success": False,
                "error": prepared_filings.get("error", "Failed to fetch filings"),
            }

        filings = prepared_filings.get("filings", [])
        if not filings:
            return {
                "ticker": ticker,
                "success": False,
                "error": "No 10-Q filings were returned",
            }

        filings = sorted(filings, key=lambda filing: filing.get("filing_date", ""))

        chunk_records = build_chunk_records_from_prepared_filings(prepared_filings)
        index_result = self.pipeline.index_chunks(chunk_records)

        historical_comparisons = self._build_historical_price_comparisons(ticker, filings)
        comparison_by_accession = {
            item.get("from_accession_number", ""): item
            for item in historical_comparisons
        }

        quarterly_results: List[Dict[str, Any]] = []

        for filing in filings:
            accession_number = filing.get("accession_number", "")
            retrieved_chunks = self._retrieve_relevant_chunks_for_filing(
                accession_number=accession_number,
                filing=filing,
                k=self.top_k,
            )

            llm_result = self._predict_label_for_filing(
                ticker=ticker,
                filing=filing,
                retrieved_chunks=retrieved_chunks,
            )

            comparison = comparison_by_accession.get(accession_number)

            quarterly_results.append(
                {
                    "quarter": filing.get("quarter", "N/A"),
                    "filing_date": filing.get("filing_date", "N/A"),
                    "accession_number": accession_number,
                    "form_type": filing.get("form_type", "10-Q"),
                    "predicted_label": llm_result.get("label", "Unknown"),
                    "predicted_reason": llm_result.get("reason", ""),
                    "predicted_raw_answer": llm_result.get("raw_answer", ""),
                    "retrieved_chunks": retrieved_chunks,
                    "realized_next_quarter_label": comparison.get("label") if comparison else "Pending",
                    "realized_next_quarter_pct_change": comparison.get("pct_change") if comparison else None,
                    "realized_next_quarter_start_price": comparison.get("start_price") if comparison else None,
                    "realized_next_quarter_end_price": comparison.get("end_price") if comparison else None,
                    "realized_next_quarter_start_date": comparison.get("start_date") if comparison else None,
                    "realized_next_quarter_end_date": comparison.get("end_date") if comparison else None,
                }
            )

        return {
            "ticker": ticker,
            "success": True,
            "generated_at": datetime.now().isoformat(),
            "filing_count": len(filings),
            "chunk_count": len(chunk_records),
            "index_result": index_result,
            "price_label_threshold": self.price_label_threshold,
            "quarterly_results": quarterly_results,
            "historical_price_comparisons": historical_comparisons,
            "prepared_for_chunking": True,
        }

    def _retrieve_relevant_chunks_for_filing(
        self,
        accession_number: str,
        filing: Dict[str, Any],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k chunks for one filing only.
        This prevents chunks from different filings from mixing together.
        """
        if not self.pipeline.chunk_records or not self.pipeline.chunk_vectors:
            return []

        query = self._build_retrieval_query(filing)
        query_vector = self.embedding_provider.embed_query(query)

        scored_chunks: List[Dict[str, Any]] = []

        for record, vector in zip(self.pipeline.chunk_records, self.pipeline.chunk_vectors):
            if getattr(record, "accession_number", "") != accession_number:
                continue

            score = self._cosine_similarity(query_vector, vector)
            scored_chunks.append(
                {
                    **record.to_dict(),
                    "score": round(score, 6),
                }
            )

        scored_chunks.sort(key=lambda item: item["score"], reverse=True)
        return scored_chunks[: max(1, k)]

    def _build_retrieval_query(self, filing: Dict[str, Any]) -> str:
        quarter = filing.get("quarter", "this quarter")
        filing_date = filing.get("filing_date", "unknown date")

        return (
            f"Identify the most important 10-Q disclosures for assessing whether the "
            f"stock may move in the next quarter. Focus on revenue trends, margins, "
            f"cash flow, liquidity, debt, guidance, operating risks, demand, costs, "
            f"and any material changes in {quarter} filed on {filing_date}."
        )

    def _predict_label_for_filing(
        self,
        ticker: str,
        filing: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Use the sentiment LLM provider to predict Bullish / Neutral / Bearish.
        """
        if not retrieved_chunks:
            return {
                "label": "Unknown",
                "reason": "No retrievable chunks were available for this filing.",
                "raw_answer": "",
            }

        threshold_pct = self.price_label_threshold * 100.0
        quarter = filing.get("quarter", "N/A")
        filing_date = filing.get("filing_date", "N/A")

        question = (
            f"You are analyzing a 10-Q filing for ticker {ticker}.\n"
            f"Filing quarter: {quarter}\n"
            f"Filing date: {filing_date}\n\n"
            f"Task: predict the stock label for the NEXT quarter based only on the "
            f"retrieved 10-Q context.\n\n"
            f"Label definitions:\n"
            f"- Bullish: expected next-quarter return is at least +{threshold_pct:.1f}%\n"
            f"- Neutral: expected next-quarter return is between -{threshold_pct:.1f}% "
            f"and +{threshold_pct:.1f}%\n"
            f"- Bearish: expected next-quarter return is at most -{threshold_pct:.1f}%\n\n"
            f"Return exactly two lines:\n"
            f"Label: <Bullish|Neutral|Bearish>\n"
            f"Reason: <one short sentence>"
        )

        try:
            ordered_chunks = order_retrieved_chunks_for_generation(retrieved_chunks)
            raw_answer = self.sentiment_llm_provider.generate_answer(
                question=question,
                retrieved_chunks=ordered_chunks,
            )
        except Exception as exc:
            return {
                "label": "Unknown",
                "reason": f"LLM generation failed: {exc}",
                "raw_answer": "",
            }

        if not isinstance(raw_answer, str):
            raw_answer = str(raw_answer)

        label = self._extract_label_from_answer(raw_answer)
        reason = self._extract_reason_from_answer(raw_answer)

        return {
            "label": label,
            "reason": reason,
            "raw_answer": raw_answer,
        }

    def _build_historical_price_comparisons(
        self,
        ticker: str,
        filings: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Compute realized quarter-to-quarter stock movement labels using adjusted close.
        We compare each filing date with the next filing date in the sorted list.
        """
        comparisons: List[Dict[str, Any]] = []

        filing_dates = [
            self._parse_date(filing.get("filing_date", ""))
            for filing in filings
            if filing.get("filing_date")
        ]
        filing_dates = [dt for dt in filing_dates if dt is not None]

        if len(filing_dates) < 2:
            return comparisons

        start_date = min(filing_dates) - timedelta(days=30)
        end_date = max(filing_dates) + timedelta(days=120)

        price_history = self._download_adjusted_close_history(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        if price_history.empty:
            return comparisons

        for current_filing, next_filing in zip(filings[:-1], filings[1:]):
            current_date = self._parse_date(current_filing.get("filing_date", ""))
            next_date = self._parse_date(next_filing.get("filing_date", ""))

            if current_date is None or next_date is None:
                continue

            start_price = self._price_on_or_before(price_history, current_date)
            end_price = self._price_on_or_before(price_history, next_date)

            pct_change = None
            label = "Unknown"

            if start_price is not None and end_price is not None and start_price != 0:
                pct_change = ((end_price - start_price) / start_price) * 100.0
                label = self._label_from_pct_change(pct_change)

            comparisons.append(
                {
                    "from_accession_number": current_filing.get("accession_number", ""),
                    "from_quarter": current_filing.get("quarter", ""),
                    "to_accession_number": next_filing.get("accession_number", ""),
                    "to_quarter": next_filing.get("quarter", ""),
                    "start_date": current_filing.get("filing_date", ""),
                    "end_date": next_filing.get("filing_date", ""),
                    "start_price": start_price,
                    "end_price": end_price,
                    "pct_change": pct_change,
                    "label": label,
                }
            )

        return comparisons

    def _download_adjusted_close_history(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Download daily price history and keep only adjusted close / close.
        """
        try:
            stock = yf.Ticker(ticker)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                history = stock.history(
                    start=start_date,
                    end=end_date + timedelta(days=1),
                    auto_adjust=False,
                    actions=False,
                )

            if history.empty:
                return pd.DataFrame()

            history = history.copy()

            if "Adj Close" in history.columns:
                price_column = "Adj Close"
            elif "Close" in history.columns:
                price_column = "Close"
            else:
                return pd.DataFrame()

            history = history[[price_column]].rename(columns={price_column: "price"})
            history.index = pd.to_datetime(history.index)

            if getattr(history.index, "tz", None) is not None:
                history.index = history.index.tz_convert(None)

            history.sort_index(inplace=True)
            return history

        except Exception as exc:
            logger.warning("Failed to download price history for %s: %s", ticker, exc)
            return pd.DataFrame()

    def _price_on_or_before(
        self,
        price_history: pd.DataFrame,
        target_date: datetime,
    ) -> Optional[float]:
        """
        Return the last available price on or before target_date.
        """
        if price_history.empty:
            return None

        target_ts = pd.Timestamp(target_date)
        if getattr(target_ts, "tzinfo", None) is not None:
            target_ts = target_ts.tz_localize(None)

        available = price_history.loc[:target_ts]
        if available.empty:
            return None

        try:
            return float(available["price"].iloc[-1])
        except Exception:
            return None

    def _extract_label_from_answer(self, raw_answer: str) -> str:
        """
        Extract Bullish / Neutral / Bearish from the model output.
        """
        if not raw_answer:
            return "Unknown"

        for line in raw_answer.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("label"):
                _, _, value = stripped.partition(":")
                candidate = value.strip()
                match = re.search(r"\b(bullish|neutral|bearish)\b", candidate, re.IGNORECASE)
                if match:
                    return match.group(1).capitalize()

        match = re.search(r"\b(bullish|neutral|bearish)\b", raw_answer, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()

        return "Unknown"

    def _extract_reason_from_answer(self, raw_answer: str) -> str:
        """
        Extract the reason line if the model followed the requested format.
        """
        if not raw_answer:
            return ""

        for line in raw_answer.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("reason"):
                _, _, value = stripped.partition(":")
                return value.strip()

        return raw_answer.strip()

    def _label_from_pct_change(self, pct_change: float) -> str:
        """
        Convert realized stock movement into Bullish / Neutral / Bearish.
        """
        threshold = self.price_label_threshold * 100.0

        if pct_change >= threshold:
            return "Bullish"
        if pct_change <= -threshold:
            return "Bearish"
        return "Neutral"

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            return None

    def _cosine_similarity(
        self,
        left: List[float],
        right: List[float],
    ) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0

        dot_product = sum(l * r for l, r in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))

        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0

        return dot_product / (left_norm * right_norm)