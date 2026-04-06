# quarterly_sentiment.py

import logging
import math
import re
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import yfinance as yf
from sentence_transformers import CrossEncoder
from transformers import pipeline

from .retrieval_pipeline import (
    FilingRetrievalPipeline,
    build_chunk_records_from_prepared_filings,
    order_retrieved_chunks_for_generation,
)
from .yahoo_finance_agent import YahooFinanceAgent

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5):
        if not chunks:
            return []

        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)

        for c, s in zip(chunks, scores):
            c["rerank_score"] = float(s)

        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return chunks[:top_k]


class SentimentEmbeddingProvider:
    """
    Local embedding provider using sentence-transformers.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        texts = [
            "Represent this sentence for searching relevant passages: " + t
            for t in texts
        ]
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        text = "Represent this sentence for searching relevant passages: " + text
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


class SentimentLLMGenerationProvider:
    """
    Label provider using zero-shot classification.

    It returns a structured answer:
    Label: Bullish|Neutral|Bearish
    Reason: ...
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.model_name = model_name
        self.classifier = None

        try:
            device = 0 if torch.cuda.is_available() else -1
            self.classifier = pipeline(
                task="zero-shot-classification",
                model=model_name,
                device=device,
            )
            logger.info("Loaded zero-shot classifier: %s", model_name)
        except Exception as exc:
            logger.warning(
                "Failed to load zero-shot classifier %s: %s",
                model_name,
                exc,
            )
            self.classifier = None

    def generate_answer(self, question: str, retrieved_chunks: list) -> str:
        evidence_chunks = self._select_evidence_chunks(retrieved_chunks)
        evidence_text = self._build_evidence_text(evidence_chunks)

        if not evidence_text.strip():
            return "Label: Unknown\nReason: No usable evidence excerpts were available."

        label = self._classify_evidence(evidence_text)
        reason = self._build_reason(evidence_chunks, label)

        return f"Label: {label}\nReason: {reason}"

    def _select_evidence_chunks(self, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Keep the most useful chunks and avoid feeding the classifier only boilerplate.
        Prefer MD&A / operating / financial sections, then fill the rest by score.
        """
        if not retrieved_chunks:
            return []

        ranked = sorted(
            retrieved_chunks,
            key=lambda c: float(c.get("rerank_score", c.get("score", 0.0))),
            reverse=True,
        )

        priority_keywords = [
            "management's discussion and analysis",
            "discussion and analysis",
            "results of operations",
            "financial statements",
            "consolidated statements",
            "gross margin",
            "operating expenses",
            "segment reporting",
            "revenue",
            "cash flow",
            "liquidity",
            "debt",
        ]

        selected: List[Dict[str, Any]] = []
        used_indices = set()

        def section_text(chunk: Dict[str, Any]) -> str:
            return f"{str(chunk.get('section', '')).lower()} {str(chunk.get('text', '')).lower()}"

        for keyword in priority_keywords:
            best_idx = None
            best_score = None
            for i, chunk in enumerate(ranked):
                if i in used_indices:
                    continue
                text = section_text(chunk)
                if keyword in text:
                    score = float(chunk.get("rerank_score", chunk.get("score", 0.0)))
                    if best_score is None or score > best_score:
                        best_score = score
                        best_idx = i
            if best_idx is not None:
                selected.append(ranked[best_idx])
                used_indices.add(best_idx)

        for i, chunk in enumerate(ranked):
            if i in used_indices:
                continue
            selected.append(chunk)
            used_indices.add(i)
            if len(selected) >= min(5, len(ranked)):
                break

        return selected

    def _build_evidence_text(self, chunks: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for i, chunk in enumerate(chunks, start=1):
            section = str(chunk.get("section", "")).strip()
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            text = text[:1200]
            if section:
                parts.append(f"[{i} | {section}]\n{text}")
            else:
                parts.append(f"[{i}]\n{text}")

        evidence_text = "\n\n".join(parts)
        if len(evidence_text) > 12000:
            evidence_text = evidence_text[:12000]
        return evidence_text

    def _classify_evidence(self, evidence_text: str) -> str:
        if self.classifier is None:
            return "Unknown"

        try:
            result = self.classifier(
                evidence_text,
                candidate_labels=["Bullish", "Neutral", "Bearish"],
                hypothesis_template="The filing is {} for the stock next quarter.",
                multi_label=False,
            )
            if isinstance(result, dict) and result.get("labels"):
                top_label = str(result["labels"][0]).strip().capitalize()
                if top_label in {"Bullish", "Neutral", "Bearish"}:
                    return top_label
        except Exception as exc:
            logger.warning("Zero-shot classification failed: %s", exc)

        return "Unknown"

    def _build_reason(self, chunks: List[Dict[str, Any]], label: str) -> str:
        if not chunks:
            return "No usable evidence excerpts were available."

        text = " ".join(str(chunk.get("text", "")) for chunk in chunks)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            return sentences[0][:220]

        return "The classifier used the selected filing excerpts."


class QuarterlySentimentAnalyzer:
    """
    4-quarter 10-Q sentiment analyzer.
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

        self.embedding_provider = embedding_provider or SentimentEmbeddingProvider(
            model_name="BAAI/bge-base-en-v1.5"
        )
        self.sentiment_llm_provider = sentiment_llm_provider or SentimentLLMGenerationProvider()

        self.pipeline = FilingRetrievalPipeline(self.embedding_provider)
        self.reranker = Reranker()

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
            record_accession = self._get_record_accession(record)
            if record_accession != accession_number:
                continue

            score = self._cosine_similarity(query_vector, vector)
            scored_chunks.append(
                {
                    **self._record_to_dict(record),
                    "score": round(score, 6),
                }
            )

        scored_chunks.sort(key=lambda item: item["score"], reverse=True)
        top_chunks = scored_chunks[:20]

        reranked = self.reranker.rerank(query, top_chunks, top_k=k)
        return reranked

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
        if not retrieved_chunks:
            return {
                "label": "Unknown",
                "reason": "No retrievable chunks were available for this filing.",
                "raw_answer": "",
            }

        quarter = filing.get("quarter", "N/A")

        question = (
            f"Analyze the following 10-Q excerpts for {ticker} ({quarter}). "
            f"Determine whether the company's fundamentals suggest the stock will go UP, DOWN, "
            f"or STAY FLAT next quarter. Focus on revenue, margins, cash flow, debt, guidance, "
            f"risks, and cost trends."
        )

        ordered_chunks = order_retrieved_chunks_for_generation(retrieved_chunks)

        try:
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

        if not reason:
            reason = "No reason returned by the model."

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
                match = re.fullmatch(r"(bullish|neutral|bearish)", candidate, re.IGNORECASE)
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
            if not stripped.lower().startswith("reason"):
                continue

            _, _, value = stripped.partition(":")
            candidate = value.strip()

            if not candidate:
                continue

            if candidate.startswith("<") and candidate.endswith(">"):
                continue

            return candidate

        return ""

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

    def _get_record_accession(self, record: Any) -> str:
        if isinstance(record, dict):
            return str(record.get("accession_number", ""))
        return str(getattr(record, "accession_number", ""))

    def _record_to_dict(self, record: Any) -> Dict[str, Any]:
        if isinstance(record, dict):
            return dict(record)
        if hasattr(record, "to_dict"):
            return record.to_dict()
        return {
            "text": getattr(record, "text", ""),
            "section": getattr(record, "section", ""),
            "accession_number": getattr(record, "accession_number", ""),
        }

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