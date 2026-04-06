import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from dotenv import load_dotenv

load_dotenv()

MAX_EMBEDDING_TEXT_CHARS = 6000
EMBEDDING_SPLIT_OVERLAP_CHARS = 400
MAX_EMBEDDING_TOKENS_PER_REQUEST = 250000
OPENAI_RETRY_ATTEMPTS = 4
OPENAI_RETRY_BASE_DELAY_SECONDS = 1.0
DATE_PATTERN = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


class GenerationProvider(Protocol):
    def generate_answer(self, question: str, retrieved_chunks: List[Dict]) -> str:
        ...


class Reranker(Protocol):
    def score_pairs(self, question: str, chunk_texts: List[str]) -> List[float]:
        ...


class OpenAIEmbeddingProvider:
    """OpenAI embeddings provider for filing chunks and queries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required to use OpenAIEmbeddingProvider. "
                "Install it with `pip install openai`."
            ) from exc

        self.client = OpenAI(api_key=resolve_openai_api_key(api_key))
        self.model = model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings: List[List[float]] = []
        for batch in _batch_texts_for_embedding(texts):
            response = _run_with_retry(
                lambda: self.client.embeddings.create(model=self.model, input=batch)
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = _run_with_retry(
            lambda: self.client.embeddings.create(model=self.model, input=[text])
        )
        return response.data[0].embedding


class OpenAIChatGenerationProvider:
    """OpenAI chat-completions provider for RAG answer synthesis."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required to use OpenAIChatGenerationProvider. "
                "Install it with `pip install openai`."
            ) from exc

        self.client = OpenAI(api_key=resolve_openai_api_key(api_key))
        self.model = model
        self.system_prompt = system_prompt or (
            "You are a financial analyst assistant answering questions about SEC 10-Q filings. "
            "Answer only from the provided retrieved context. If the context is insufficient, "
            "say so clearly. Cite specific figures and details when available."
        )

    def generate_answer(self, question: str, retrieved_chunks: List[Dict]) -> str:
        context = build_generation_context(retrieved_chunks)
        user_prompt = (
            f"Retrieved 10-Q context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the retrieved context."
        )
        response = _run_with_retry(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
        )
        return response.choices[0].message.content.strip()


class CrossEncoderReranker:
    """Optional cross-encoder reranker backed by sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required to use CrossEncoderReranker. "
                "Install it with `pip install sentence-transformers`."
            ) from exc

        self.model = CrossEncoder(model_name)

    def score_pairs(self, question: str, chunk_texts: List[str]) -> List[float]:
        if not chunk_texts:
            return []
        pairs = [[question, chunk_text] for chunk_text in chunk_texts]
        return [float(score) for score in self.model.predict(pairs)]


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    ticker: str
    filing_date: str
    accession_number: str
    quarter: str
    form_type: str
    source_type: str
    table_title: Optional[str] = None
    section_name: Optional[str] = None
    statement_type: Optional[str] = None
    metric_name: Optional[str] = None
    period_type: Optional[str] = None
    period_end: Optional[str] = None
    segment_name: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "ticker": self.ticker,
            "filing_date": self.filing_date,
            "accession_number": self.accession_number,
            "quarter": self.quarter,
            "form_type": self.form_type,
            "source_type": self.source_type,
            "table_title": self.table_title,
            "section_name": self.section_name,
            "statement_type": self.statement_type,
            "metric_name": self.metric_name,
            "period_type": self.period_type,
            "period_end": self.period_end,
            "segment_name": self.segment_name,
        }


class FilingRetrievalPipeline:
    """Hybrid in-memory retrieval over prepared filing chunks."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        reranker: Optional[Reranker] = None,
        dense_candidates: int = 40,
        sparse_candidates: int = 40,
        fused_candidates: int = 20,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.dense_candidates = dense_candidates
        self.sparse_candidates = sparse_candidates
        self.fused_candidates = fused_candidates
        self.chunk_records: List[ChunkRecord] = []
        self.chunk_vectors: List[List[float]] = []
        self.bm25_index: Optional[SimpleBM25Index] = None

    def index_chunks(self, chunk_records: List[ChunkRecord]) -> Dict:
        self.chunk_records = expand_oversized_chunk_records(chunk_records)
        self.chunk_vectors = self.embedding_provider.embed_texts(
            [record.text for record in self.chunk_records]
        )
        self.bm25_index = SimpleBM25Index([record.text for record in self.chunk_records])
        return {
            "indexed_chunks": len(self.chunk_records),
            "embedding_dimensions": len(self.chunk_vectors[0]) if self.chunk_vectors else 0,
            "success": True,
        }

    def search(self, query: str, k: int = 5) -> Dict:
        if not self.chunk_records or not self.chunk_vectors:
            return {
                "query": query,
                "results": [],
                "success": False,
                "error": "No indexed chunks available",
            }

        query_vector = self.embedding_provider.embed_query(query)
        query_metadata = parse_question_metadata(query)
        query_tokens = _tokenize_for_bm25(query)
        bm25_scores = self.bm25_index.score(query_tokens) if self.bm25_index else [0.0] * len(self.chunk_records)

        candidates = []
        for record_index, (record, vector) in enumerate(zip(self.chunk_records, self.chunk_vectors)):
            candidates.append(
                {
                    "record_index": record_index,
                    "record": record,
                    "vector": vector,
                    "dense_score": _cosine_similarity(query_vector, vector),
                    "sparse_score": bm25_scores[record_index] if self.bm25_index else 0.0,
                    "metadata_score": _metadata_match_score(record, query_metadata),
                }
            )

        dense_sorted = sorted(
            candidates,
            key=lambda item: (item["dense_score"], item["metadata_score"]),
            reverse=True,
        )[: max(k, self.dense_candidates)]
        sparse_sorted = sorted(
            candidates,
            key=lambda item: (item["sparse_score"], item["metadata_score"]),
            reverse=True,
        )[: max(k, self.sparse_candidates)]
        fused = _reciprocal_rank_fuse(dense_sorted, sparse_sorted)
        fused_candidates = sorted(
            fused.values(),
            key=lambda item: (
                item["hybrid_score"],
                item["metadata_score"],
                _table_preference_bonus(item["record"], query_metadata),
            ),
            reverse=True,
        )[: max(k, self.fused_candidates)]

        if self.reranker and fused_candidates:
            rerank_scores = self.reranker.score_pairs(
                query,
                [candidate["record"].text for candidate in fused_candidates],
            )
            for candidate, rerank_score in zip(fused_candidates, rerank_scores):
                candidate["rerank_score"] = rerank_score
            fused_candidates.sort(
                key=lambda item: (
                    item.get("rerank_score", float("-inf")),
                    item["hybrid_score"],
                    item["metadata_score"],
                ),
                reverse=True,
            )

        scored_records = []
        for item in fused_candidates[:k]:
            record = item["record"]
            scored_records.append(
                {
                    **record.to_dict(),
                    "score": item.get("rerank_score", item["hybrid_score"]),
                    "dense_score": item["dense_score"],
                    "sparse_score": item["sparse_score"],
                    "hybrid_score": item["hybrid_score"],
                    "metadata_score": item["metadata_score"],
                    "rerank_score": item.get("rerank_score"),
                }
            )

        return {
            "query": query,
            "results": scored_records[:k],
            "success": True,
        }

    def answer_question(
        self,
        question: str,
        generation_provider: GenerationProvider,
        k: int = 5,
    ) -> Dict:
        retrieval_result = self.search(question, k=k)
        if not retrieval_result.get("success"):
            return retrieval_result

        ordered_results = order_retrieved_chunks_for_generation(
            retrieval_result["results"]
        )
        answer = generation_provider.generate_answer(
            question,
            ordered_results,
        )
        return {
            "question": question,
            "answer": answer,
            "sources": ordered_results,
            "success": True,
        }


def build_chunk_records_from_prepared_filings(prepared_filings_data: Dict) -> List[ChunkRecord]:
    chunk_records: List[ChunkRecord] = []
    ticker = prepared_filings_data.get("ticker", "UNKNOWN")

    for filing in prepared_filings_data.get("filings", []):
        filing_common = {
            "ticker": ticker,
            "filing_date": filing.get("filing_date", ""),
            "accession_number": filing.get("accession_number", ""),
            "quarter": filing.get("quarter", ""),
            "form_type": filing.get("form_type", "10-Q"),
        }
        prepared_chunk_data = filing.get("prepared_chunk_data", {})

        prose_chunk_records = prepared_chunk_data.get("prose_chunk_records")
        if prose_chunk_records:
            for index, chunk_info in enumerate(prose_chunk_records):
                chunk_records.append(
                    ChunkRecord(
                        chunk_id=f"{filing_common['accession_number']}:prose:{index}",
                        text=chunk_info.get("text", ""),
                        source_type="prose",
                        section_name=chunk_info.get("section_name"),
                        statement_type=chunk_info.get("statement_type"),
                        metric_name=chunk_info.get("metric_name"),
                        period_type=chunk_info.get("period_type"),
                        period_end=chunk_info.get("period_end"),
                        segment_name=chunk_info.get("segment_name"),
                        **filing_common,
                    )
                )
        else:
            for index, chunk_text in enumerate(prepared_chunk_data.get("prose_chunks", [])):
                chunk_records.append(
                    ChunkRecord(
                        chunk_id=f"{filing_common['accession_number']}:prose:{index}",
                        text=chunk_text,
                        source_type="prose",
                        metric_name=_infer_chunk_metric_name(chunk_text),
                        statement_type=_infer_chunk_statement_type(chunk_text),
                        segment_name=_infer_chunk_segment_name(chunk_text),
                        **filing_common,
                    )
                )

        table_chunk_records = prepared_chunk_data.get("table_chunk_records")
        if table_chunk_records:
            for index, chunk_info in enumerate(table_chunk_records):
                chunk_records.append(
                    ChunkRecord(
                        chunk_id=f"{filing_common['accession_number']}:table:{index}",
                        text=chunk_info.get("text", ""),
                        source_type="table",
                        table_title=chunk_info.get("table_title"),
                        section_name=chunk_info.get("section_name"),
                        statement_type=chunk_info.get("statement_type"),
                        metric_name=chunk_info.get("metric_name"),
                        period_type=chunk_info.get("period_type"),
                        period_end=chunk_info.get("period_end"),
                        segment_name=chunk_info.get("segment_name"),
                        **filing_common,
                    )
                )
        else:
            for index, chunk_text in enumerate(prepared_chunk_data.get("table_chunks", [])):
                table_title = None
                lines = chunk_text.splitlines()
                if lines and lines[0].startswith("[") and lines[0].endswith("]"):
                    table_title = lines[0][1:-1]

                chunk_records.append(
                    ChunkRecord(
                        chunk_id=f"{filing_common['accession_number']}:table:{index}",
                        text=chunk_text,
                        source_type="table",
                        table_title=table_title,
                        metric_name=_infer_chunk_metric_name(chunk_text),
                        statement_type=_infer_chunk_statement_type(chunk_text, table_title),
                        segment_name=_infer_chunk_segment_name(chunk_text),
                        **filing_common,
                    )
                )

    return chunk_records


def build_generation_context(retrieved_chunks: List[Dict]) -> str:
    context_blocks = [
        "Retrieved 10-Q excerpts are intentionally ordered chronologically from oldest filing to newest filing."
    ]
    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = (
            f"Ticker: {chunk.get('ticker', 'N/A')} | "
            f"Quarter: {chunk.get('quarter', 'N/A')} | "
            f"Filed: {chunk.get('filing_date', 'N/A')} | "
            f"Source: {chunk.get('source_type', 'N/A')} | "
            f"Table: {chunk.get('table_title') or 'N/A'} | "
            f"Section: {chunk.get('section_name') or 'N/A'} | "
            f"Statement: {chunk.get('statement_type') or 'N/A'} | "
            f"Metric: {chunk.get('metric_name') or 'N/A'} | "
            f"Period End: {chunk.get('period_end') or 'N/A'}"
        )
        context_blocks.append(
            f"Chunk {index}\n{metadata}\n{chunk.get('text', '')}"
        )
    return "\n\n---\n\n".join(context_blocks)


def order_retrieved_chunks_for_generation(retrieved_chunks: List[Dict]) -> List[Dict]:
    return sorted(
        retrieved_chunks,
        key=lambda chunk: (
            _parse_filing_date(chunk.get("filing_date", "")),
            chunk.get("accession_number", ""),
            0 if chunk.get("source_type") == "prose" else 1,
            chunk.get("chunk_id", ""),
        ),
    )


def parse_question_metadata(question: str) -> Dict[str, Optional[str]]:
    lowered = question.lower()
    filing_date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", question)
    quarter_match = re.search(r"\b(20\d{2}\s+q[1-4])\b", lowered)
    period_end_match = DATE_PATTERN.search(question)

    metric_name = _infer_chunk_metric_name(question)
    segment_name = _infer_chunk_segment_name(question)
    statement_type = _infer_chunk_statement_type(question)

    return {
        "filing_date": filing_date_match.group(0) if filing_date_match else None,
        "quarter": quarter_match.group(1).upper() if quarter_match else None,
        "period_end": _normalize_date_string(period_end_match.group(0)) if period_end_match else None,
        "metric_name": metric_name,
        "segment_name": segment_name,
        "statement_type": statement_type,
        "question_type": _infer_query_style(question),
    }


def _metadata_match_score(record: ChunkRecord, query_metadata: Dict[str, Optional[str]]) -> int:
    score = 0
    filing_date = query_metadata.get("filing_date")
    if filing_date:
        if record.filing_date == filing_date:
            score += 4
        else:
            return 0

    quarter = query_metadata.get("quarter")
    if quarter:
        if (record.quarter or "").upper() == quarter:
            score += 3
        else:
            return 0

    period_end = query_metadata.get("period_end")
    if period_end:
        record_period_end = _normalize_date_string(record.period_end)
        if record_period_end:
            if record_period_end == period_end:
                score += 4
            else:
                return 0

    metric_name = query_metadata.get("metric_name")
    if metric_name and _metadata_text_matches(record.metric_name, metric_name):
        score += 3

    segment_name = query_metadata.get("segment_name")
    if segment_name and _metadata_text_matches(record.segment_name, segment_name):
        score += 2

    statement_type = query_metadata.get("statement_type")
    if statement_type and record.statement_type == statement_type:
        score += 1

    return score


def _metadata_text_matches(record_value: Optional[str], query_value: Optional[str]) -> bool:
    if not record_value or not query_value:
        return False
    left = record_value.lower()
    right = query_value.lower()
    return left in right or right in left


def resolve_openai_api_key(api_key: Optional[str] = None) -> str:
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "OPENAI_API_KEY was not provided and was not found in the environment."
        )
    return resolved_api_key


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(l * r for l, r in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def _parse_filing_date(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (TypeError, ValueError):
        return datetime.max


def _normalize_date_string(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = value.strip()
    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def expand_oversized_chunk_records(
    chunk_records: List[ChunkRecord],
    max_chars: int = MAX_EMBEDDING_TEXT_CHARS,
    overlap_chars: int = EMBEDDING_SPLIT_OVERLAP_CHARS,
) -> List[ChunkRecord]:
    expanded: List[ChunkRecord] = []
    for record in chunk_records:
        if len(record.text) <= max_chars:
            expanded.append(record)
            continue

        parts = _split_text_for_embedding(record.text, max_chars, overlap_chars)
        for index, part in enumerate(parts):
            expanded.append(
                ChunkRecord(
                    **{
                        **record.to_dict(),
                        "chunk_id": f"{record.chunk_id}#part{index + 1}",
                        "text": part,
                    }
                )
            )
    return expanded


def _split_text_for_embedding(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    separators = ["\n\n", "\n", ". ", " "]
    parts: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + max_chars, text_length)
        split_at = end

        if end < text_length:
            window = text[start:end]
            for separator in separators:
                split_index = window.rfind(separator)
                if split_index > max_chars // 2:
                    split_at = start + split_index + len(separator)
                    break

        if split_at <= start:
            split_at = end

        part = text[start:split_at].strip()
        if part:
            parts.append(part)

        if split_at >= text_length:
            break

        start = max(split_at - overlap_chars, start + 1)

    return parts


def _batch_texts_for_embedding(
    texts: List[str],
    max_estimated_tokens: int = MAX_EMBEDDING_TOKENS_PER_REQUEST,
) -> List[List[str]]:
    batches: List[List[str]] = []
    current_batch: List[str] = []
    current_tokens = 0

    for text in texts:
        estimated_tokens = _estimate_token_count(text)
        if current_batch and current_tokens + estimated_tokens > max_estimated_tokens:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += estimated_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def _estimate_token_count(text: str) -> int:
    return max(1, len(text) // 3)


def _run_with_retry(operation, attempts: int = OPENAI_RETRY_ATTEMPTS):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if not _is_retryable_openai_error(exc) or attempt == attempts:
                raise
            time.sleep(OPENAI_RETRY_BASE_DELAY_SECONDS * attempt)
    raise last_error


def _is_retryable_openai_error(exc: Exception) -> bool:
    message = str(exc).lower()
    retryable_terms = [
        "connection error",
        "timed out",
        "timeout",
        "temporary failure",
        "brotli",
        "decodingerror",
        "server disconnected",
        "remoteprotocolerror",
    ]
    return any(term in message for term in retryable_terms)


def _infer_query_style(question: str) -> str:
    lowered = question.lower()
    if any(
        phrase in lowered for phrase in [
            "what was",
            "how much",
            "how many",
            "as of",
            "diluted",
            "basic earnings per share",
        ]
    ):
        return "fact"
    if any(phrase in lowered for phrase in ["why", "driver", "main driver", "what did", "expect"]):
        return "narrative"
    return "general"


def _table_preference_bonus(record: ChunkRecord, query_metadata: Dict[str, Optional[str]]) -> int:
    if query_metadata.get("question_type") == "fact":
        return 2 if record.source_type == "table" else 0
    if query_metadata.get("question_type") == "narrative":
        return 2 if record.source_type == "prose" else 0
    return 0


def _infer_chunk_metric_name(text: str) -> Optional[str]:
    metric_patterns = [
        "net cash provided by operating activities",
        "cash and cash equivalents",
        "income from operations",
        "operating income",
        "gross margin",
        "gross profit",
        "net income",
        "revenue",
        "revenues",
        "basic earnings per share",
        "diluted earnings per share",
        "earnings per share",
        "income tax expense",
        "provision for income taxes",
        "research and development",
        "total assets",
        "total liabilities",
    ]
    lowered = text.lower()
    for pattern in metric_patterns:
        if pattern in lowered:
            return pattern
    return None


def _infer_chunk_segment_name(text: str) -> Optional[str]:
    segment_patterns = [
        "data center",
        "gaming",
        "professional visualization",
        "automotive",
        "google cloud",
        "google services",
        "family of apps",
        "reality labs",
        "services",
        "iphone",
        "wearables, home and accessories",
        "energy generation and storage",
    ]
    lowered = text.lower()
    for pattern in segment_patterns:
        if pattern in lowered:
            return pattern
    return None


def _infer_chunk_statement_type(text: str, table_title: Optional[str] = None) -> Optional[str]:
    haystack = " ".join(part for part in [table_title or "", text] if part).lower()
    if any(term in haystack for term in ["operating activities", "investing activities", "financing activities", "cash flow"]):
        return "cash_flow"
    if any(term in haystack for term in ["cash and cash equivalents", "total assets", "accounts receivable", "inventories", "total liabilities"]):
        return "balance_sheet"
    if any(term in haystack for term in ["revenue", "revenues", "net income", "gross profit", "gross margin", "operating income", "income from operations", "earnings per share", "income tax expense", "research and development"]):
        return "income_statement"
    if _infer_chunk_segment_name(haystack):
        return "segment"
    if "risk factor" in haystack:
        return "risk_factors"
    return None


def _tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", text.lower())


class SimpleBM25Index:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents = [_tokenize_for_bm25(document) for document in documents]
        self.doc_freqs: List[Dict[str, int]] = []
        self.term_doc_counts: Dict[str, int] = {}
        self.doc_lengths = [len(document) for document in self.documents]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0

        for document in self.documents:
            frequencies: Dict[str, int] = {}
            for token in document:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.doc_freqs.append(frequencies)
            for token in frequencies:
                self.term_doc_counts[token] = self.term_doc_counts.get(token, 0) + 1

    def score(self, query_tokens: Sequence[str]) -> List[float]:
        scores = [0.0] * len(self.documents)
        total_docs = len(self.documents)
        if total_docs == 0:
            return scores

        for token in query_tokens:
            doc_count = self.term_doc_counts.get(token, 0)
            if doc_count == 0:
                continue
            idf = math.log(1 + (total_docs - doc_count + 0.5) / (doc_count + 0.5))
            for index, frequencies in enumerate(self.doc_freqs):
                frequency = frequencies.get(token, 0)
                if frequency == 0:
                    continue
                doc_length = self.doc_lengths[index] or 1
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.avgdl, 1.0)
                )
                scores[index] += idf * (frequency * (self.k1 + 1)) / denominator
        return scores


def _reciprocal_rank_fuse(
    dense_ranked: List[Dict],
    sparse_ranked: List[Dict],
    rrf_k: int = 60,
) -> Dict[str, Dict]:
    fused: Dict[str, Dict] = {}
    for ranked_list, label in ((dense_ranked, "dense"), (sparse_ranked, "sparse")):
        for rank, item in enumerate(ranked_list, start=1):
            chunk_id = item["record"].chunk_id
            fused_item = fused.setdefault(
                chunk_id,
                {
                    **item,
                    "hybrid_score": 0.0,
                },
            )
            fused_item["hybrid_score"] += 1.0 / (rrf_k + rank)
            fused_item[f"{label}_rank"] = rank
    return fused
