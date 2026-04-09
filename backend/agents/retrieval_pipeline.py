import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from dotenv import load_dotenv

load_dotenv()

MAX_EMBEDDING_TEXT_CHARS = 6000
EMBEDDING_SPLIT_OVERLAP_CHARS = 400
MAX_EMBEDDING_TOKENS_PER_REQUEST = 250000
MAX_GENERATION_CONTEXT_CHARS = 18000
MAX_GENERATION_CHUNK_CHARS = 3000
OPENAI_RETRY_ATTEMPTS = 4
OPENAI_RETRY_BASE_DELAY_SECONDS = 1.0
DATE_PATTERN = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "could",
    "did",
    "does",
    "for",
    "forward",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "main",
    "most",
    "of",
    "or",
    "quarter",
    "the",
    "these",
    "this",
    "to",
    "upcoming",
    "was",
    "were",
    "what",
    "which",
}
QUESTION_TYPE_SECTION_HINTS = {
    "risk_material": ["risk factors", "legal proceedings", "management's discussion and analysis"],
    "management_actions": ["legal proceedings", "liquidity and capital resources", "management's discussion and analysis"],
    "revenue_driver": ["net sales", "results of operations", "management's discussion and analysis"],
    "forward_revenue_driver": ["net sales", "results of operations", "management's discussion and analysis"],
    "profitability_margin": ["gross margin", "results of operations", "management's discussion and analysis"],
    "forward_pressure": ["risk factors", "gross margin", "market risk", "management's discussion and analysis"],
    "liquidity": ["liquidity and capital resources", "cash flows", "management's discussion and analysis"],
    "cash_flow": ["cash flows", "liquidity and capital resources", "management's discussion and analysis"],
    "fact": ["results of operations", "management's discussion and analysis"],
    "narrative": ["results of operations", "management's discussion and analysis"],
}
QUESTION_TYPE_METRIC_ALIASES = {
    "risk_material": ["risk", "legal", "regulatory", "fine", "proceedings"],
    "management_actions": ["appealed", "compliance", "hedging", "liquidity", "mitigate", "stabilize"],
    "revenue_driver": ["revenue", "net sales", "sales", "driver", "growth", "decline", "segment"],
    "forward_revenue_driver": ["revenue", "net sales", "sales", "driver", "outlook", "demand"],
    "profitability_margin": ["profitability", "gross margin", "margin", "operating income", "net income", "product gross margin", "services gross margin"],
    "forward_pressure": ["pressure", "headwind", "cost", "expense", "tariff", "volatility", "downward pressure"],
    "liquidity": ["liquidity", "capital resources", "cash", "cash equivalents", "debt"],
    "cash_flow": ["cash flow", "operating cash flow", "investing", "financing", "liquidity"],
}
QUESTION_TYPE_RECOMMENDED_K = {
    "risk_material": 10,
    "management_actions": 10,
    "revenue_driver": 8,
    "forward_revenue_driver": 8,
    "profitability_margin": 8,
    "forward_pressure": 8,
    "liquidity": 8,
    "cash_flow": 8,
    "fact": 5,
    "narrative": 6,
    "general": 5,
}

QUESTION_TYPE_SOURCE_PREFERENCES = {
    "risk_material": ["prose", "table"],
    "management_actions": ["prose", "table"],
    "revenue_driver": ["prose", "table"],
    "forward_revenue_driver": ["prose", "table"],
    "profitability_margin": ["table", "prose"],
    "forward_pressure": ["prose", "table"],
    "liquidity": ["table", "prose"],
    "cash_flow": ["table", "prose"],
    "fact": ["table", "prose"],
    "narrative": ["prose", "table"],
    "general": ["prose", "table"],
}


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


class EmbeddingCache(Protocol):
    def get(self, model: str, text: str, kind: str) -> Optional[List[float]]:
        ...

    def set(self, model: str, text: str, kind: str, embedding: List[float]) -> None:
        ...


class JsonEmbeddingCache:
    """Persistent embedding cache keyed by model, text kind, and text hash."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: Dict[str, List[float]] = {}
        self._loaded = False

    def get(self, model: str, text: str, kind: str) -> Optional[List[float]]:
        self._ensure_loaded()
        embedding = self._entries.get(self._key(model, text, kind))
        return list(embedding) if embedding is not None else None

    def set(self, model: str, text: str, kind: str, embedding: List[float]) -> None:
        self._ensure_loaded()
        self._entries[self._key(model, text, kind)] = list(embedding)
        self._persist()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = {
                key: [float(value) for value in values]
                for key, values in payload.get("entries", {}).items()
            }
        self._loaded = True

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "entries": self._entries,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _key(model: str, text: str, kind: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{kind}:{model}:{digest}"


class OpenAIEmbeddingProvider:
    """OpenAI embeddings provider for filing chunks and queries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        cache: Optional[EmbeddingCache] = None,
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
        self.cache = cache

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings: List[Optional[List[float]]] = [None] * len(texts)
        missing_indices: List[int] = []

        if self.cache:
            for index, text in enumerate(texts):
                cached = self.cache.get(self.model, text, kind="chunk")
                if cached is not None:
                    embeddings[index] = cached
                else:
                    missing_indices.append(index)
        else:
            missing_indices = list(range(len(texts)))

        missing_texts = [texts[index] for index in missing_indices]
        filled_embeddings: List[List[float]] = []
        for batch in _batch_texts_for_embedding(missing_texts):
            response = _run_with_retry(
                lambda: self.client.embeddings.create(model=self.model, input=batch)
            )
            filled_embeddings.extend(item.embedding for item in response.data)

        for index, embedding in zip(missing_indices, filled_embeddings):
            embeddings[index] = embedding
            if self.cache:
                self.cache.set(self.model, texts[index], kind="chunk", embedding=embedding)

        return [embedding for embedding in embeddings if embedding is not None]

    def embed_query(self, text: str) -> List[float]:
        if self.cache:
            cached = self.cache.get(self.model, text, kind="query")
            if cached is not None:
                return cached
        response = _run_with_retry(
            lambda: self.client.embeddings.create(model=self.model, input=[text])
        )
        embedding = response.data[0].embedding
        if self.cache:
            self.cache.set(self.model, text, kind="query", embedding=embedding)
        return embedding


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
    section_name: Optional[str] = None
    period_end: Optional[str] = None

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
            "section_name": self.section_name,
            "period_end": self.period_end,
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

        query_plan = build_query_plan(query)
        subqueries = query_plan.get("subqueries") or [query]
        query_vectors = [self.embedding_provider.embed_query(subquery) for subquery in subqueries]
        bm25_scores_by_subquery = [
            self.bm25_index.score(_tokenize_for_bm25(subquery))
            if self.bm25_index else [0.0] * len(self.chunk_records)
            for subquery in subqueries
        ]

        candidates = []
        for record_index, (record, vector) in enumerate(zip(self.chunk_records, self.chunk_vectors)):
            dense_scores = [
                _cosine_similarity(query_vector, vector)
                for query_vector in query_vectors
            ]
            sparse_scores = [
                subquery_scores[record_index]
                for subquery_scores in bm25_scores_by_subquery
            ]
            candidates.append(
                {
                    "record_index": record_index,
                    "record": record,
                    "vector": vector,
                    "dense_scores": dense_scores,
                    "sparse_scores": sparse_scores,
                    "dense_score": max(dense_scores) if dense_scores else 0.0,
                    "sparse_score": max(sparse_scores) if sparse_scores else 0.0,
                    "metadata_score": _metadata_match_score(record, query_plan),
                    "section_score": _section_match_score(record, query_plan),
                    "source_type_bonus": _source_type_preference_bonus(record, query_plan),
                    "metric_score": _metric_alias_match_score(record, query_plan),
                    "numeric_score": _numeric_evidence_score(record, query_plan),
                    "coverage_score": _query_coverage_score(dense_scores, sparse_scores),
                }
            )

        ranked_lists = _build_ranked_candidate_lists(
            candidates,
            subqueries=subqueries,
            k=max(k, self.fused_candidates),
            dense_candidates=self.dense_candidates,
            sparse_candidates=self.sparse_candidates,
            preferred_source_types=query_plan.get("preferred_source_types") or ["prose", "table"],
        )
        fused = _reciprocal_rank_fuse_lists(ranked_lists)
        fused_candidates = sorted(
            fused.values(),
            key=lambda item: (
                item["hybrid_score"],
                item.get("coverage_score", 0),
                item.get("metric_score", 0),
                item.get("numeric_score", 0),
                item["section_score"],
                item["metadata_score"],
                item.get("source_type_bonus", 0),
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
                    item.get("coverage_score", 0),
                    item.get("metric_score", 0),
                    item.get("numeric_score", 0),
                    item["section_score"],
                    item["metadata_score"],
                    item.get("source_type_bonus", 0),
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
                    "section_score": item["section_score"],
                    "metric_score": item.get("metric_score"),
                    "numeric_score": item.get("numeric_score"),
                    "coverage_score": item.get("coverage_score"),
                    "rerank_score": item.get("rerank_score"),
                }
            )

        return {
            "query": query,
            "query_plan": query_plan,
            "results": scored_records[:k],
            "success": True,
        }

    def answer_question(
        self,
        question: str,
        generation_provider: GenerationProvider,
        k: Optional[int] = None,
    ) -> Dict:
        effective_k = k if k is not None else build_query_plan(question).get("recommended_k", 5)
        retrieval_result = self.search(question, k=effective_k)
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
                        period_end=chunk_info.get("period_end"),
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
                        section_name=None,
                        period_end=None,
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
                        section_name=chunk_info.get("section_name"),
                        period_end=chunk_info.get("period_end"),
                        **filing_common,
                    )
                )
        else:
            for index, chunk_text in enumerate(prepared_chunk_data.get("table_chunks", [])):
                chunk_records.append(
                    ChunkRecord(
                        chunk_id=f"{filing_common['accession_number']}:table:{index}",
                        text=chunk_text,
                        source_type="table",
                        section_name=None,
                        period_end=None,
                        **filing_common,
                    )
                )

    return chunk_records


def build_generation_context(retrieved_chunks: List[Dict]) -> str:
    intro = (
        "Retrieved 10-Q excerpts are intentionally ordered chronologically from oldest filing to newest filing."
    )
    context_blocks = [intro]
    total_chars = len(intro)
    omitted_chunks = 0

    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = (
            f"Ticker: {_sanitize_generation_value(chunk.get('ticker', 'N/A'))} | "
            f"Quarter: {_sanitize_generation_value(chunk.get('quarter', 'N/A'))} | "
            f"Filed: {_sanitize_generation_value(chunk.get('filing_date', 'N/A'))} | "
            f"Source: {_sanitize_generation_value(chunk.get('source_type', 'N/A'))} | "
            f"Section: {_sanitize_generation_value(chunk.get('section_name') or 'N/A')} | "
            f"Period End: {_sanitize_generation_value(chunk.get('period_end') or 'N/A')}"
        )
        text = _sanitize_generation_text(chunk.get("text", ""))
        text = _truncate_generation_text(
            text,
            max_chars=MAX_GENERATION_CHUNK_CHARS,
            suffix="\n[Excerpt truncated for prompt size]",
        )

        block = f"Chunk {index}\n{metadata}\n{text}"
        block_with_separator = (
            f"\n\n---\n\n{block}" if len(context_blocks) > 1 else f"\n\n{block}"
        )

        if total_chars + len(block_with_separator) > MAX_GENERATION_CONTEXT_CHARS:
            omitted_chunks = len(retrieved_chunks) - index + 1
            break

        context_blocks.append(block)
        total_chars += len(block_with_separator)

    if omitted_chunks:
        context_blocks.append(
            f"[{omitted_chunks} additional retrieved chunks omitted for prompt size]"
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

    return {
        "filing_date": filing_date_match.group(0) if filing_date_match else None,
        "quarter": quarter_match.group(1).upper() if quarter_match else None,
        "period_end": _normalize_date_string(period_end_match.group(0)) if period_end_match else None,
        "question_type": _infer_query_style(question),
    }


def build_query_plan(question: str) -> Dict[str, object]:
    metadata = parse_question_metadata(question)
    question_type = metadata.get("question_type") or "general"
    metric_aliases = _build_metric_aliases(question, question_type)
    preferred_sections = _build_preferred_sections(question, question_type)
    preferred_source_types = QUESTION_TYPE_SOURCE_PREFERENCES.get(question_type, ["prose", "table"])
    subqueries = _build_subqueries(question, metric_aliases, preferred_sections, question_type)

    return {
        **metadata,
        "question": question,
        "metric_aliases": metric_aliases,
        "preferred_sections": preferred_sections,
        "preferred_source_types": preferred_source_types,
        "subqueries": subqueries,
        "recommended_k": QUESTION_TYPE_RECOMMENDED_K.get(question_type, 5),
        "needs_numeric_support": question_type in {"fact", "profitability_margin", "cash_flow", "liquidity"},
        "needs_explanatory_support": question_type in {
            "narrative",
            "revenue_driver",
            "forward_revenue_driver",
            "forward_pressure",
            "management_actions",
            "risk_material",
            "profitability_margin",
        },
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

    return score


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


def _sanitize_generation_value(value: object) -> str:
    if value is None:
        return "N/A"
    return _sanitize_generation_text(str(value)) or "N/A"


def _sanitize_generation_text(text: object) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    sanitized_chars: List[str] = []
    for char in text:
        codepoint = ord(char)
        if codepoint == 0:
            continue
        if 0xD800 <= codepoint <= 0xDFFF:
            continue
        if codepoint < 32 and char not in "\n\r\t":
            sanitized_chars.append(" ")
            continue
        sanitized_chars.append(char)

    sanitized = "".join(sanitized_chars)
    sanitized = sanitized.replace("\r\n", "\n").replace("\r", "\n")
    return sanitized.strip()


def _truncate_generation_text(text: str, max_chars: int, suffix: str) -> str:
    if len(text) <= max_chars:
        return text

    available_chars = max(0, max_chars - len(suffix))
    truncated = text[:available_chars].rstrip()

    for separator in ("\n\n", "\n", ". ", " "):
        split_index = truncated.rfind(separator)
        if split_index >= max_chars // 2:
            truncated = truncated[:split_index].rstrip()
            break

    return f"{truncated}{suffix}"


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
    if any(phrase in lowered for phrase in ["most material risks", "risk explicitly disclosed"]):
        return "risk_material"
    if any(phrase in lowered for phrase in ["actions has management taken", "stabilize performance", "address these risks"]):
        return "management_actions"
    if any(phrase in lowered for phrase in ["profitability", "margin change", "margin changes", "gross margin", "operating margin"]):
        return "profitability_margin"
    if any(phrase in lowered for phrase in ["operational or financial pressures", "pressures could affect", "headwinds", "downward pressure"]):
        return "forward_pressure"
    if any(phrase in lowered for phrase in ["expected main drivers of revenue growth", "moving forward", "revenue growth moving forward"]):
        return "forward_revenue_driver"
    if any(phrase in lowered for phrase in ["drivers of revenue growth", "drivers of revenue", "drove revenue growth", "revenue growth or decline"]):
        return "revenue_driver"
    if any(phrase in lowered for phrase in ["liquidity", "capital resources"]):
        return "liquidity"
    if any(phrase in lowered for phrase in ["cash flow", "cash flows"]):
        return "cash_flow"
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


def _build_metric_aliases(question: str, question_type: str) -> List[str]:
    aliases = list(QUESTION_TYPE_METRIC_ALIASES.get(question_type, []))
    lowered = question.lower()

    if "revenue" in lowered or "sales" in lowered:
        aliases.extend(["revenue", "net sales", "sales"])
    if "margin" in lowered:
        aliases.extend(["margin", "gross margin", "operating margin"])
    if "profit" in lowered:
        aliases.extend(["profitability", "operating income", "net income"])
    if "risk" in lowered:
        aliases.extend(["risk", "regulatory", "legal"])
    if "cash" in lowered:
        aliases.extend(["cash", "cash flow", "cash flows"])

    return _unique_preserving_order(aliases)


def _build_preferred_sections(question: str, question_type: str) -> List[str]:
    sections = list(QUESTION_TYPE_SECTION_HINTS.get(question_type, []))
    lowered = question.lower()
    if "margin" in lowered:
        sections.append("gross margin")
    if "revenue" in lowered or "sales" in lowered:
        sections.extend(["net sales", "results of operations"])
    if "risk" in lowered:
        sections.extend(["risk factors", "legal proceedings"])
    return _unique_preserving_order(sections)


def _build_subqueries(
    question: str,
    metric_aliases: List[str],
    preferred_sections: List[str],
    question_type: str,
) -> List[str]:
    clauses = _split_question_into_clauses(question)
    focus_terms = [token for token in _tokenize_for_bm25(question) if token not in QUERY_STOPWORDS]
    subqueries = [question.strip()]
    subqueries.extend(clause for clause in clauses if clause and clause != question.strip())

    if metric_aliases:
        subqueries.append(" ".join(metric_aliases[:5]))
    if preferred_sections and metric_aliases:
        subqueries.append(" ".join(metric_aliases[:3] + preferred_sections[:2]))
    elif preferred_sections:
        subqueries.append(" ".join(preferred_sections[:2]))

    if question_type == "profitability_margin":
        subqueries.extend(
            [
                "gross margin trend",
                "margin drivers product mix costs",
            ]
        )
    elif question_type in {"revenue_driver", "forward_revenue_driver"}:
        subqueries.extend(
            [
                "revenue growth drivers",
                "net sales by segment product mix",
            ]
        )
    elif question_type == "forward_pressure":
        subqueries.extend(
            [
                "operating pressures costs margins",
                "forward looking risks expenses tariffs",
            ]
        )
    elif question_type == "management_actions":
        subqueries.extend(
            [
                "management actions appealed plan hedging liquidity",
                "steps taken to address risks",
            ]
        )

    if focus_terms:
        subqueries.append(" ".join(focus_terms[:8]))

    return _unique_preserving_order(query for query in subqueries if query)


def _split_question_into_clauses(question: str) -> List[str]:
    stripped = question.strip().rstrip("?")
    clauses = [
        part.strip()
        for part in re.split(r"\b(?:and|while|versus|vs\.?)\b|[,;]", stripped, flags=re.IGNORECASE)
        if part.strip()
    ]
    return clauses


def _source_type_preference_bonus(record: ChunkRecord, query_plan: Dict[str, object]) -> int:
    preferred_source_types = query_plan.get("preferred_source_types") or []
    if record.source_type in preferred_source_types:
        preferred_index = preferred_source_types.index(record.source_type)
        return max(0, 3 - preferred_index)
    if query_plan.get("question_type") == "narrative":
        return 2 if record.source_type == "prose" else 0
    return 0


def _section_match_score(record: ChunkRecord, query_metadata: Dict[str, object]) -> int:
    section = (record.section_name or "").lower()
    if not section:
        return 0

    question_type = query_metadata.get("question_type")
    if question_type == "risk_material":
        if "risk factors" in section:
            return 2
        if "legal proceedings" in section:
            return 1
    if question_type == "management_actions":
        if "legal proceedings" in section or "liquidity and capital resources" in section:
            return 2
        if "management's discussion and analysis" in section:
            return 1
    preferred_sections = query_metadata.get("preferred_sections") or []
    for preferred_section in preferred_sections:
        if preferred_section.lower() in section:
            return 2
    return 0


def _metric_alias_match_score(record: ChunkRecord, query_plan: Dict[str, object]) -> int:
    aliases = query_plan.get("metric_aliases") or []
    if not aliases:
        return 0

    lowered = record.text.lower()
    matches = sum(1 for alias in aliases if alias.lower() in lowered)
    return min(matches, 4)


def _numeric_evidence_score(record: ChunkRecord, query_plan: Dict[str, object]) -> int:
    if not query_plan.get("needs_numeric_support"):
        return 0

    score = 0
    if record.source_type == "table":
        score += 2
    if re.search(r"\d", record.text):
        score += 1
    if "%" in record.text or "$" in record.text:
        score += 1
    return score


def _query_coverage_score(dense_scores: List[float], sparse_scores: List[float]) -> int:
    coverage = 0
    for dense_score, sparse_score in zip(dense_scores, sparse_scores):
        if dense_score > 0 or sparse_score > 0:
            coverage += 1
    return coverage


def _build_ranked_candidate_lists(
    candidates: List[Dict],
    subqueries: List[str],
    k: int,
    dense_candidates: int,
    sparse_candidates: int,
    preferred_source_types: List[str],
) -> List[Tuple[str, List[Dict]]]:
    ranked_lists: List[Tuple[str, List[Dict]]] = []
    source_types = _ordered_source_types(candidates, preferred_source_types)

    for source_type in source_types:
        source_candidates = [candidate for candidate in candidates if candidate["record"].source_type == source_type]
        if not source_candidates:
            continue

        for subquery_index, _subquery in enumerate(subqueries):
            dense_sorted = sorted(
                source_candidates,
                key=lambda item: (
                    item["dense_scores"][subquery_index],
                    item["metric_score"],
                    item["numeric_score"],
                    item["section_score"],
                    item["metadata_score"],
                    item["source_type_bonus"],
                ),
                reverse=True,
            )[: max(k, dense_candidates)]
            sparse_sorted = sorted(
                source_candidates,
                key=lambda item: (
                    item["sparse_scores"][subquery_index],
                    item["metric_score"],
                    item["numeric_score"],
                    item["section_score"],
                    item["metadata_score"],
                    item["source_type_bonus"],
                ),
                reverse=True,
            )[: max(k, sparse_candidates)]
            ranked_lists.append((f"{source_type}:dense:{subquery_index}", dense_sorted))
            ranked_lists.append((f"{source_type}:sparse:{subquery_index}", sparse_sorted))

    return ranked_lists


def _ordered_source_types(candidates: List[Dict], preferred_source_types: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []

    for source_type in preferred_source_types:
        if source_type not in seen:
            ordered.append(source_type)
            seen.add(source_type)

    for candidate in candidates:
        source_type = candidate["record"].source_type
        if source_type not in seen:
            ordered.append(source_type)
            seen.add(source_type)

    return ordered


def _unique_preserving_order(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


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
    return _reciprocal_rank_fuse_lists(
        [("dense", dense_ranked), ("sparse", sparse_ranked)],
        rrf_k=rrf_k,
    )


def _reciprocal_rank_fuse_lists(
    ranked_lists: List[Tuple[str, List[Dict]]],
    rrf_k: int = 60,
) -> Dict[str, Dict]:
    fused: Dict[str, Dict] = {}
    for label, ranked_list in ranked_lists:
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
