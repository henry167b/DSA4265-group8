"""
Standalone RAG service for SEC 10-Q filings.

Usage
-----
from backend.agents.filing_rag_service import FilingRAGService

service = FilingRAGService(embedding_model="text-embedding-3-small", generation_model="gpt-4o-mini")

# Option A: index from raw HTML + metadata
service.index_from_html(html_string, filing_meta={"ticker": "AAPL", "filing_date": "2026-01-30", ...})

# Option B: index from a prepared-filings payload (output of YahooFinanceAgent.get_recent_10q_filings)
service.index_from_prepared_filings(prepared_filings_payload)

# Retrieve and answer
result = service.answer("What were the main drivers of revenue growth this quarter?")
print(result["answer"])
print(result["sources"])

# Retrieve only (no generation)
hits = service.search("EU DMA fine legal proceedings")
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .filing_chunker import prepare_filing_html_for_chunking
from .retrieval_pipeline import (
    FilingRetrievalPipeline,
    OpenAIChatGenerationProvider,
    OpenAIEmbeddingProvider,
    build_chunk_records_from_prepared_filings,
    ChunkRecord,
)


class FilingRAGService:
    """
    Self-contained RAG service over SEC 10-Q filings.

    Decoupled from any data-fetching agent — accepts either raw filing HTML
    or a prepared-filings payload and exposes search + answer methods.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        generation_model: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        prose_chunk_size: int = 1200,
        prose_chunk_overlap: int = 150,
        table_window: int = 10,
        table_overlap: int = 2,
    ) -> None:
        self._embedding_provider = OpenAIEmbeddingProvider(
            api_key=openai_api_key,
            model=embedding_model,
        )
        self._generation_provider = OpenAIChatGenerationProvider(
            api_key=openai_api_key,
            model=generation_model,
        )
        self._pipeline = FilingRetrievalPipeline(self._embedding_provider)
        self._prose_chunk_size = prose_chunk_size
        self._prose_chunk_overlap = prose_chunk_overlap
        self._table_window = table_window
        self._table_overlap = table_overlap
        self._indexed = False

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_from_html(
        self,
        html: str,
        filing_meta: Optional[Dict] = None,
    ) -> Dict:
        """
        Chunk a single raw 10-Q HTML document and index it for retrieval.

        Parameters
        ----------
        html:
            Raw HTML string of the 10-Q filing.
        filing_meta:
            Optional dict with keys: ticker, filing_date, accession_number,
            quarter, form_type. Defaults to empty strings if not supplied.
        """
        meta = filing_meta or {}
        prepared_chunk_data = prepare_filing_html_for_chunking(
            html,
            prose_chunk_size=self._prose_chunk_size,
            prose_chunk_overlap=self._prose_chunk_overlap,
            table_window=self._table_window,
            table_overlap=self._table_overlap,
        )
        prepared_payload = {
            "ticker": meta.get("ticker", "UNKNOWN"),
            "filings": [
                {
                    "filing_date": meta.get("filing_date", ""),
                    "accession_number": meta.get("accession_number", ""),
                    "quarter": meta.get("quarter", ""),
                    "form_type": meta.get("form_type", "10-Q"),
                    "prepared_chunk_data": prepared_chunk_data,
                }
            ],
        }
        return self._index_payload(prepared_payload)

    def index_from_prepared_filings(self, prepared_filings_payload: Dict) -> Dict:
        """
        Index from a prepared-filings payload.

        Accepts the dict produced by YahooFinanceAgent.get_recent_10q_filings
        (with include_document_html=True) after chunking, or any dict matching
        the schema: {"ticker": str, "filings": [{"prepared_chunk_data": {...}, ...}]}.
        """
        # If filings contain raw HTML but no prepared_chunk_data, chunk them first.
        filings = prepared_filings_payload.get("filings", [])
        needs_chunking = any(
            "prepared_chunk_data" not in f and f.get("document_html")
            for f in filings
        )
        if needs_chunking:
            prepared_filings_payload = self._chunk_raw_filings(prepared_filings_payload)

        return self._index_payload(prepared_filings_payload)

    # ------------------------------------------------------------------
    # Retrieval and generation
    # ------------------------------------------------------------------

    def search(self, question: str, k: Optional[int] = None) -> Dict:
        """
        Retrieve the most relevant chunks for a question.

        Returns a dict with keys: query, results (list of chunk dicts), success.
        """
        self._require_indexed()
        return self._pipeline.search(question, k=k or 5)

    def answer(self, question: str, k: Optional[int] = None) -> Dict:
        """
        Retrieve relevant chunks and generate an answer.

        Returns a dict with keys: question, answer, sources, success.
        k defaults to the per-question-type recommended_k.
        """
        self._require_indexed()
        return self._pipeline.answer_question(
            question=question,
            generation_provider=self._generation_provider,
            k=k,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_payload(self, prepared_payload: Dict) -> Dict:
        chunk_records: List[ChunkRecord] = build_chunk_records_from_prepared_filings(
            prepared_payload
        )
        index_result = self._pipeline.index_chunks(chunk_records)
        self._indexed = index_result.get("success", False)
        return {
            **index_result,
            "ticker": prepared_payload.get("ticker"),
            "chunk_count": len(chunk_records),
        }

    def _chunk_raw_filings(self, filings_payload: Dict) -> Dict:
        prepared_filings = []
        for filing in filings_payload.get("filings", []):
            prepared = dict(filing)
            if "prepared_chunk_data" not in prepared:
                prepared["prepared_chunk_data"] = prepare_filing_html_for_chunking(
                    filing.get("document_html") or "",
                    prose_chunk_size=self._prose_chunk_size,
                    prose_chunk_overlap=self._prose_chunk_overlap,
                    table_window=self._table_window,
                    table_overlap=self._table_overlap,
                )
            prepared_filings.append(prepared)
        return {**filings_payload, "filings": prepared_filings}

    def _require_indexed(self) -> None:
        if not self._indexed:
            raise RuntimeError(
                "No chunks indexed. Call index_from_html() or index_from_prepared_filings() first."
            )
