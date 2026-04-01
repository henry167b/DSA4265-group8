import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

from dotenv import load_dotenv

load_dotenv()


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


class GenerationProvider(Protocol):
    def generate_answer(self, question: str, retrieved_chunks: List[Dict]) -> str:
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
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model, input=[text])
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()


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
        }


class FilingRetrievalPipeline:
    """Simple in-memory vector retrieval over prepared filing chunks."""

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider
        self.chunk_records: List[ChunkRecord] = []
        self.chunk_vectors: List[List[float]] = []

    def index_chunks(self, chunk_records: List[ChunkRecord]) -> Dict:
        self.chunk_records = chunk_records
        self.chunk_vectors = self.embedding_provider.embed_texts(
            [record.text for record in chunk_records]
        )
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
        scored_records = []

        for record, vector in zip(self.chunk_records, self.chunk_vectors):
            scored_records.append(
                {
                    **record.to_dict(),
                    "score": _cosine_similarity(query_vector, vector),
                }
            )

        scored_records.sort(key=lambda item: item["score"], reverse=True)
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

        answer = generation_provider.generate_answer(
            question,
            retrieval_result["results"],
        )
        return {
            "question": question,
            "answer": answer,
            "sources": retrieval_result["results"],
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

        for index, chunk_text in enumerate(prepared_chunk_data.get("prose_chunks", [])):
            chunk_records.append(
                ChunkRecord(
                    chunk_id=(
                        f"{filing_common['accession_number']}:prose:{index}"
                    ),
                    text=chunk_text,
                    source_type="prose",
                    **filing_common,
                )
            )

        for index, chunk_text in enumerate(prepared_chunk_data.get("table_chunks", [])):
            table_title = None
            lines = chunk_text.splitlines()
            if lines and lines[0].startswith("[") and lines[0].endswith("]"):
                table_title = lines[0][1:-1]

            chunk_records.append(
                ChunkRecord(
                    chunk_id=(
                        f"{filing_common['accession_number']}:table:{index}"
                    ),
                    text=chunk_text,
                    source_type="table",
                    table_title=table_title,
                    **filing_common,
                )
            )

    return chunk_records


def build_generation_context(retrieved_chunks: List[Dict]) -> str:
    context_blocks = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = (
            f"Ticker: {chunk.get('ticker', 'N/A')} | "
            f"Quarter: {chunk.get('quarter', 'N/A')} | "
            f"Filed: {chunk.get('filing_date', 'N/A')} | "
            f"Source: {chunk.get('source_type', 'N/A')}"
        )
        context_blocks.append(
            f"Chunk {index}\n{metadata}\n{chunk.get('text', '')}"
        )
    return "\n\n---\n\n".join(context_blocks)


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
