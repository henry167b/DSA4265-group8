import sys
from types import SimpleNamespace

sys.modules.setdefault("yfinance", SimpleNamespace())
sys.modules.setdefault("pandas", SimpleNamespace(MultiIndex=type("MultiIndex", (), {})))

from backend.agents.retrieval_pipeline import (
    FilingRetrievalPipeline,
    build_chunk_records_from_prepared_filings,
    build_generation_context,
    order_retrieved_chunks_for_generation,
    resolve_openai_api_key,
)


class FakeEmbeddingProvider:
    def embed_texts(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        lowered = text.lower()
        revenue = 1.0 if "revenue" in lowered else 0.0
        risk = 1.0 if "risk" in lowered else 0.0
        table = 1.0 if lowered.startswith("[") else 0.0
        return [revenue, risk, table]


class FakeGenerationProvider:
    def generate_answer(self, question, retrieved_chunks):
        return f"Answer to '{question}' using {len(retrieved_chunks)} chunks"


def test_build_chunk_records_from_prepared_filings_preserves_metadata():
    prepared = {
        "ticker": "NVDA",
        "filings": [
            {
                "filing_date": "2024-08-28",
                "accession_number": "0001045810-24-000123",
                "quarter": "2024 Q3",
                "form_type": "10-Q",
                "prepared_chunk_data": {
                    "prose_chunks": ["Revenue increased strongly this quarter."],
                    "table_chunks": ["[Quarterly Revenue]\nData Center (2024): $47,500"],
                },
            }
        ],
    }

    chunk_records = build_chunk_records_from_prepared_filings(prepared)

    assert len(chunk_records) == 2
    assert chunk_records[0].source_type == "prose"
    assert chunk_records[1].source_type == "table"
    assert chunk_records[1].table_title == "Quarterly Revenue"
    assert chunk_records[0].ticker == "NVDA"


def test_filing_retrieval_pipeline_returns_highest_scoring_chunks():
    prepared = {
        "ticker": "NVDA",
        "filings": [
            {
                "filing_date": "2024-08-28",
                "accession_number": "0001045810-24-000123",
                "quarter": "2024 Q3",
                "form_type": "10-Q",
                "prepared_chunk_data": {
                    "prose_chunks": [
                        "Revenue increased strongly this quarter.",
                        "Risk factors remained broadly similar.",
                    ],
                    "table_chunks": ["[Quarterly Revenue]\nData Center (2024): $47,500"],
                },
            }
        ],
    }

    chunk_records = build_chunk_records_from_prepared_filings(prepared)
    pipeline = FilingRetrievalPipeline(FakeEmbeddingProvider())
    index_result = pipeline.index_chunks(chunk_records)
    search_result = pipeline.search("What was revenue?", k=2)

    assert index_result["success"] is True
    assert index_result["indexed_chunks"] == 3
    assert search_result["success"] is True
    assert len(search_result["results"]) == 2
    assert "Revenue" in search_result["results"][0]["text"]


def test_resolve_openai_api_key_prefers_explicit_value(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    assert resolve_openai_api_key("explicit-key") == "explicit-key"


def test_resolve_openai_api_key_reads_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    assert resolve_openai_api_key() == "env-key"


def test_build_generation_context_includes_metadata():
    context = build_generation_context(
        [
            {
                "ticker": "NVDA",
                "quarter": "2024 Q3",
                "filing_date": "2024-08-28",
                "source_type": "table",
                "text": "[Quarterly Revenue]\nData Center (2024): $47,500",
            }
        ]
    )

    assert "Ticker: NVDA" in context
    assert "Quarter: 2024 Q3" in context
    assert "Data Center (2024): $47,500" in context
    assert "intentionally ordered chronologically" in context


def test_order_retrieved_chunks_for_generation_sorts_oldest_to_newest():
    ordered = order_retrieved_chunks_for_generation(
        [
            {
                "chunk_id": "latest-table",
                "filing_date": "2024-08-28",
                "accession_number": "b",
                "source_type": "table",
                "text": "Latest table",
            },
            {
                "chunk_id": "oldest-prose",
                "filing_date": "2024-02-28",
                "accession_number": "a",
                "source_type": "prose",
                "text": "Oldest prose",
            },
            {
                "chunk_id": "latest-prose",
                "filing_date": "2024-08-28",
                "accession_number": "b",
                "source_type": "prose",
                "text": "Latest prose",
            },
        ]
    )

    assert [chunk["chunk_id"] for chunk in ordered] == [
        "oldest-prose",
        "latest-prose",
        "latest-table",
    ]


def test_filing_retrieval_pipeline_can_generate_answer_from_search_results():
    prepared = {
        "ticker": "NVDA",
        "filings": [
            {
                "filing_date": "2024-08-28",
                "accession_number": "0001045810-24-000123",
                "quarter": "2024 Q3",
                "form_type": "10-Q",
                "prepared_chunk_data": {
                    "prose_chunks": ["Revenue increased strongly this quarter."],
                    "table_chunks": ["[Quarterly Revenue]\nData Center (2024): $47,500"],
                },
            }
        ],
    }

    chunk_records = build_chunk_records_from_prepared_filings(prepared)
    pipeline = FilingRetrievalPipeline(FakeEmbeddingProvider())
    pipeline.index_chunks(chunk_records)

    result = pipeline.answer_question(
        question="What was revenue?",
        generation_provider=FakeGenerationProvider(),
        k=2,
    )

    assert result["success"] is True
    assert "Answer to 'What was revenue?'" in result["answer"]
    assert len(result["sources"]) == 2


def test_answer_question_reorders_selected_chunks_for_generation():
    prepared = {
        "ticker": "NVDA",
        "filings": [
            {
                "filing_date": "2024-08-28",
                "accession_number": "0001045810-24-000123",
                "quarter": "2024 Q3",
                "form_type": "10-Q",
                "prepared_chunk_data": {
                    "prose_chunks": ["Revenue increased strongly this quarter."],
                    "table_chunks": [],
                },
            },
            {
                "filing_date": "2024-05-29",
                "accession_number": "0001045810-24-000122",
                "quarter": "2024 Q2",
                "form_type": "10-Q",
                "prepared_chunk_data": {
                    "prose_chunks": ["Revenue was lower in the prior quarter."],
                    "table_chunks": [],
                },
            },
        ],
    }

    captured = {}

    class CapturingGenerationProvider:
        def generate_answer(self, question, retrieved_chunks):
            captured["retrieved_chunks"] = retrieved_chunks
            return "Chronological answer"

    chunk_records = build_chunk_records_from_prepared_filings(prepared)
    pipeline = FilingRetrievalPipeline(FakeEmbeddingProvider())
    pipeline.index_chunks(chunk_records)

    result = pipeline.answer_question(
        question="What was revenue?",
        generation_provider=CapturingGenerationProvider(),
        k=2,
    )

    assert result["success"] is True
    assert result["answer"] == "Chronological answer"
    assert [chunk["filing_date"] for chunk in captured["retrieved_chunks"]] == [
        "2024-05-29",
        "2024-08-28",
    ]
