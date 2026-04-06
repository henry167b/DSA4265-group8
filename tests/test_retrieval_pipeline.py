import sys
from types import SimpleNamespace

sys.modules.setdefault("yfinance", SimpleNamespace())
sys.modules.setdefault("pandas", SimpleNamespace(MultiIndex=type("MultiIndex", (), {})))

from backend.agents.retrieval_pipeline import (
    FilingRetrievalPipeline,
    _batch_texts_for_embedding,
    build_chunk_records_from_prepared_filings,
    build_generation_context,
    expand_oversized_chunk_records,
    order_retrieved_chunks_for_generation,
    parse_question_metadata,
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


class CapturingEmbeddingProvider(FakeEmbeddingProvider):
    def __init__(self):
        self.seen_texts = []

    def embed_texts(self, texts):
        self.seen_texts = list(texts)
        return super().embed_texts(texts)


class ZeroEmbeddingProvider:
    def embed_texts(self, texts):
        return [[0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0, 0.0, 0.0]


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
                    "prose_chunk_records": [
                        {
                            "text": "Revenue increased strongly this quarter.",
                            "section_name": "ITEM 2. RESULTS OF OPERATIONS",
                            "statement_type": "income_statement",
                            "metric_name": "revenue",
                            "period_type": "quarter",
                            "period_end": "Aug 28, 2024",
                            "segment_name": None,
                        }
                    ],
                    "table_chunk_records": [
                        {
                            "text": "[Quarterly Revenue]\nData Center (2024): $47,500",
                            "table_title": "Quarterly Revenue",
                            "section_name": "Quarterly Revenue",
                            "statement_type": "income_statement",
                            "metric_name": "Data Center",
                            "period_type": None,
                            "period_end": None,
                            "segment_name": "Data Center",
                        }
                    ],
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
    assert chunk_records[0].statement_type == "income_statement"
    assert chunk_records[1].segment_name == "Data Center"


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


def test_parse_question_metadata_extracts_filterable_fields():
    metadata = parse_question_metadata(
        "What was Google Cloud revenue in 2025 Q3 filed on 2025-10-30?"
    )

    assert metadata["filing_date"] == "2025-10-30"
    assert metadata["quarter"] == "2025 Q3"
    assert metadata["metric_name"] == "revenue"
    assert metadata["segment_name"] == "google cloud"


def test_parse_question_metadata_extracts_natural_language_period_end():
    metadata = parse_question_metadata(
        "What was NVIDIA's revenue in the quarter ended October 26, 2025?"
    )

    assert metadata["period_end"] == "2025-10-26"
    assert metadata["question_type"] == "fact"


def test_search_prefers_metadata_matching_chunks_before_similarity():
    prepared = {
        "ticker": "GOOG",
        "filings": [
            {
                "filing_date": "2025-10-30",
                "accession_number": "good",
                "quarter": "2025 Q3",
                "form_type": "10-Q",
                "prepared_chunk_data": {
                    "table_chunk_records": [
                        {
                            "text": "[Segment Revenue]\nGoogle Cloud revenue: $15,157",
                            "table_title": "Segment Revenue",
                            "section_name": "Segment Revenue",
                            "statement_type": "income_statement",
                            "metric_name": "revenue",
                            "period_type": "quarter",
                            "period_end": "Sep 30, 2025",
                            "segment_name": "google cloud",
                        }
                    ],
                    "prose_chunk_records": [],
                },
            },
            {
                "filing_date": "2025-10-30",
                "accession_number": "wrong",
                "quarter": "2025 Q3",
                "form_type": "10-Q",
                "prepared_chunk_data": {
                    "table_chunk_records": [
                        {
                            "text": "[Narrative]\nCloud advertising strategy and ad revenue discussion.",
                            "table_title": "Narrative",
                            "section_name": "Narrative",
                            "statement_type": "income_statement",
                            "metric_name": "revenue",
                            "period_type": "quarter",
                            "period_end": "Sep 30, 2025",
                            "segment_name": None,
                        }
                    ],
                    "prose_chunk_records": [],
                },
            },
        ],
    }

    pipeline = FilingRetrievalPipeline(FakeEmbeddingProvider())
    pipeline.index_chunks(build_chunk_records_from_prepared_filings(prepared))

    result = pipeline.search("What was Google Cloud revenue in 2025 Q3 filed on 2025-10-30?", k=2)

    assert result["success"] is True
    assert result["results"][0]["chunk_id"] == "good:table:0"


def test_search_uses_bm25_signal_when_dense_scores_tie():
    prepared = {
        "ticker": "NVDA",
        "filings": [
            {
                "filing_date": "2025-11-19",
                "accession_number": "target",
                "quarter": "2025 Q4",
                "form_type": "10-Q",
                "prepared_chunk_data": {
                    "table_chunk_records": [
                        {
                            "text": "[Segment Revenue]\nBlackwell systems revenue: $51,215",
                            "table_title": "Segment Revenue",
                            "section_name": "Segment Revenue",
                            "statement_type": "income_statement",
                            "metric_name": "revenue",
                            "period_type": "quarter",
                            "period_end": "October 26, 2025",
                            "segment_name": "Data Center",
                        },
                        {
                            "text": "[Segment Revenue]\nLegacy systems revenue: $4,100",
                            "table_title": "Segment Revenue",
                            "section_name": "Segment Revenue",
                            "statement_type": "income_statement",
                            "metric_name": "revenue",
                            "period_type": "quarter",
                            "period_end": "October 26, 2025",
                            "segment_name": "Data Center",
                        },
                    ],
                    "prose_chunk_records": [],
                },
            }
        ],
    }

    pipeline = FilingRetrievalPipeline(ZeroEmbeddingProvider())
    pipeline.index_chunks(build_chunk_records_from_prepared_filings(prepared))

    result = pipeline.search("What was Blackwell systems revenue in the quarter ended October 26, 2025?", k=1)

    assert result["success"] is True
    assert result["results"][0]["chunk_id"] == "target:table:0"


def test_expand_oversized_chunk_records_splits_large_text_and_preserves_metadata():
    long_text = ("Revenue increased strongly. " * 500).strip()
    records = [
        build_chunk_records_from_prepared_filings(
            {
                "ticker": "NVDA",
                "filings": [
                    {
                        "filing_date": "2024-08-28",
                        "accession_number": "0001045810-24-000123",
                        "quarter": "2024 Q3",
                        "form_type": "10-Q",
                        "prepared_chunk_data": {
                            "prose_chunk_records": [
                                {
                                    "text": long_text,
                                    "section_name": "ITEM 2",
                                    "statement_type": "income_statement",
                                    "metric_name": "revenue",
                                    "period_type": "quarter",
                                    "period_end": "Aug 28, 2024",
                                    "segment_name": None,
                                }
                            ]
                        },
                    }
                ],
            }
        )[0]
    ]

    expanded = expand_oversized_chunk_records(records, max_chars=1000, overlap_chars=100)

    assert len(expanded) > 1
    assert all(record.statement_type == "income_statement" for record in expanded)
    assert expanded[0].chunk_id.startswith("0001045810-24-000123:prose:0#part")


def test_index_chunks_splits_oversized_records_before_embedding():
    long_text = ("Revenue increased strongly. " * 500).strip()
    prepared = {
        "ticker": "NVDA",
        "filings": [
            {
                "filing_date": "2024-08-28",
                "accession_number": "0001045810-24-000123",
                "quarter": "2024 Q3",
                "form_type": "10-Q",
                "prepared_chunk_data": {
                    "prose_chunk_records": [
                        {
                            "text": long_text,
                            "section_name": "ITEM 2",
                            "statement_type": "income_statement",
                            "metric_name": "revenue",
                            "period_type": "quarter",
                            "period_end": "Aug 28, 2024",
                            "segment_name": None,
                        }
                    ]
                },
            }
        ],
    }

    embedding_provider = CapturingEmbeddingProvider()
    pipeline = FilingRetrievalPipeline(embedding_provider)
    result = pipeline.index_chunks(build_chunk_records_from_prepared_filings(prepared))

    assert result["success"] is True
    assert len(pipeline.chunk_records) > 1
    assert all(len(text) <= 6000 for text in embedding_provider.seen_texts)


def test_batch_texts_for_embedding_splits_large_requests():
    texts = ["a" * 3000, "b" * 3000, "c" * 3000]

    batches = _batch_texts_for_embedding(texts, max_estimated_tokens=1500)

    assert len(batches) == 3
    assert batches[0] == ["a" * 3000]
