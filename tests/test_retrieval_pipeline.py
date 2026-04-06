import sys
from types import SimpleNamespace

sys.modules.setdefault("yfinance", SimpleNamespace())
sys.modules.setdefault("pandas", SimpleNamespace(MultiIndex=type("MultiIndex", (), {})))

from backend.agents.retrieval_pipeline import (
    FilingRetrievalPipeline,
    JsonEmbeddingCache,
    OpenAIEmbeddingProvider,
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
        return [revenue, risk, 0.0]


class FakeGenerationProvider:
    def generate_answer(self, question, retrieved_chunks):
        return f"Answer to '{question}' using {len(retrieved_chunks)} chunks"


class CapturingEmbeddingProvider(FakeEmbeddingProvider):
    def __init__(self):
        self.seen_texts = []

    def embed_texts(self, texts):
        self.seen_texts = list(texts)
        return super().embed_texts(texts)


def test_build_chunk_records_from_prepared_filings_preserves_minimal_metadata():
    prepared = {
        "ticker": "AAPL",
        "filings": [
            {
                "filing_date": "2026-01-30",
                "accession_number": "0000320193-26-000001",
                "quarter": "2026 Q1",
                "form_type": "10-Q",
                "prepared_chunk_data": {
                    "prose_chunk_records": [
                        {
                            "text": "Revenue grew due to iPhone and Services strength.",
                            "section_name": "ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS",
                            "period_end": "Dec 27, 2025",
                        }
                    ],
                    "table_chunk_records": [],
                },
            }
        ],
    }

    chunk_records = build_chunk_records_from_prepared_filings(prepared)

    assert len(chunk_records) == 1
    assert chunk_records[0].source_type == "prose"
    assert chunk_records[0].ticker == "AAPL"
    assert chunk_records[0].section_name == "ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS"
    assert chunk_records[0].period_end == "Dec 27, 2025"


def test_filing_retrieval_pipeline_returns_highest_scoring_prose_chunks():
    prepared = {
        "ticker": "AAPL",
        "filings": [
            {
                "filing_date": "2026-01-30",
                "accession_number": "0000320193-26-000001",
                "quarter": "2026 Q1",
                "form_type": "10-Q",
                "prepared_chunk_data": {
                    "prose_chunks": [
                        "Revenue increased because of iPhone and Services demand.",
                        "Risk factors remained broadly similar to prior filings.",
                    ],
                    "table_chunks": [],
                },
            }
        ],
    }

    chunk_records = build_chunk_records_from_prepared_filings(prepared)
    pipeline = FilingRetrievalPipeline(FakeEmbeddingProvider())
    index_result = pipeline.index_chunks(chunk_records)
    search_result = pipeline.search("What drove revenue growth?", k=1)

    assert index_result["success"] is True
    assert index_result["indexed_chunks"] == 2
    assert search_result["success"] is True
    assert "Revenue" in search_result["results"][0]["text"]


def test_resolve_openai_api_key_prefers_explicit_value(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert resolve_openai_api_key("explicit-key") == "explicit-key"


def test_resolve_openai_api_key_reads_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert resolve_openai_api_key() == "env-key"


def test_build_generation_context_includes_minimal_metadata():
    context = build_generation_context(
        [
            {
                "ticker": "AAPL",
                "quarter": "2026 Q1",
                "filing_date": "2026-01-30",
                "source_type": "prose",
                "section_name": "ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS",
                "period_end": "Dec 27, 2025",
                "text": "Revenue increased because of iPhone and Services demand.",
            }
        ]
    )

    assert "Ticker: AAPL" in context
    assert "Quarter: 2026 Q1" in context
    assert "Section: ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS" in context
    assert "Period End: Dec 27, 2025" in context
    assert "Revenue increased because of iPhone and Services demand." in context


def test_order_retrieved_chunks_for_generation_sorts_oldest_to_newest():
    ordered = order_retrieved_chunks_for_generation(
        [
            {
                "chunk_id": "latest",
                "filing_date": "2026-01-30",
                "accession_number": "b",
                "source_type": "prose",
                "text": "Latest prose",
            },
            {
                "chunk_id": "oldest",
                "filing_date": "2025-10-31",
                "accession_number": "a",
                "source_type": "prose",
                "text": "Oldest prose",
            },
        ]
    )

    assert [chunk["chunk_id"] for chunk in ordered] == ["oldest", "latest"]


def test_filing_retrieval_pipeline_can_generate_answer_from_search_results():
    prepared = {
        "ticker": "AAPL",
        "filings": [
            {
                "filing_date": "2026-01-30",
                "accession_number": "0000320193-26-000001",
                "quarter": "2026 Q1",
                "form_type": "10-Q",
                "prepared_chunk_data": {
                    "prose_chunks": ["Revenue increased because of iPhone and Services demand."],
                    "table_chunks": [],
                },
            }
        ],
    }

    chunk_records = build_chunk_records_from_prepared_filings(prepared)
    pipeline = FilingRetrievalPipeline(FakeEmbeddingProvider())
    pipeline.index_chunks(chunk_records)

    result = pipeline.answer_question(
        question="What drove revenue growth?",
        generation_provider=FakeGenerationProvider(),
        k=1,
    )

    assert result["success"] is True
    assert "Answer to 'What drove revenue growth?'" in result["answer"]
    assert len(result["sources"]) == 1


def test_parse_question_metadata_extracts_filterable_fields():
    metadata = parse_question_metadata(
        "What were the main drivers of revenue growth in 2026 Q1 filed on 2026-01-30?"
    )

    assert metadata["filing_date"] == "2026-01-30"
    assert metadata["quarter"] == "2026 Q1"


def test_parse_question_metadata_extracts_natural_language_period_end():
    metadata = parse_question_metadata(
        "How is profitability trending in the quarter ended December 27, 2025?"
    )

    assert metadata["period_end"] == "2025-12-27"


def test_parse_question_metadata_classifies_risk_and_management_action_questions():
    risk = parse_question_metadata(
        "What are the most material risks explicitly disclosed by management this quarter?"
    )
    actions = parse_question_metadata(
        "What actions has management taken to address these risks or stabilize performance?"
    )

    assert risk["question_type"] == "risk_material"
    assert actions["question_type"] == "management_actions"


def test_expand_oversized_chunk_records_splits_large_text_and_preserves_metadata():
    long_text = ("Revenue increased strongly. " * 500).strip()
    records = [
        build_chunk_records_from_prepared_filings(
            {
                "ticker": "AAPL",
                "filings": [
                    {
                        "filing_date": "2026-01-30",
                        "accession_number": "0000320193-26-000001",
                        "quarter": "2026 Q1",
                        "form_type": "10-Q",
                        "prepared_chunk_data": {
                            "prose_chunk_records": [
                                {
                                    "text": long_text,
                                    "section_name": "ITEM 2",
                                    "period_end": "Dec 27, 2025",
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
    assert all(record.section_name == "ITEM 2" for record in expanded)
    assert expanded[0].chunk_id.startswith("0000320193-26-000001:prose:0#part")


def test_index_chunks_splits_oversized_records_before_embedding():
    long_text = ("Revenue increased strongly. " * 500).strip()
    prepared = {
        "ticker": "AAPL",
        "filings": [
            {
                "filing_date": "2026-01-30",
                "accession_number": "0000320193-26-000001",
                "quarter": "2026 Q1",
                "form_type": "10-Q",
                "prepared_chunk_data": {
                    "prose_chunk_records": [
                        {
                            "text": long_text,
                            "section_name": "ITEM 2",
                            "period_end": "Dec 27, 2025",
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


def test_risk_questions_prefer_risk_factor_sections_when_scores_are_otherwise_equal():
    chunk_records = build_chunk_records_from_prepared_filings(
        {
            "ticker": "AAPL",
            "filings": [
                {
                    "filing_date": "2026-01-30",
                    "accession_number": "0000320193-26-000001",
                    "quarter": "2026 Q1",
                    "form_type": "10-Q",
                    "prepared_chunk_data": {
                        "prose_chunk_records": [
                            {
                                "text": "Management discussed legal exposure.",
                                "section_name": "ITEM 1A. RISK FACTORS",
                                "period_end": "Dec 27, 2025",
                            },
                            {
                                "text": "Management discussed legal exposure.",
                                "section_name": "ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS",
                                "period_end": "Dec 27, 2025",
                            },
                        ]
                    },
                }
            ],
        }
    )

    pipeline = FilingRetrievalPipeline(FakeEmbeddingProvider())
    pipeline.index_chunks(chunk_records)
    result = pipeline.search(
        "What are the most material risks explicitly disclosed by management this quarter?",
        k=1,
    )

    assert result["results"][0]["section_name"] == "ITEM 1A. RISK FACTORS"


def test_management_action_questions_prefer_legal_proceedings_sections_when_scores_are_otherwise_equal():
    chunk_records = build_chunk_records_from_prepared_filings(
        {
            "ticker": "AAPL",
            "filings": [
                {
                    "filing_date": "2026-01-30",
                    "accession_number": "0000320193-26-000001",
                    "quarter": "2026 Q1",
                    "form_type": "10-Q",
                    "prepared_chunk_data": {
                        "prose_chunk_records": [
                            {
                                "text": "Apple appealed the decision and changed its compliance plan.",
                                "section_name": "ITEM 3. LEGAL PROCEEDINGS",
                                "period_end": "Dec 27, 2025",
                            },
                            {
                                "text": "Apple appealed the decision and changed its compliance plan.",
                                "section_name": "ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS",
                                "period_end": "Dec 27, 2025",
                            },
                        ]
                    },
                }
            ],
        }
    )

    pipeline = FilingRetrievalPipeline(FakeEmbeddingProvider())
    pipeline.index_chunks(chunk_records)
    result = pipeline.search(
        "What actions has management taken to address these risks or stabilize performance?",
        k=1,
    )

    assert result["results"][0]["section_name"] == "ITEM 3. LEGAL PROCEEDINGS"


def test_batch_texts_for_embedding_splits_large_requests():
    texts = ["a" * 3000, "b" * 3000, "c" * 3000]
    batches = _batch_texts_for_embedding(texts, max_estimated_tokens=1500)
    assert len(batches) == 3
    assert batches[0] == ["a" * 3000]


def test_json_embedding_cache_reuses_chunk_and_query_embeddings(tmp_path, monkeypatch):
    calls = []

    class FakeEmbeddingsAPI:
        def create(self, model, input):
            calls.append((model, tuple(input)))
            return SimpleNamespace(
                data=[
                    SimpleNamespace(embedding=[float(len(text)), float(index)])
                    for index, text in enumerate(input)
                ]
            )

    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.embeddings = FakeEmbeddingsAPI()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    cache = JsonEmbeddingCache(tmp_path / "embedding_cache.json")
    provider = OpenAIEmbeddingProvider(api_key="test-key", model="test-model", cache=cache)

    first_chunk_embeddings = provider.embed_texts(["revenue", "risk"])
    second_chunk_embeddings = provider.embed_texts(["revenue", "risk"])
    first_query_embedding = provider.embed_query("What drove revenue growth?")
    second_query_embedding = provider.embed_query("What drove revenue growth?")

    assert first_chunk_embeddings == second_chunk_embeddings
    assert first_query_embedding == second_query_embedding
    assert calls == [
        ("test-model", ("revenue", "risk")),
        ("test-model", ("What drove revenue growth?",)),
    ]
