import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.modules.setdefault("yfinance", SimpleNamespace())
sys.modules.setdefault("pandas", SimpleNamespace(MultiIndex=type("MultiIndex", (), {})))

from backend.agents.financial_data_agent import YahooFinanceAgent


def _mock_response(payload=None, text="", status_code=200):
    class MockResponse:
        def __init__(self, payload, text, status_code):
            self._payload = payload
            self.text = text
            self.status_code = status_code

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    return MockResponse(payload, text, status_code)


@patch("backend.agents.financial_data_agent.time.sleep", return_value=None)
@patch("backend.agents.financial_data_agent.requests.get")
def test_get_recent_10q_filings_includes_document_html_when_requested(mock_get, _mock_sleep):
    agent = YahooFinanceAgent(user_email="test@example.com")

    company_tickers_payload = {
        "0": {"ticker": "NVDA", "cik_str": 1045810, "title": "NVIDIA CORP"}
    }
    submissions_payload = {
        "name": "NVIDIA CORP",
        "filings": {
            "recent": {
                "form": ["10-Q", "8-K", "10-Q"],
                "accessionNumber": ["0001045810-24-000123", "0001045810-24-000124", "0001045810-24-000125"],
                "filingDate": ["2024-08-28", "2024-08-20", "2024-05-29"],
                "primaryDocument": ["nvda-20240728x10q.htm", "nvda-8k.htm", "nvda-20240428x10q.htm"],
            }
        },
    }

    mock_get.side_effect = [
        _mock_response(payload=company_tickers_payload),
        _mock_response(payload=submissions_payload),
        _mock_response(text="<html><body><h1>Quarterly Report</h1><p>Revenue grew strongly.</p></body></html>"),
        _mock_response(text="<html><body><p>Second filing body.</p></body></html>"),
    ]

    result = agent.get_recent_10q_filings("NVDA", num_quarters=2, include_document_html=True)

    assert result["success"] is True
    assert len(result["filings"]) == 2
    assert result["filings"][0]["document_fetch_success"] is True
    assert result["filings"][0]["filing_date"] == "2024-05-29"
    assert result["filings"][1]["filing_date"] == "2024-08-28"
    assert result["filings"][0]["document_html"] == "<html><body><p>Second filing body.</p></body></html>"
    assert "Quarterly Report" in result["filings"][1]["document_html"]
    assert "Revenue grew strongly." in result["filings"][1]["document_html"]
    assert result["filings"][0]["document_html_length"] == len(result["filings"][0]["document_html"])
    assert result["filings"][1]["document_html_length"] == len(result["filings"][1]["document_html"])


@patch("backend.agents.financial_data_agent.time.sleep", return_value=None)
@patch("backend.agents.financial_data_agent.requests.get")
def test_get_recent_10q_filings_skips_document_download_by_default(mock_get, _mock_sleep):
    agent = YahooFinanceAgent(user_email="test@example.com")

    company_tickers_payload = {
        "0": {"ticker": "NVDA", "cik_str": 1045810, "title": "NVIDIA CORP"}
    }
    submissions_payload = {
        "name": "NVIDIA CORP",
        "filings": {
            "recent": {
                "form": ["10-Q"],
                "accessionNumber": ["0001045810-24-000123"],
                "filingDate": ["2024-08-28"],
                "primaryDocument": ["nvda-20240728x10q.htm"],
            }
        },
    }

    mock_get.side_effect = [
        _mock_response(payload=company_tickers_payload),
        _mock_response(payload=submissions_payload),
    ]

    result = agent.get_recent_10q_filings("NVDA", num_quarters=1)

    assert result["success"] is True
    assert len(result["filings"]) == 1
    assert "document_html" not in result["filings"][0]
    assert mock_get.call_count == 2


@patch("backend.agents.financial_data_agent.time.sleep", return_value=None)
@patch("backend.agents.financial_data_agent.requests.get")
def test_get_recent_10q_filings_sorts_filings_chronologically(mock_get, _mock_sleep):
    agent = YahooFinanceAgent(user_email="test@example.com")

    company_tickers_payload = {
        "0": {"ticker": "NVDA", "cik_str": 1045810, "title": "NVIDIA CORP"}
    }
    submissions_payload = {
        "name": "NVIDIA CORP",
        "filings": {
            "recent": {
                "form": ["10-Q", "10-Q", "10-Q"],
                "accessionNumber": [
                    "0001045810-24-000123",
                    "0001045810-24-000121",
                    "0001045810-24-000122",
                ],
                "filingDate": ["2024-08-28", "2024-02-28", "2024-05-29"],
                "primaryDocument": [
                    "nvda-20240728x10q.htm",
                    "nvda-20240128x10q.htm",
                    "nvda-20240428x10q.htm",
                ],
            }
        },
    }

    mock_get.side_effect = [
        _mock_response(payload=company_tickers_payload),
        _mock_response(payload=submissions_payload),
    ]

    result = agent.get_recent_10q_filings("NVDA", num_quarters=3)

    assert result["success"] is True
    assert [filing["filing_date"] for filing in result["filings"]] == [
        "2024-02-28",
        "2024-05-29",
        "2024-08-28",
    ]


def test_prepare_recent_10q_filings_for_chunking_adds_prepared_chunk_data():
    agent = YahooFinanceAgent(user_email="test@example.com")

    filings_payload = {
        "ticker": "NVDA",
        "cik": "0001045810",
        "company_name": "NVIDIA CORP",
        "filings": [
            {
                "filing_date": "2024-08-28",
                "accession_number": "0001045810-24-000123",
                "form_type": "10-Q",
                "document_url": "https://www.sec.gov/example.htm",
                "quarter": "2024 Q3",
                "document_fetch_success": True,
                "document_html": "<html><body><h1>ITEM 2</h1><p>Revenue expanded.</p></body></html>",
                "document_html_length": 69,
            }
        ],
        "success": True,
    }

    with patch.object(agent, "get_recent_10q_filings", return_value=filings_payload):
        prepared = agent.prepare_recent_10q_filings_for_chunking("NVDA", num_quarters=1)

    assert prepared["prepared_for_chunking"] is True
    prepared_filing = prepared["filings"][0]
    assert "prepared_chunk_data" in prepared_filing
    assert prepared_filing["prepared_chunk_data"]["chunks"]
    assert "Revenue expanded." in prepared_filing["prepared_chunk_data"]["prose_text"]


def test_build_recent_10q_retrieval_corpus_flattens_chunk_records():
    agent = YahooFinanceAgent(user_email="test@example.com")

    prepared_payload = {
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
                    "chunks": [
                        "Revenue increased strongly this quarter.",
                        "[Quarterly Revenue]\nData Center (2024): $47,500",
                    ],
                },
            }
        ],
        "success": True,
    }

    with patch.object(agent, "prepare_recent_10q_filings_for_chunking", return_value=prepared_payload):
        corpus = agent.build_recent_10q_retrieval_corpus("NVDA", num_quarters=1)

    assert corpus["success"] is True
    assert corpus["chunk_count"] == 2
    assert corpus["chunk_records"][0]["source_type"] == "prose"
    assert corpus["chunk_records"][1]["source_type"] == "table"


def test_answer_10q_question_uses_retrieval_pipeline_and_generation():
    agent = YahooFinanceAgent(user_email="test@example.com")

    fake_pipeline = SimpleNamespace(
        answer_question=lambda question, generation_provider, k: {
            "question": question,
            "answer": "Generated answer",
            "sources": [{"chunk_id": "chunk-1", "text": "Revenue increased strongly."}],
            "success": True,
        }
    )

    with patch.object(
        agent,
        "create_recent_10q_retrieval_pipeline",
        return_value={
            "pipeline": fake_pipeline,
            "filings": [{"filing_date": "2024-08-28"}],
            "success": True,
        },
    ), patch("backend.agents.financial_data_agent.OpenAIChatGenerationProvider"):
        result = agent.answer_10q_question(
            ticker="NVDA",
            question="What changed in revenue?",
        )

    assert result["success"] is True
    assert result["answer"] == "Generated answer"
    assert result["sources"][0]["chunk_id"] == "chunk-1"
