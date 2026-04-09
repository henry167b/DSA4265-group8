from RAG_test.common import resolve_companies


def test_resolve_companies_defaults_to_all_benchmark_companies():
    companies = resolve_companies()

    assert [company["ticker"] for company in companies] == [
        "AAPL",
        "GOOG",
        "META",
        "NVDA",
        "TSLA",
    ]


def test_resolve_companies_returns_requested_subset_in_requested_order():
    companies = resolve_companies(["nvda", "tsla"])

    assert [company["ticker"] for company in companies] == ["NVDA", "TSLA"]


def test_resolve_companies_rejects_unknown_tickers():
    try:
        resolve_companies(["MSFT"])
    except ValueError as exc:
        assert "Unsupported ticker(s): MSFT." in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected resolve_companies to reject unknown tickers")
