def test_yahoo_finance_only():
    """Test only Yahoo Finance functionality."""
    agent = YahooFinanceAgent()

    result = agent.get_stock_data("NVDA")

    if result["success"]:
        print("Yahoo Finance Agent executed successfully!")
        print("\n" + "="*60)
        print(agent.format_for_next_agent(result))
        print("="*60)
    else:
        print(f"Error: {result['error']}")


def test_sec_only():
    """Test only SEC EDGAR functionality."""
    agent = YahooFinanceAgent()
    ticker = "NVDA"

    print(f"Testing SEC EDGAR data for {ticker}...\n")

    # Test CIK conversion
    cik = agent.ticker_to_cik(ticker)
    print(f"CIK: {cik}")

    # Test 10-Q filings
    print("\n" + "="*60)
    filings = agent.get_recent_10q_filings(ticker)
    if filings.get('success'):
        print(agent.format_sec_data_for_next_agent(filings))
    else:
        print(f"Error: {filings.get('error')}")

    # Test financial facts
    print("\n" + "="*60)
    facts = agent.get_financial_facts(ticker)
    if facts.get('success'):
        print(agent.format_financial_facts_for_next_agent(facts))
    else:
        print(f"Error: {facts.get('error')}")


def test_complete_agent():
    """Test the complete agent with both Yahoo Finance and SEC data."""
    agent = YahooFinanceAgent()

    ticker = "NVDA"
    print(f"Testing complete Agent 1 with {ticker}...\n")

    # Get complete analysis data
    complete_data = agent.get_complete_analysis_data(ticker)

    if complete_data.get("success"):
        print("Agent 1 executed successfully!")

        # Show Yahoo Finance data
        print("\n" + "="*60)
        print("YAHOO FINANCE DATA")
        print("="*60)
        print(agent.format_for_next_agent(complete_data))

        # Show SEC Filings data
        print("\n" + "="*60)
        print("SEC EDGAR DATA")
        print("="*60)
        sec_filings = complete_data.get('sec_filings', {})
        print(agent.format_sec_data_for_next_agent(sec_filings))

        # Show Financial Facts
        print("\n" + "="*60)
        print("XBRL FINANCIAL FACTS")
        print("="*60)
        financial_facts = complete_data.get('financial_facts', {})
        print(agent.format_financial_facts_for_next_agent(financial_facts))

        # Show data summary
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        print(f"📄 10-Q Filings Retrieved: {len(sec_filings.get('filings', []))}")

        metrics = financial_facts.get('financial_metrics', {})
        print(f"Financial Metrics Retrieved: {len(metrics)}")
        for metric in metrics.keys():
            print(f"   - {metric}")

    else:
        print(f"Error: {complete_data.get('error')}")


def test_multiple_tickers():
    """Test the agent with multiple tickers."""
    agent = YahooFinanceAgent()
    tickers = ["NVDA", "AAPL", "MSFT", "GOOGL"]

    print("Testing multiple tickers...\n")

    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"Processing {ticker}...")
        print('='*50)

        # Get Yahoo Finance data
        stock_data = agent.get_stock_data(ticker)
        if stock_data.get("success"):
            price = stock_data['current_data'].get('current_price')
            print(f"{ticker}: ${price:.2f}" if price else f"{ticker}: Data retrieved")
        else:
            print(f"{ticker}: {stock_data.get('error')}")

        # Get SEC filings
        sec_data = agent.get_recent_10q_filings(ticker, num_quarters=2)
        if sec_data.get('success'):
            print(f" Found {len(sec_data.get('filings', []))} recent 10-Q filings")
        else:
            print(f" SEC data: {sec_data.get('error', 'Unknown error')}")


if __name__ == "__main__":
    print("="*60)
    print("AGENT 1: YAHOO FINANCE + SEC EDGAR DATA FETCHER")
    print("="*60)

    # Choose which test to run
    print("\n1. Testing Yahoo Finance only...\n")
    test_yahoo_finance_only()

    print("\n\n2. Testing SEC EDGAR only...\n")
    test_sec_only()

    print("\n\n3. Testing complete agent...\n")
    test_complete_agent()

    print("\n\n4. Testing multiple tickers...\n")
    test_multiple_tickers()
