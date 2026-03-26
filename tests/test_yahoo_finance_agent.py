# Example usage and test function
def test_agent():
    """Test the Yahoo Finance Agent with Nvidia."""
    agent = YahooFinanceAgent()

    result = agent.get_stock_data("NVDA")

    if result["success"]:
        print("Agent 1 executed successfully!")
        print("\n" + "="*60)
        print(agent.format_for_next_agent(result))
        print("="*60)

        print("\n📊 Raw data structure:")
        print(f"- Ticker: {result['ticker']}")
        print(f"- Current Price: ${result['current_data'].get('current_price')}")
        print(f"- P/E Ratio: {result['key_metrics'].get('pe_ratio')}")
        print(f"- Analyst Rating: {result['analyst_recommendations'].get('recommendation_key')}")
        print(f"- News Count: {len(result['recent_news'])}")

        # Test with multiple tickers
        print("\n Testing with multiple tickers...")
        tickers = ["AAPL", "MSFT", "GOOGL"]

        for ticker in tickers:
            result = agent.get_stock_data(ticker)
            if result["success"]:
                price = result['current_data'].get('current_price', 'N/A')
                print(f"{ticker}: ${price}")

    else:
        print(f"❌ Error: {result['error']}")


if __name__ == "__main__":
    test_agent()
