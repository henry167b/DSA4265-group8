# tests/test_quarterly_sentiment.py

import os
import sys

# Make sure the project root is importable when running:
# python .\tests\test_quarterly_sentiment.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.agents.yahoo_finance_agent import YahooFinanceAgent
from backend.agents.quarterly_sentiment import QuarterlySentimentAnalyzer


def main() -> None:
    ticker = "NVDA"

    yahoo_agent = YahooFinanceAgent()
    analyzer = QuarterlySentimentAnalyzer(
        yahoo_agent=yahoo_agent,
        top_k=5,
        num_quarters=4,
        price_label_threshold=0.02,
    )

    result = analyzer.analyze_ticker(ticker)

    print(f"\nQuarterly sentiment analysis for {ticker}")
    print("=" * 110)

    if not result.get("success"):
        print("Failed:", result.get("error"))
        return

    quarterly_results = result.get("quarterly_results", [])

    print(
        f"{'Quarter':<10} | {'Filing Date':<12} | {'Predicted':<10} | {'Realized':<10} | "
        f"{'% Change':>10} | {'Start Price':>12} | {'End Price':>10}"
    )
    print("-" * 110)

    for quarter_data in quarterly_results:
        quarter = quarter_data.get("quarter", "")
        filing_date = quarter_data.get("filing_date", "")
        predicted = quarter_data.get("predicted_label", "N/A")
        realized = quarter_data.get("realized_next_quarter_label", "N/A")

        pct_change = quarter_data.get("realized_next_quarter_pct_change")
        start_price = quarter_data.get("realized_next_quarter_start_price")
        end_price = quarter_data.get("realized_next_quarter_end_price")

        pct_change_str = f"{pct_change:.2f}%" if pct_change is not None else "N/A"
        start_price_str = f"{start_price:.2f}" if start_price is not None else "N/A"
        end_price_str = f"{end_price:.2f}" if end_price is not None else "N/A"

        print(
            f"{quarter:<10} | {filing_date:<12} | {predicted:<10} | {realized:<10} | "
            f"{pct_change_str:>10} | {start_price_str:>12} | {end_price_str:>10}"
        )

    print("\nSummary")
    print("=" * 110)
    print(f"Generated at: {result.get('generated_at')}")
    print(f"Filing count: {result.get('filing_count')}")
    print(f"Chunk count: {result.get('chunk_count')}")


if __name__ == "__main__":
    main()