# tests/test_quarterly_sentiment.py

import os
import sys
from collections import Counter

# Make sure the project root is importable when running:
# python .\tests\test_quarterly_sentiment.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.agents.yahoo_finance_agent import YahooFinanceAgent
from backend.agents.quarterly_sentiment import QuarterlySentimentAnalyzer


def print_chunk_preview(chunks, max_chars=220):
    if not chunks:
        print("  No retrieved chunks.")
        return

    for i, chunk in enumerate(chunks[:3], start=1):
        text = str(chunk.get("text", "")).replace("\n", " ").strip()
        section = str(chunk.get("section", "")).strip()
        score = chunk.get("score", None)
        rerank_score = chunk.get("rerank_score", None)

        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
        rerank_str = f"{rerank_score:.4f}" if isinstance(rerank_score, (int, float)) else "N/A"

        print(f"  [{i}] section={section or 'N/A'} score={score_str} rerank={rerank_str}")
        print(f"      {text[:max_chars]}{'...' if len(text) > max_chars else ''}")


def main() -> None:
    ticker = "MSTF"

    yahoo_agent = YahooFinanceAgent()
    analyzer = QuarterlySentimentAnalyzer(
        yahoo_agent=yahoo_agent,
        top_k=5,
        num_quarters=4,
        price_label_threshold=0.05,
    )

    result = analyzer.analyze_ticker(ticker)

    print(f"\nQuarterly sentiment analysis for {ticker}")
    print("=" * 120)

    if not result.get("success"):
        print("Failed:", result.get("error"))
        return

    quarterly_results = result.get("quarterly_results", [])

    print(
        f"{'Quarter':<10} | {'Filing Date':<12} | {'Predicted':<10} | {'Realized':<10} | "
        f"{'% Change':>10} | {'Start Price':>12} | {'End Price':>10}"
    )
    print("-" * 120)

    labels = []
    raw_answers = []

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

        print("\n--- DEBUG ---")
        print("Raw Answer:")
        print(quarter_data.get("predicted_raw_answer", ""))

        print("\nExtracted Reason:")
        print(quarter_data.get("predicted_reason", ""))

        print("\n--- RETRIEVED CHUNKS (Top 3 preview) ---")
        print_chunk_preview(quarter_data.get("retrieved_chunks", []))

        print("-" * 120)

        labels.append(predicted)
        raw_answers.append(quarter_data.get("predicted_raw_answer", ""))

    print("\nSummary")
    print("=" * 120)
    print(f"Generated at: {result.get('generated_at')}")
    print(f"Filing count: {result.get('filing_count')}")
    print(f"Chunk count: {result.get('chunk_count')}")

    print("\nDiagnostics")
    print("=" * 120)
    print("Label distribution:", dict(Counter(labels)))
    print("Unique raw answers:", len(set(raw_answers)), "/", len(raw_answers))

    if len(set(raw_answers)) == 1 and raw_answers:
        print("WARNING: All raw answers are identical.")
    if len(set(labels)) == 1 and labels:
        print("WARNING: All predicted labels are identical.")


if __name__ == "__main__":
    main()