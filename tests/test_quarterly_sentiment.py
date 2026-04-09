# tests/test_quarterly_sentiment_integration.py

import os
import sys
from collections import Counter
from datetime import datetime

# Make sure the project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.agents.retrieval_pipeline import OpenAIEmbeddingProvider, OpenAIChatGenerationProvider
from backend.agents.quarterly_sentiment import QuarterlySentimentPipeline


def main() -> None:
    ticker = "AAPL"
    quarters = ["2025Q1", "2025Q2", "2025Q3", "2025Q4"]

    embedding_provider = OpenAIEmbeddingProvider()
    generation_provider = OpenAIChatGenerationProvider(model="gpt-4o-mini")

    pipeline = QuarterlySentimentPipeline(
        embedding_provider=embedding_provider,
        generation_provider=generation_provider,
    )

    print(f"\nQuarterly sentiment analysis for {ticker}")
    print("=" * 120)

    all_results = []

    for quarter in quarters:
        result = pipeline.analyze_ticker_for_quarter(ticker, quarter)
        all_results.append(result)

        print(f"\nQuarter {quarter}:")
        print(f"Predicted Label: {result['predicted_label']}")
        print(f"Actual Label: {result['actual_label']}")
        print(f"Meta: {result['meta']}")
        print("-" * 120)

    # Summary
    labels = [r["predicted_label"] for r in all_results]
    print("\nSummary")
    print("=" * 120)
    print(f"Label distribution: {dict(Counter(labels))}")
    print(f"Unique predicted labels: {len(set(labels))} / {len(labels)}")

    if len(set(labels)) == 1 and labels:
        print("WARNING: All predicted labels are identical.")


if __name__ == "__main__":
    main()