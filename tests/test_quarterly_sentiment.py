import os
import re
import sys
from collections import Counter

# Make sure the project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.agents.quarterly_sentiment import QuarterlySentimentAnalyzer


def _extract_section(chunk: dict) -> str:
    return (
        chunk.get("section_name")
        or chunk.get("section")
        or chunk.get("metadata", {}).get("section_name")
        or "UNKNOWN"
    )


def _preview_text(text: str, limit: int = 300) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    normalized = re.sub(r"-{8,}", " --- ", normalized)
    return normalized[:limit]


def main() -> None:
    ticker = "NVDA"
    num_quarters = 4  # Analyze last 4 quarters

    print(f"\nRunning real integration test for {ticker} (last {num_quarters} quarters)")
    print("=" * 120)

    # Create analyzer with real providers
    analyzer = QuarterlySentimentAnalyzer(
        yahoo_agent=None,  # Use default YahooFinanceAgent
        num_quarters=num_quarters,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

    result = analyzer.analyze_ticker(ticker)

    if not result.get("success"):
        print(f"Error fetching filings: {result.get('error')}")
        return

    all_results = result.get("quarterly_results", [])

    for q_result in all_results:
        print(f"\nQuarter {q_result['quarter']}:")
        print(f"Filing Date: {q_result['filing_date']}")
        print(f"Predicted Label: {q_result['predicted_label']}")
        print(f"Actual Label: {q_result['actual_label']}")
        print(f"Realized % Change: {q_result['realized_next_quarter_pct_change']}")
        print(f"Chunks Used: {q_result['chunks_used']}")
        print(f"Selected Sections: {q_result['selected_sections']}")
        print(f"Price Error: {q_result['price_error']}")
        print("-" * 60)

        # --- DEBUG: print filtered chunks ---
        retrieved_chunks = q_result.get("retrieved_chunks", [])
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            text = chunk.get("text", "")
            section = _extract_section(chunk)
            source_type = chunk.get("source_type", "UNKNOWN")
            chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
            selection_debug = chunk.get("selection_debug", {})

            print(
                f"Chunk {idx} | ID: {chunk_id} | Source: {source_type} | "
                f"Section: {section} | Len: {len(text)}"
            )
            if selection_debug:
                debug_bucket = selection_debug.get("section_bucket", "UNKNOWN")
                debug_rank = selection_debug.get("section_rank", "?")
                debug_score = selection_debug.get("adjusted_score", "?")
                debug_reasons = "; ".join(selection_debug.get("reasons", []))
                print(
                    f"Selection Debug | Bucket: {debug_bucket} | "
                    f"Rank in Section: {debug_rank} | Adjusted Score: {debug_score}"
                )
                print(f"Reasons: {debug_reasons}")
            print(_preview_text(text, 300))
            print("-" * 40)

        print("=" * 120)

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