#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from backend.agents.financial_data_agent import YahooFinanceAgent
except ModuleNotFoundError as exc:
    missing_module = getattr(exc, "name", "a required dependency")
    raise SystemExit(
        f"Missing dependency: {missing_module}. "
        "Install project dependencies first, then rerun this script."
    ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual end-to-end tester for 10-Q retrieval and RAG generation."
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. NVDA")
    parser.add_argument("--question", help="Question to ask over recent 10-Q filings")
    parser.add_argument(
        "--mode",
        choices=["prepare", "retrieve", "answer"],
        default="answer",
        help="prepare = inspect chunking, retrieve = top chunks only, answer = full RAG",
    )
    parser.add_argument("--num-quarters", type=int, default=4, help="Number of recent 10-Qs")
    parser.add_argument("-k", type=int, default=5, help="Top k chunks to retrieve")
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model for retrieval",
    )
    parser.add_argument(
        "--generation-model",
        default="gpt-4o-mini",
        help="Chat model for answer generation",
    )
    return parser


def print_sources(sources):
    if not sources:
        print("No sources returned.")
        return

    for index, source in enumerate(sources, start=1):
        score = source.get("score")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
        print(
            f"{index}. {source.get('quarter', 'N/A')} | "
            f"{source.get('source_type', 'N/A')} | score={score_text}"
        )
        print(source.get("text", "")[:500])
        print("-" * 80)


def run_prepare(agent: YahooFinanceAgent, ticker: str, num_quarters: int) -> None:
    prepared = agent.prepare_recent_10q_filings_for_chunking(
        ticker=ticker,
        num_quarters=num_quarters,
    )
    print(f"Success: {prepared.get('success')}")
    print(f"Filings returned: {len(prepared.get('filings', []))}")
    for filing in prepared.get("filings", []):
        chunk_data = filing.get("prepared_chunk_data", {})
        print(
            f"{filing.get('quarter')} | prose={len(chunk_data.get('prose_chunks', []))} "
            f"| tables={len(chunk_data.get('table_chunks', []))} "
            f"| total={len(chunk_data.get('chunks', []))}"
        )
        if chunk_data.get("chunks"):
            print(chunk_data["chunks"][0][:500])
            print("-" * 80)


def run_retrieve(
    agent: YahooFinanceAgent,
    ticker: str,
    question: str,
    num_quarters: int,
    k: int,
    embedding_model: str,
) -> None:
    pipeline_result = agent.create_recent_10q_retrieval_pipeline(
        ticker=ticker,
        num_quarters=num_quarters,
        embedding_model=embedding_model,
    )
    print(f"Success: {pipeline_result.get('success')}")
    print(f"Indexed chunks: {pipeline_result.get('chunk_count', 0)}")
    search_result = pipeline_result["pipeline"].search(question, k=k)
    print_sources(search_result.get("results", []))


def run_answer(
    agent: YahooFinanceAgent,
    ticker: str,
    question: str,
    num_quarters: int,
    k: int,
    embedding_model: str,
    generation_model: str,
) -> None:
    result = agent.answer_10q_question(
        ticker=ticker,
        question=question,
        num_quarters=num_quarters,
        k=k,
        embedding_model=embedding_model,
        generation_model=generation_model,
    )
    print(f"Success: {result.get('success')}")
    print("\nAnswer:\n")
    print(result.get("answer"))
    print("\nSources:\n")
    print_sources(result.get("sources", []))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    agent = YahooFinanceAgent()

    if args.mode in {"retrieve", "answer"} and not args.question:
        parser.error("--question is required for retrieve and answer modes")

    if args.mode == "prepare":
        run_prepare(agent, args.ticker, args.num_quarters)
    elif args.mode == "retrieve":
        run_retrieve(
            agent,
            args.ticker,
            args.question,
            args.num_quarters,
            args.k,
            args.embedding_model,
        )
    else:
        run_answer(
            agent,
            args.ticker,
            args.question,
            args.num_quarters,
            args.k,
            args.embedding_model,
            args.generation_model,
        )


if __name__ == "__main__":
    main()
