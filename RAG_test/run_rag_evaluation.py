#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

from RAG_test.common import (
    BENCHMARK_DATASET_PATH,
    CHUNKED_FILINGS_DIR,
    EMBEDDING_CACHE_DIR,
    RESULTS_DIR,
    chunked_filings_path,
    ensure_data_dirs,
    ensure_repo_on_path,
    load_json,
    write_json,
)
from RAG_test.evaluators import (
    answer_matches_oracle,
    evaluate_answer_with_llm,
    evaluate_oracle_match_with_llm,
    optional_bertscore,
)

ensure_repo_on_path()

from backend.agents.retrieval_pipeline import (
    CrossEncoderReranker,
    FilingRetrievalPipeline,
    JsonEmbeddingCache,
    OpenAIChatGenerationProvider,
    OpenAIEmbeddingProvider,
    build_chunk_records_from_prepared_filings,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation from locally cached chunked filings and a local benchmark dataset."
    )
    parser.add_argument(
        "--dataset",
        default=str(BENCHMARK_DATASET_PATH),
        help="Path to the benchmark dataset JSON file.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model used for retrieval.",
    )
    parser.add_argument(
        "--generation-model",
        default="gpt-4o-mini",
        help="Generation model used for RAG answering.",
    )
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-large",
        help="Optional cross-encoder reranker model.",
    )
    parser.add_argument(
        "--enable-reranker",
        action="store_true",
        help="Enable cross-encoder reranking if sentence-transformers is available.",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Use an LLM judge to score relevance, completeness, and faithfulness.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=None,
        help="Top-k chunks to retrieve. Defaults to per-question-type recommended_k if not set.",
    )
    parser.add_argument(
        "--report-file",
        default=str(RESULTS_DIR / "rag_evaluation_report.json"),
        help="Where to save the evaluation report JSON.",
    )
    parser.add_argument(
        "--disable-embedding-cache",
        action="store_true",
        help="Disable persistent embedding caching during evaluation.",
    )
    return parser


def select_filings(cached_payload: Dict, filing_date: str | None) -> List[Dict]:
    filings = cached_payload.get("filings", [])
    if not filing_date:
        if not filings:
            return []
        most_recent = max(filings, key=lambda filing: filing.get("filing_date", ""))
        return [most_recent]
    return [
        filing for filing in filings
        if filing.get("filing_date") == filing_date
    ]


def build_pipeline_for_scope(
    ticker: str,
    filing_date: str | None,
    embedding_provider: OpenAIEmbeddingProvider,
    reranker,
    cache: Dict[Tuple[str, str | None], Tuple[FilingRetrievalPipeline, List[Dict]]],
) -> Tuple[FilingRetrievalPipeline, List[Dict]]:
    key = (ticker, filing_date)
    if key in cache:
        return cache[key]

    payload = load_json(chunked_filings_path(ticker))
    filings = select_filings(payload, filing_date)
    if not filings:
        raise ValueError(
            f"No cached filings available for {ticker} with filing_date={filing_date!r}."
        )
    prepared_payload = {
        "ticker": ticker,
        "filings": filings,
    }
    chunk_records = build_chunk_records_from_prepared_filings(prepared_payload)
    pipeline = FilingRetrievalPipeline(embedding_provider, reranker=reranker)
    pipeline.index_chunks(chunk_records)
    cache[key] = (pipeline, filings)
    return cache[key]


def main() -> None:
    args = build_parser().parse_args()
    ensure_data_dirs()

    dataset = load_json(Path(args.dataset))
    embedding_cache = None
    if not args.disable_embedding_cache:
        embedding_cache = JsonEmbeddingCache(
            EMBEDDING_CACHE_DIR / f"{args.embedding_model.replace('/', '__')}.json"
        )
    embedding_provider = OpenAIEmbeddingProvider(
        model=args.embedding_model,
        cache=embedding_cache,
    )
    generation_provider = OpenAIChatGenerationProvider(model=args.generation_model)
    reranker = None
    if args.enable_reranker:
        try:
            reranker = CrossEncoderReranker(model_name=args.reranker_model)
        except ImportError:
            reranker = None
    pipeline_cache: Dict[Tuple[str, str | None], Tuple[FilingRetrievalPipeline, List[Dict]]] = {}

    results = []
    for example in dataset:
        ticker = example["ticker"]
        filing_date = example.get("filing_date")
        cache_path = chunked_filings_path(ticker)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing chunk cache for {ticker}. Run RAG_test/cache_chunked_filings.py first."
            )

        pipeline, filings = build_pipeline_for_scope(
            ticker=ticker,
            filing_date=filing_date,
            embedding_provider=embedding_provider,
            reranker=reranker,
            cache=pipeline_cache,
        )
        answer_result = pipeline.answer_question(
            question=example["question"],
            generation_provider=generation_provider,
            k=args.k,
        )
        retrieved_chunks = answer_result.get("sources", [])
        answer = answer_result.get("answer")
        resolved_filing_date = filing_date or (
            filings[0].get("filing_date") if filings else None
        )
        oracle_answer_judge = evaluate_oracle_match_with_llm(
            generation_provider,
            example["question"],
            answer,
            example.get("oracle_answer"),
        )
        answer_correct = (
            oracle_answer_judge.get("correct")
            if oracle_answer_judge and oracle_answer_judge.get("correct") is not None
            else answer_matches_oracle(answer, example.get("oracle_answer"))
        )

        llm_judge = None
        if args.llm_judge:
            llm_judge = evaluate_answer_with_llm(
                generation_provider,
                example["question"],
                answer or "",
                retrieved_chunks,
            )

        results.append(
            {
                **example,
                "resolved_filing_date": resolved_filing_date,
                "generated_answer": answer,
                "retrieved_chunks": retrieved_chunks,
                "answer_correct": answer_correct,
                "oracle_answer_judge": oracle_answer_judge,
                "llm_judge": llm_judge,
            }
        )

    answer_rows = [row for row in results if row["answer_correct"] is not None]
    pipeline_bertscores = optional_bertscore(
        [row.get("generated_answer") or "" for row in answer_rows],
        [row.get("oracle_answer") or "" for row in answer_rows],
    )
    if pipeline_bertscores:
        for row, score in zip(answer_rows, pipeline_bertscores):
            row["bertscore_f1"] = score

    judge_rows = [row for row in results if row.get("llm_judge")]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": args.dataset,
        "chunk_cache_dir": str(CHUNKED_FILINGS_DIR),
        "embedding_model": args.embedding_model,
        "generation_model": args.generation_model,
        "reranker_model": args.reranker_model if reranker is not None else None,
        "embedding_cache_file": None if embedding_cache is None else str(embedding_cache.path),
        "k": args.k if args.k is not None else "per_question_type",
        "summary": {
            "num_examples": len(results),
            "answer_accuracy": (
                mean(1.0 if row["answer_correct"] else 0.0 for row in answer_rows)
                if answer_rows else None
            ),
            "pipeline_bertscore_f1": mean(pipeline_bertscores) if pipeline_bertscores else None,
            "llm_judge_overall": (
                mean(
                    row["llm_judge"]["scores"]["overall"]
                    for row in judge_rows
                    if row["llm_judge"]["scores"]["overall"] is not None
                )
                if judge_rows else None
            ),
        },
        "results": results,
    }

    report_destination = Path(args.report_file)
    write_json(report_destination, report)

    print(f"Saved evaluation report to {report_destination}")
    print(f"Examples evaluated: {len(results)}")
    if report["summary"]["answer_accuracy"] is not None:
        print(f"Answer accuracy: {report['summary']['answer_accuracy']:.4f}")


if __name__ == "__main__":
    main()
