#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

from RAG_test.common import (
    BENCHMARK_DATASET_PATH,
    CHUNKED_FILINGS_DIR,
    RESULTS_DIR,
    chunked_filings_path,
    ensure_data_dirs,
    ensure_repo_on_path,
    load_json,
    write_json,
)

ensure_repo_on_path()

from backend.agents.retrieval_pipeline import (
    CrossEncoderReranker,
    FilingRetrievalPipeline,
    OpenAIChatGenerationProvider,
    OpenAIEmbeddingProvider,
    build_chunk_records_from_prepared_filings,
    order_retrieved_chunks_for_generation,
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
        "--disable-reranker",
        action="store_true",
        help="Disable cross-encoder reranking even if sentence-transformers is available.",
    )
    parser.add_argument(
        "--oracle-generator-test",
        action="store_true",
        help="Generate answers again using oracle chunks only.",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Use an LLM judge to score relevance, completeness, and faithfulness.",
    )
    parser.add_argument("-k", type=int, default=5, help="Top-k chunks to retrieve.")
    parser.add_argument(
        "--report-file",
        default=str(RESULTS_DIR / "rag_evaluation_report.json"),
        help="Where to save the evaluation report JSON.",
    )
    return parser


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    lowered = value.lower().strip()
    lowered = lowered.replace("$", " usd ")
    lowered = re.sub(r"[^a-z0-9.%]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def answer_matches_oracle(answer: str | None, oracle_answer: str | None) -> bool | None:
    if not oracle_answer:
        return None
    normalized_answer = normalize_text(answer)
    normalized_oracle = normalize_text(oracle_answer)
    if not normalized_answer or not normalized_oracle:
        return False
    return normalized_oracle in normalized_answer or normalized_answer in normalized_oracle


def evaluate_oracle_match_with_llm(
    generation_provider: OpenAIChatGenerationProvider,
    question: str,
    generated_answer: str | None,
    oracle_answer: str | None,
) -> Optional[Dict]:
    if not generated_answer or not oracle_answer:
        return None

    prompt = f"""You are evaluating whether a generated answer is semantically correct relative to an oracle answer for a 10-Q question.

Question: {question}

Oracle answer:
{oracle_answer}

Generated answer:
{generated_answer}

Judge whether the generated answer should count as correct. Be tolerant of:
- equivalent units such as millions vs billions
- rounded values that preserve the same meaning
- paraphrases and different wording

Be strict about:
- wrong metric
- wrong period
- unsupported claims
- materially wrong numbers or direction of change

Respond in exactly this format:
Correct: YES or NO
Score: X/5
Reason: one short paragraph"""

    response = generation_provider.client.chat.completions.create(
        model=generation_provider.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    correct_match = re.search(r"Correct:\s*(YES|NO)", raw, re.IGNORECASE)
    score_match = re.search(r"Score:\s*(\d)/5", raw)
    reason_match = re.search(r"Reason:\s*(.*)", raw, re.IGNORECASE | re.DOTALL)
    return {
        "raw": raw,
        "correct": (correct_match.group(1).upper() == "YES") if correct_match else None,
        "score": int(score_match.group(1)) if score_match else None,
        "reason": reason_match.group(1).strip() if reason_match else None,
    }


def get_oracle_chunk_ids(example: Dict) -> List[str]:
    oracle_ids = example.get("oracle_chunk_ids")
    if isinstance(oracle_ids, list):
        return [chunk_id for chunk_id in oracle_ids if chunk_id]
    oracle_id = example.get("oracle_chunk_id")
    return [oracle_id] if oracle_id else []


def precision_at_k(retrieved_ids: List[str], oracle_ids: List[str], k: int) -> Optional[float]:
    if not oracle_ids:
        return None
    return sum(1 for chunk_id in retrieved_ids[:k] if chunk_id in oracle_ids) / max(k, 1)


def recall_at_k(retrieved_ids: List[str], oracle_ids: List[str], k: int) -> Optional[float]:
    if not oracle_ids:
        return None
    return sum(1 for chunk_id in retrieved_ids[:k] if chunk_id in oracle_ids) / len(oracle_ids)


def mean_reciprocal_rank(retrieved_ids: List[str], oracle_ids: List[str]) -> Optional[float]:
    if not oracle_ids:
        return None
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in oracle_ids:
            return 1.0 / rank
    return 0.0


def build_chunk_lookup(filings: List[Dict], ticker: str) -> Dict[str, Dict]:
    prepared_payload = {
        "ticker": ticker,
        "filings": filings,
    }
    return {
        record.chunk_id: record.to_dict()
        for record in build_chunk_records_from_prepared_filings(prepared_payload)
    }


def optional_bertscore(candidates: List[str], references: List[str]) -> Optional[List[float]]:
    if not candidates or not references or len(candidates) != len(references):
        return None
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        return None

    _, _, f1_scores = bert_score_fn(candidates, references, lang="en", verbose=False)
    return [float(score) for score in f1_scores]


def evaluate_answer_with_llm(
    generation_provider: OpenAIChatGenerationProvider,
    question: str,
    rag_answer: str,
    sources: List[Dict],
) -> Optional[Dict]:
    if not rag_answer:
        return None

    context = "\n\n---\n\n".join(
        [f"Chunk {index + 1}:\n{chunk.get('text', '')}" for index, chunk in enumerate(sources)]
    )
    prompt = f"""You are evaluating a RAG system's answer to a question about a company's 10-Q filing.

Question: {question}

Retrieved context:
{context}

RAG answer:
{rag_answer}

Score each criterion from 1 to 5:
1. Relevance
2. Completeness
3. Faithfulness

Respond in exactly this format:
Relevance: X/5 - reason
Completeness: X/5 - reason
Faithfulness: X/5 - reason
Overall: X/5"""

    response = generation_provider.client.chat.completions.create(
        model=generation_provider.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    scores = {}
    for label in ["Relevance", "Completeness", "Faithfulness", "Overall"]:
        match = re.search(rf"{label}:\s*(\d)/5", raw)
        scores[label.lower()] = int(match.group(1)) if match else None
    return {"raw": raw, "scores": scores}


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
    embedding_provider = OpenAIEmbeddingProvider(model=args.embedding_model)
    generation_provider = OpenAIChatGenerationProvider(model=args.generation_model)
    reranker = None
    if not args.disable_reranker:
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
        chunk_lookup = build_chunk_lookup(filings, ticker)
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
        retrieved_chunk_ids = [chunk.get("chunk_id") for chunk in retrieved_chunks if chunk.get("chunk_id")]
        oracle_chunk_ids = get_oracle_chunk_ids(example)
        oracle_chunks = [chunk_lookup[chunk_id] for chunk_id in oracle_chunk_ids if chunk_id in chunk_lookup]

        retrieval_hit = None
        if oracle_chunk_ids:
            retrieval_hit = any(chunk_id in oracle_chunk_ids for chunk_id in retrieved_chunk_ids)

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
        oracle_generated_answer = None
        oracle_answer_correct = None
        if args.oracle_generator_test and oracle_chunks:
            ordered_oracle_chunks = order_retrieved_chunks_for_generation(oracle_chunks)
            oracle_generated_answer = generation_provider.generate_answer(
                example["question"],
                ordered_oracle_chunks,
            )
            oracle_answer_correct = answer_matches_oracle(
                oracle_generated_answer,
                example.get("oracle_answer"),
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
                "oracle_chunks": oracle_chunks,
                "retrieval_hit": retrieval_hit,
                "precision_at_k": precision_at_k(retrieved_chunk_ids, oracle_chunk_ids, args.k),
                "recall_at_k": recall_at_k(retrieved_chunk_ids, oracle_chunk_ids, args.k),
                "mrr": mean_reciprocal_rank(retrieved_chunk_ids, oracle_chunk_ids),
                "answer_correct": answer_correct,
                "oracle_answer_judge": oracle_answer_judge,
                "oracle_generated_answer": oracle_generated_answer,
                "oracle_answer_correct": oracle_answer_correct,
                "llm_judge": llm_judge,
            }
        )

    retrieval_rows = [row for row in results if row["retrieval_hit"] is not None]
    answer_rows = [row for row in results if row["answer_correct"] is not None]
    oracle_rows = [row for row in results if row["oracle_answer_correct"] is not None]
    pipeline_bertscores = optional_bertscore(
        [row.get("generated_answer") or "" for row in answer_rows],
        [row.get("oracle_answer") or "" for row in answer_rows],
    )
    oracle_bertscores = optional_bertscore(
        [row.get("oracle_generated_answer") or "" for row in oracle_rows],
        [row.get("oracle_answer") or "" for row in oracle_rows],
    )
    if pipeline_bertscores:
        for row, score in zip(answer_rows, pipeline_bertscores):
            row["bertscore_f1"] = score
    if oracle_bertscores:
        for row, score in zip(oracle_rows, oracle_bertscores):
            row["oracle_bertscore_f1"] = score

    judge_rows = [row for row in results if row.get("llm_judge")]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": args.dataset,
        "chunk_cache_dir": str(CHUNKED_FILINGS_DIR),
        "embedding_model": args.embedding_model,
        "generation_model": args.generation_model,
        "reranker_model": None if args.disable_reranker or reranker is None else args.reranker_model,
        "k": args.k,
        "summary": {
            "num_examples": len(results),
            "retrieval_hit_rate": (
                mean(1.0 if row["retrieval_hit"] else 0.0 for row in retrieval_rows)
                if retrieval_rows else None
            ),
            "precision_at_k": (
                mean(row["precision_at_k"] for row in retrieval_rows if row["precision_at_k"] is not None)
                if retrieval_rows else None
            ),
            "recall_at_k": (
                mean(row["recall_at_k"] for row in retrieval_rows if row["recall_at_k"] is not None)
                if retrieval_rows else None
            ),
            "mrr": (
                mean(row["mrr"] for row in retrieval_rows if row["mrr"] is not None)
                if retrieval_rows else None
            ),
            "answer_accuracy": (
                mean(1.0 if row["answer_correct"] else 0.0 for row in answer_rows)
                if answer_rows else None
            ),
            "oracle_generator_accuracy": (
                mean(1.0 if row["oracle_answer_correct"] else 0.0 for row in oracle_rows)
                if oracle_rows else None
            ),
            "pipeline_bertscore_f1": mean(pipeline_bertscores) if pipeline_bertscores else None,
            "oracle_bertscore_f1": mean(oracle_bertscores) if oracle_bertscores else None,
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
    if report["summary"]["retrieval_hit_rate"] is not None:
        print(f"Retrieval hit rate: {report['summary']['retrieval_hit_rate']:.4f}")
    if report["summary"]["precision_at_k"] is not None:
        print(f"Precision@{args.k}: {report['summary']['precision_at_k']:.4f}")
    if report["summary"]["recall_at_k"] is not None:
        print(f"Recall@{args.k}: {report['summary']['recall_at_k']:.4f}")
    if report["summary"]["mrr"] is not None:
        print(f"MRR: {report['summary']['mrr']:.4f}")
    if report["summary"]["answer_accuracy"] is not None:
        print(f"Answer accuracy: {report['summary']['answer_accuracy']:.4f}")
    if report["summary"]["oracle_generator_accuracy"] is not None:
        print(f"Oracle answer accuracy: {report['summary']['oracle_generator_accuracy']:.4f}")


if __name__ == "__main__":
    main()
