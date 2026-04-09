"""
Evaluation utilities for RAG development and testing only.

These functions are NOT imported by any production agent or service.
They are only used by run_rag_evaluation.py during offline benchmarking.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional


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
    generation_provider,
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
    generation_provider,
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
