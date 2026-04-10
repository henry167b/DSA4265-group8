import re
from typing import Dict, List, Optional, Protocol, Tuple


RISK_SECTION = "RISK"
MDA_SECTION = "MDA"
UNKNOWN_SECTION = "UNKNOWN"

TOP_PER_SECTION = 2
MIN_CHUNK_TEXT_LEN = 120

LABEL_PATTERN = re.compile(r"\b(bullish|neutral|bearish)\b", re.IGNORECASE)

MDA_POSITIVE_SIGNAL_RE = re.compile(
    r"\b(revenue|gross margin|operating income|net income|earnings|guidance|outlook|"
    r"demand|data center|gaming|automotive|cash flow|results of operations|"
    r"financial condition|liquidity|capital resources)\b",
    re.IGNORECASE,
)

BOILERPLATE_RE = re.compile(
    r"\b(forward-looking statements|safe harbor|controls and procedures|"
    r"disclosure controls|legal proceedings|quantitative and qualitative disclosures"
    r" about market risk)\b",
    re.IGNORECASE,
)


class GenerationProvider(Protocol):
    def generate_answer(self, question: str, retrieved_chunks: List[Dict]) -> str:
        ...


def build_classification_prompt(ticker: str, filing: Dict) -> str:
    return (
        f"You are a financial analyst.\n\n"
        f"Task: Based ONLY on the retrieved 10-Q excerpts for {ticker} "
        f"(quarter: {filing.get('quarter', 'N/A')}, filing date: {filing.get('filing_date', 'N/A')}), "
        f"classify the stock outlook.\n\n"
        f"Return ONLY one word from this list:\n"
        f"Bullish\n"
        f"Neutral\n"
        f"Bearish\n\n"
        f"Do not explain your answer."
    )


def normalize_label(text: str) -> str:
    if not text:
        return "Neutral"

    match = LABEL_PATTERN.search(text.strip())
    if not match:
        return "Neutral"

    return match.group(1).capitalize()


def _canonicalize_section_name(section_name: str) -> str:
    if not section_name:
        return ""
    normalized = section_name.strip().lower()
    normalized = normalized.replace("\u2019", "'").replace("`", "'")
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _normalize_section_name(chunk: Dict) -> str:
    return (
        chunk.get("section_name")
        or chunk.get("section")
        or chunk.get("metadata", {}).get("section_name")
        or ""
    ).strip()


def _section_bucket_from_name(section_name: str) -> str:
    s = _canonicalize_section_name(section_name)
    if "risk factor" in s or "risk factors" in s or "item 1a" in s:
        return RISK_SECTION
    if (
        "management's discussion and analysis" in s
        or "managements discussion and analysis" in s
        or "management discussion and analysis" in s
        or "md&a" in s
        or "md and a" in s
        or "item 2" in s
    ):
        return MDA_SECTION
    return UNKNOWN_SECTION


def _chunk_score(chunk: Dict) -> float:
    for key in ("score", "rerank_score", "hybrid_score", "dense_score", "sparse_score"):
        value = chunk.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _chunk_text(chunk: Dict) -> str:
    text = chunk.get("text")
    if isinstance(text, str):
        return text
    return ""


def _section_adjusted_score_details(chunk: Dict, bucket: str) -> Tuple[float, List[str]]:
    text = _chunk_text(chunk)
    if not text:
        return float("-inf"), ["empty_text_excluded"]

    base_score = _chunk_score(chunk)
    score = base_score
    reasons: List[str] = [f"base_score={base_score:.4f}"]

    if len(text) < MIN_CHUNK_TEXT_LEN:
        score -= 1.5
        reasons.append("short_text_penalty=-1.5")

    if BOILERPLATE_RE.search(text):
        score -= 2.5
        reasons.append("boilerplate_penalty=-2.5")

    if bucket == MDA_SECTION and MDA_POSITIVE_SIGNAL_RE.search(text):
        score += 0.8
        reasons.append("mda_signal_boost=+0.8")

    reasons.append(f"adjusted_score={score:.4f}")
    return score, reasons


def _annotate_chunk_debug(
    chunk: Dict,
    bucket: str,
    section_rank: int,
    score: float,
    reasons: List[str],
) -> Dict:
    annotated = dict(chunk)
    annotated["selection_debug"] = {
        "section_bucket": bucket,
        "section_rank": section_rank,
        "adjusted_score": round(score, 4),
        "reasons": reasons,
    }
    return annotated


def select_top_chunks_by_section(
    chunks: List[Dict],
    top_per_section: int = TOP_PER_SECTION,
) -> Tuple[List[Dict], List[str]]:
    risk_candidates: List[Dict] = []
    mda_candidates: List[Dict] = []

    for chunk in chunks:
        section_name = _normalize_section_name(chunk)
        bucket = _section_bucket_from_name(section_name)
        if bucket == RISK_SECTION:
            risk_candidates.append(chunk)
        elif bucket == MDA_SECTION:
            mda_candidates.append(chunk)

    risk_scored = [(chunk, *_section_adjusted_score_details(chunk, RISK_SECTION)) for chunk in risk_candidates]
    mda_scored = [(chunk, *_section_adjusted_score_details(chunk, MDA_SECTION)) for chunk in mda_candidates]

    risk_ranked = sorted(risk_scored, key=lambda item: item[1], reverse=True)
    mda_ranked = sorted(mda_scored, key=lambda item: item[1], reverse=True)

    selected_risk = [
        _annotate_chunk_debug(chunk, RISK_SECTION, idx, score, reasons)
        for idx, (chunk, score, reasons) in enumerate(risk_ranked[:top_per_section], start=1)
    ]
    selected_mda = [
        _annotate_chunk_debug(chunk, MDA_SECTION, idx, score, reasons)
        for idx, (chunk, score, reasons) in enumerate(mda_ranked[:top_per_section], start=1)
    ]

    selected_sections: List[str] = []
    if selected_risk:
        selected_sections.append(RISK_SECTION)
    if selected_mda:
        selected_sections.append(MDA_SECTION)

    return selected_risk + selected_mda, selected_sections


class QuarterlySentimentTool:
    """
    Modular sentiment tool.

    This class does not fetch filings, prepare chunks, or retrieve context itself.
    It only consumes chunks and filing metadata that are passed in.
    """

    def __init__(
        self,
        generation_provider: Optional[GenerationProvider] = None,
        top_per_section: int = TOP_PER_SECTION,
    ) -> None:
        self.generation_provider = generation_provider
        self.top_per_section = top_per_section

    def analyze_single_filing(
        self,
        ticker: str,
        filing: Dict,
        candidate_chunks: List[Dict],
        actual_result: Optional[Dict] = None,
        chunk_count: Optional[int] = None,
        raw_model_answer: Optional[str] = None,
        generation_provider: Optional[GenerationProvider] = None,
    ) -> Dict:
        """
        Analyze one filing from externally supplied candidate chunks.

        Output schema mirrors QuarterlySentimentAnalyzer.analyze_single_filing.
        """
        selected_chunks, selected_sections = select_top_chunks_by_section(
            candidate_chunks,
            top_per_section=self.top_per_section,
        )

        provider = generation_provider or self.generation_provider
        model_error = None
        raw_answer = raw_model_answer or ""

        if not raw_answer:
            if provider is None:
                model_error = "No generation provider or raw_model_answer supplied."
            elif not selected_chunks:
                model_error = "No relevant chunks selected from input candidate_chunks."
            else:
                prompt = build_classification_prompt(ticker, filing)
                raw_answer = provider.generate_answer(
                    question=prompt,
                    retrieved_chunks=selected_chunks,
                )

        predicted_label = normalize_label(raw_answer)

        actual_payload = actual_result or {}
        default_price_error = None if actual_result is not None else "Actual result not supplied."
        resolved_chunk_count = int(chunk_count) if chunk_count is not None else len(candidate_chunks)

        return {
            "ticker": ticker.upper(),
            "quarter": filing.get("quarter", "N/A"),
            "filing_date": filing.get("filing_date", "N/A"),
            "accession_number": filing.get("accession_number", "N/A"),
            "form_type": filing.get("form_type", "10-Q"),
            "predicted_label": predicted_label,
            "predicted_raw_answer": raw_answer,
            "realized_next_quarter_label": actual_payload.get("actual_label", "Unknown"),
            "actual_label": actual_payload.get("actual_label", "Unknown"),
            "realized_next_quarter_pct_change": actual_payload.get("actual_return_percent"),
            "actual_return_percent": actual_payload.get("actual_return_percent"),
            "realized_next_quarter_start_price": actual_payload.get("actual_price_start"),
            "actual_price_start": actual_payload.get("actual_price_start"),
            "realized_next_quarter_end_price": actual_payload.get("actual_price_end"),
            "actual_price_end": actual_payload.get("actual_price_end"),
            "chunks_used": len(selected_chunks),
            "chunk_count": resolved_chunk_count,
            "selected_sections": selected_sections,
            "retrieved_chunks": selected_chunks,
            "model_error": model_error,
            "price_error": actual_payload.get("error", default_price_error),
        }

    def analyze_chunks(
        self,
        ticker: str,
        filing: Dict,
        candidate_chunks: List[Dict],
        actual_result: Optional[Dict] = None,
        chunk_count: Optional[int] = None,
        raw_model_answer: Optional[str] = None,
        generation_provider: Optional[GenerationProvider] = None,
    ) -> Dict:
        """Backward-compatible alias for analyze_single_filing."""
        return self.analyze_single_filing(
            ticker=ticker,
            filing=filing,
            candidate_chunks=candidate_chunks,
            actual_result=actual_result,
            chunk_count=chunk_count,
            raw_model_answer=raw_model_answer,
            generation_provider=generation_provider,
        )
