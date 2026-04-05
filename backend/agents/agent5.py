import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, List, Dict

from .retrieval_pipeline import OpenAIChatGenerationProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EditorInput:
    ticker: str
    draft_report: str
    chart_paths: List[str] = field(default_factory=list)
    price_data: Dict[str, Any] = field(default_factory=dict)
    ratios: Dict[str, Any] = field(default_factory=dict)
    filings: List[Dict[str, Any]] = field(default_factory=list)
    user_requirements: str = ""


@dataclass
class EditorOutput:
    recommendation: str
    confidence: str
    final_report: str
    edit_log: List[str]
    key_drivers: List[str]
    key_risks: List[str]
    success: bool
    error: Optional[str] = None


SYSTEM_PROMPT = """\
You are a senior equity research analyst at a tier-1 investment bank with 20 years of
experience covering technology and energy stocks. You are reviewing a draft investment
note written by a junior analyst before it goes to clients.

This is the final judgment layer in a multi-agent investment workflow:
- upstream agents already retrieved market data, SEC filings, sentiment, ratios, and a draft note
- your job is to produce the final investment stance for the user question such as:
  "Should I buy NVDA?"

Your job is to edit — not rewrite from scratch. Preserve facts and structure where possible,
but sharpen language, fix inconsistencies, and ensure the note is decision-useful.

Always respond with a valid JSON object matching this schema exactly:
{
  "recommendation": "<BUY | HOLD | SELL>",
  "confidence": "<High | Medium | Low>",
  "final_report": "<complete edited note as a single string with \\n for line breaks>",
  "edit_summary": ["<concise description of each edit made>"],
  "quality_score": <integer 1-10 rating the draft BEFORE your edits>,
  "risk_flags": ["<any factual inconsistencies or unsupported claims you found>"],
  "key_drivers": ["<top positive drivers>"],
  "key_risks": ["<top risks>"]
}

Rules:
- recommendation must be exactly one of BUY, HOLD, SELL
- confidence must be exactly one of High, Medium, Low
- do not invent facts not supported by the verified data or draft
- if evidence is mixed or insufficient, prefer HOLD rather than overstating conviction
- do not include markdown fences or any text outside the JSON object
"""


EDIT_PROMPT_TEMPLATE = """\
=== TASK ===
Edit the draft investment note below. Apply the user requirements and fact-check
against the verified data. Return your response as the JSON schema specified.

=== USER REQUIREMENTS ===
{user_requirements}

=== VERIFIED DATA (ground truth — use to fact-check the draft) ===
Ticker: {ticker}
Current price:   ${current_price}
Market cap:      ${market_cap_b}B
P/E (TTM):       {pe_ttm}
Forward P/E:     {forward_pe}
EV/EBITDA:       {ev_ebitda}
EBITDA margin:   {ebitda_margin}%
Net margin:      {net_margin}%

Sentiment trend (last {n_quarters} quarters):
{sentiment_summary}

Chart files attached to this report: {chart_paths}

=== INVESTMENT CONTEXT ===
The user ultimately wants an answer to whether the stock is attractive.
Your final note should therefore:
- make the stance explicit
- summarize the main reasons
- acknowledge the principal risks
- remain professional and balanced

=== EDITING CHECKLIST ===
[ ] 1. RECOMMENDATION
      — State a clear BUY / HOLD / SELL recommendation.
      — Confidence must match the strength of evidence.

[ ] 2. THESIS CLARITY
      — The executive summary must open with the recommendation and one-line rationale.

[ ] 3. FACT CONSISTENCY
      — Every ratio cited must match the verified data above (±0.1 rounding ok).
      — Flag contradictions in risk_flags.

[ ] 4. TONE & AUDIENCE
      — Apply user requirements strictly.
      — Default: formal, direct, no filler, client-ready.

[ ] 5. SENTIMENT NARRATIVE
      — Management tone must match the verified filing sentiment data.

[ ] 6. WHAT TO WATCH
      — Add a "What to Watch" section with 3 bullets if absent.

[ ] 7. CHART REFERENCES
      — If chart_paths is non-empty, reference them clearly in the note.

[ ] 8. DISCLAIMER
      — End with this exact block:
        ---
        DISCLAIMER: This note is produced by an AI research assistant for
        informational purposes only and does not constitute investment advice.
        All data sourced from public filings and market data providers.
        Past performance is not indicative of future results.
        ---

[ ] 9. LENGTH
      — Honour any word-count requirement.
      — Default maximum: 700 words for body (excluding disclaimer).

=== DRAFT NOTE ===
{draft_report}
"""


class SeniorAnalystAgent:
    """
    Agent 5 — Senior Analyst Editor / Final Investment Judgment

    Role in pipeline:
      - Receives the draft note from Agent 4
      - Reviews output from Agents 1-4 implicitly through verified inputs
      - Produces the final recommendation layer for the end user

    Primary outputs:
      - recommendation : BUY / HOLD / SELL
      - confidence     : High / Medium / Low
      - final_report   : readable investor note (text)
      - edit_log       : audit trail of edits / flags
      - key_drivers    : top supporting factors
      - key_risks      : top downside factors
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        generation_model: str = "gpt-4o-mini",
    ):
        self.agent_name = "SeniorAnalystEditor"
        self.generation_provider = OpenAIChatGenerationProvider(
            api_key=openai_api_key,
            model=generation_model,
        )

    def edit_report(
        self,
        ticker: str,
        draft_report: str,
        chart_paths: Optional[List[str]] = None,
        price_data: Optional[Dict[str, Any]] = None,
        ratios: Optional[Dict[str, Any]] = None,
        filings: Optional[List[Dict[str, Any]]] = None,
        user_requirements: str = "",
    ) -> Dict[str, Any]:
        inp = EditorInput(
            ticker=ticker,
            draft_report=draft_report,
            chart_paths=chart_paths or [],
            price_data=price_data or {},
            ratios=ratios or {},
            filings=filings or [],
            user_requirements=user_requirements or "",
        )

        result = self._edit(inp)

        return {
            "agent": self.agent_name,
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "recommendation": result.recommendation,
            "confidence": result.confidence,
            "final_report": result.final_report,
            "edit_log": result.edit_log,
            "key_drivers": result.key_drivers,
            "key_risks": result.key_risks,
            "success": result.success,
            "error": result.error,
        }

    def _edit(self, inp: EditorInput) -> EditorOutput:
        logger.info(f"[SeniorAnalyst] Starting edit for {inp.ticker}")

        pre_warnings = self._pre_validate(inp.draft_report, inp.ratios, inp.price_data)
        if pre_warnings:
            logger.warning(f"[SeniorAnalyst] Pre-validation warnings: {pre_warnings}")

        prompt = self._build_prompt(inp, pre_warnings)
        raw_response = self._call_llm_with_retry(prompt, max_retries=2)

        if raw_response is None:
            fallback_report = self._fallback_report(inp)
            return EditorOutput(
                recommendation="HOLD",
                confidence="Low",
                final_report=fallback_report,
                edit_log=["LLM call failed; returning fallback note."],
                key_drivers=[],
                key_risks=["Model generation failed"],
                success=False,
                error="LLM call failed after retries.",
            )

        parsed = self._parse_response(raw_response)
        if parsed is None:
            fallback_report = self._fallback_report(inp)
            return EditorOutput(
                recommendation="HOLD",
                confidence="Low",
                final_report=fallback_report,
                edit_log=["Could not parse model JSON response; returning fallback note."],
                key_drivers=[],
                key_risks=["Model JSON parse failure"],
                success=False,
                error="JSON parse failure.",
            )

        recommendation = self._normalize_recommendation(parsed.get("recommendation"))
        confidence = self._normalize_confidence(parsed.get("confidence"))
        final_report = self._post_process(parsed.get("final_report", ""), inp, recommendation, confidence)
        edit_log = self._build_edit_log(parsed, pre_warnings)
        key_drivers = self._safe_list(parsed.get("key_drivers"))
        key_risks = self._safe_list(parsed.get("key_risks"))

        logger.info(
            f"[SeniorAnalyst] Edit complete for {inp.ticker} | "
            f"Recommendation={recommendation} | Confidence={confidence}"
        )

        return EditorOutput(
            recommendation=recommendation,
            confidence=confidence,
            final_report=final_report,
            edit_log=edit_log,
            key_drivers=key_drivers,
            key_risks=key_risks,
            success=True,
        )

    def _build_prompt(self, inp: EditorInput, pre_warnings: List[str]) -> str:
        ratios = inp.ratios
        price_data = inp.price_data

        current_price = price_data.get("current_price")
        market_cap = price_data.get("market_cap")
        market_cap_b = round(market_cap / 1e9, 1) if market_cap else "N/A"

        user_req = inp.user_requirements.strip() or (
            "Standard professional tone. Balanced view. Maximum 500 words body. "
            "Suitable for a sophisticated retail investor."
        )

        if pre_warnings:
            user_req += "\n\nPRE-VALIDATION WARNINGS:\n" + "\n".join(
                f"- {w}" for w in pre_warnings
            )

        chart_display = ", ".join(inp.chart_paths) if inp.chart_paths else "None"

        return EDIT_PROMPT_TEMPLATE.format(
            user_requirements=user_req,
            ticker=inp.ticker,
            current_price=current_price if current_price is not None else "N/A",
            market_cap_b=market_cap_b,
            pe_ttm=ratios.get("pe_ttm", ratios.get("pe_ratio", "N/A")),
            forward_pe=ratios.get("forward_pe", "N/A"),
            ev_ebitda=ratios.get("ev_ebitda", "N/A"),
            ebitda_margin=ratios.get("ebitda_margin", "N/A"),
            net_margin=ratios.get("net_margin", "N/A"),
            sentiment_summary=self._build_sentiment_summary(inp.filings),
            n_quarters=len(inp.filings),
            chart_paths=chart_display,
            draft_report=inp.draft_report,
        )

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 2) -> Optional[str]:
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

        for attempt in range(max_retries + 1):
            try:
                response = None

                if hasattr(self.generation_provider, "generate"):
                    response = self.generation_provider.generate(full_prompt)
                elif hasattr(self.generation_provider, "complete"):
                    response = self.generation_provider.complete(full_prompt)
                elif hasattr(self.generation_provider, "invoke"):
                    response = self.generation_provider.invoke(full_prompt)
                else:
                    raise AttributeError(
                        "OpenAIChatGenerationProvider has no supported text-generation method."
                    )

                if isinstance(response, str):
                    return response

                if isinstance(response, dict):
                    return (
                        response.get("text")
                        or response.get("answer")
                        or response.get("content")
                        or json.dumps(response)
                    )

                if hasattr(response, "content"):
                    return response.content

                return str(response)

            except Exception as e:
                logger.warning(f"[SeniorAnalyst] LLM attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    logger.error("[SeniorAnalyst] All retries exhausted.")
                    return None

        return None

    def _build_sentiment_summary(self, filings: List[Dict[str, Any]]) -> str:
        if not filings:
            return "No filing sentiment data available."

        lines = []
        for f in filings:
            quarter = f.get("quarter", "N/A")
            score = f.get("sentiment_score", "N/A")
            tone = f.get("tone", "N/A")
            themes = ", ".join(f.get("key_themes", []))
            lines.append(f"  {quarter}: score={score}, tone={tone}, themes=[{themes}]")

        scores = [
            f.get("sentiment_score")
            for f in filings
            if isinstance(f.get("sentiment_score"), (int, float))
        ]

        if len(scores) >= 2:
            direction = (
                "improving"
                if scores[0] > scores[-1]
                else "deteriorating"
                if scores[0] < scores[-1]
                else "stable"
            )
            lines.append(
                f"  Trend: {direction} (oldest→latest: {scores[-1]:.2f} → {scores[0]:.2f})"
            )

        return "\n".join(lines)

    def _extract_numbers_from_draft(self, draft: str) -> List[float]:
        pattern = r"\$?\d[\d,]*(?:\.\d+)?"
        matches = re.findall(pattern, draft)

        numbers = []
        for m in matches:
            cleaned = m.replace("$", "").replace(",", "").strip()
            if cleaned:
                try:
                    numbers.append(float(cleaned))
                except ValueError:
                    continue
        return numbers

    def _pre_validate(
        self,
        draft: str,
        ratios: Dict[str, Any],
        price_data: Dict[str, Any],
    ) -> List[str]:
        warnings = []

        pe = ratios.get("pe_ttm") or ratios.get("pe_ratio")
        if pe:
            numbers_in_draft = self._extract_numbers_from_draft(draft)
            plausible_pe_range = (pe * 0.8, pe * 1.2)
            mentioned_pes = [
                n for n in numbers_in_draft
                if plausible_pe_range[0] <= n <= plausible_pe_range[1]
            ]
            if not mentioned_pes and pe < 200:
                warnings.append(f"Draft may not correctly cite TTM P/E of {pe}.")

        current_price = price_data.get("current_price")
        if current_price:
            floor = current_price * 0.5
            ceiling = current_price * 2.0
            very_wrong = [
                n for n in self._extract_numbers_from_draft(draft)
                if n > 10 and not (floor <= n <= ceiling) and n < 10000
            ]
            if very_wrong:
                warnings.append(
                    f"Draft contains price-like figures {very_wrong[:3]} that may be inconsistent "
                    f"with current price ${current_price}."
                )

        if not ratios:
            warnings.append("Ratio inputs are missing or empty.")

        if not price_data:
            warnings.append("Price data inputs are missing or empty.")

        return warnings

    def _parse_response(self, raw: str) -> Optional[Dict[str, Any]]:
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.error(f"[SeniorAnalyst] Could not parse JSON response: {raw[:500]}")
        return None

    def _normalize_recommendation(self, recommendation: Any) -> str:
        if not recommendation:
            return "HOLD"
        value = str(recommendation).strip().upper()
        if value in {"BUY", "HOLD", "SELL"}:
            return value
        return "HOLD"

    def _normalize_confidence(self, confidence: Any) -> str:
        if not confidence:
            return "Low"
        value = str(confidence).strip().title()
        if value in {"High", "Medium", "Low"}:
            return value
        return "Low"

    def _safe_list(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return []

    def _post_process(
        self,
        report: str,
        inp: EditorInput,
        recommendation: str,
        confidence: str,
    ) -> str:
        header = (
            f"EQUITY RESEARCH NOTE — {inp.ticker.upper()}\n"
            f"Generated: {datetime.now().strftime('%d %B %Y')}\n"
            f"Recommendation: {recommendation}\n"
            f"Confidence: {confidence}\n"
            f"{'─' * 60}\n\n"
        )

        body = report.strip()

        if not body:
            body = self._fallback_report(inp)

        if "DISCLAIMER:" not in body:
            body += (
                "\n\n---\n"
                "DISCLAIMER: This note is produced by an AI research assistant for\n"
                "informational purposes only and does not constitute investment advice.\n"
                "All data sourced from public filings and market data providers.\n"
                "Past performance is not indicative of future results.\n"
                "---"
            )

        for i, path in enumerate(inp.chart_paths, start=1):
            label = f"Figure {i}: {self._path_to_label(path)}"
            body = body.replace(path, label)

        return header + body

    def _fallback_report(self, inp: EditorInput) -> str:
        return (
            f"Executive Summary\n"
            f"HOLD — insufficiently reliable final model output to upgrade the draft into a "
            f"fully validated recommendation for {inp.ticker.upper()}.\n\n"
            f"Investment View\n"
            f"The upstream pipeline produced a draft note, but the final editorial layer could "
            f"not complete robust validation. Review the draft, source data, and ratio inputs "
            f"before relying on the recommendation.\n\n"
            f"What to Watch\n"
            f"- Next earnings release and management guidance\n"
            f"- Changes in valuation multiples versus peers\n"
            f"- Any material deterioration in filing tone or operating metrics\n\n"
            f"---\n"
            f"DISCLAIMER: This note is produced by an AI research assistant for\n"
            f"informational purposes only and does not constitute investment advice.\n"
            f"All data sourced from public filings and market data providers.\n"
            f"Past performance is not indicative of future results.\n"
            f"---"
        )

    def _build_edit_log(self, parsed: Dict[str, Any], pre_warnings: List[str]) -> List[str]:
        log = []

        quality = parsed.get("quality_score")
        if quality is not None:
            log.append(f"Draft quality score (pre-edit): {quality}/10")

        recommendation = parsed.get("recommendation")
        confidence = parsed.get("confidence")
        if recommendation:
            log.append(f"FINAL RECOMMENDATION: {recommendation}")
        if confidence:
            log.append(f"CONFIDENCE: {confidence}")

        for item in parsed.get("edit_summary", []):
            log.append(f"EDIT: {item}")

        for flag in parsed.get("risk_flags", []):
            log.append(f"RISK FLAG: {flag}")

        for warning in pre_warnings:
            log.append(f"PRE-CHECK: {warning}")

        return log

    def _path_to_label(self, path: str) -> str:
        name = path.split("/")[-1].replace(".png", "").replace("_", " ")
        return name.title()

    def format_for_next_agent(self, data: Dict[str, Any]) -> str:
        if not data.get("success"):
            return (
                f"Error editing report for {data.get('ticker', 'N/A')}: "
                f"{data.get('error')}"
            )

        final_report = data.get("final_report", "")
        edit_log = data.get("edit_log", [])
        recommendation = data.get("recommendation", "N/A")
        confidence = data.get("confidence", "N/A")
        key_drivers = data.get("key_drivers", [])
        key_risks = data.get("key_risks", [])

        formatted = f"""
=== SENIOR ANALYST FINAL OUTPUT ===

TICKER: {data.get('ticker', 'N/A')}
TIMESTAMP: {data.get('timestamp', 'N/A')}
SUCCESS: {data.get('success', False)}
RECOMMENDATION: {recommendation}
CONFIDENCE: {confidence}

KEY DRIVERS:
{self._format_bullets(key_drivers)}

KEY RISKS:
{self._format_bullets(key_risks)}

FINAL REPORT:
{final_report}

EDIT LOG:
{self._format_bullets(edit_log)}
"""
        return formatted.strip()

    def _format_bullets(self, items: List[str]) -> str:
        if not items:
            return "- None"
        return "\n".join(f"- {item}" for item in items)