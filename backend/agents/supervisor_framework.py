import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .retrieval_pipeline import OpenAIChatGenerationProvider

logger = logging.getLogger(__name__)


SUPERVISOR_SYSTEM_PROMPT = """\
You are a strict supervisory editor reviewing the final output produced by another AI agent.

Your job is to:
- check whether the answer addresses the user request directly
- remove unsupported claims, obvious speculation, and filler
- improve clarity, structure, and professional tone
- preserve the original meaning unless it is clearly wrong, unsupported, or incomplete

Return only valid JSON matching this schema exactly:
{
  "approved": true,
  "revised_output": "<edited final answer as a single string with \\n for line breaks>",
  "issues_found": ["<specific issue identified during review>"],
  "edit_summary": ["<concise summary of what you changed>"],
  "quality_score": <integer 1-10 rating the original answer before edits>
}

Rules:
- do not add facts not present in the original answer or the supervision notes
- if the original answer is already strong, keep edits light
- if the answer is incomplete, make that limitation explicit instead of fabricating details
- do not include markdown fences or any text outside the JSON object
"""


SUPERVISOR_PROMPT_TEMPLATE = """\
=== REVIEW TARGET ===
Agent name: {agent_name}
Timestamp: {timestamp}

=== USER QUERY ===
{query}

=== SUPERVISION NOTES ===
{supervision_notes}

=== ORIGINAL AGENT OUTPUT ===
{agent_output}
"""


@dataclass
class SupervisorReviewInput:
    agent_name: str
    query: str
    agent_output: str
    supervision_notes: str = ""


@dataclass
class SupervisorReviewOutput:
    approved: bool
    revised_output: str
    issues_found: List[str]
    edit_summary: List[str]
    quality_score: Optional[int]
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OutputSupervisor:
    """Reviews and lightly edits another agent's final answer."""

    def __init__(
        self,
        generation_provider: Optional[Any] = None,
        openai_api_key: Optional[str] = None,
        generation_model: str = "gpt-4o-mini",
    ) -> None:
        self.generation_provider = generation_provider or OpenAIChatGenerationProvider(
            api_key=openai_api_key,
            model=generation_model,
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        )

    def review(
        self,
        agent_name: str,
        query: str,
        agent_output: str,
        supervision_notes: str = "",
    ) -> SupervisorReviewOutput:
        clean_output = (agent_output or "").strip()
        if not clean_output:
            return SupervisorReviewOutput(
                approved=False,
                revised_output="",
                issues_found=["Agent returned an empty response."],
                edit_summary=[],
                quality_score=1,
                success=False,
                error="Empty agent output.",
            )

        review_input = SupervisorReviewInput(
            agent_name=agent_name,
            query=query,
            agent_output=clean_output,
            supervision_notes=(supervision_notes or "").strip() or "No extra supervision notes.",
        )

        prompt = self._build_prompt(review_input)
        raw_response = self._call_provider(prompt)

        if raw_response is None:
            return self._fallback_review(clean_output, "Supervisor model call failed.")

        parsed = self._parse_response(raw_response)
        if parsed is None:
            return self._fallback_review(clean_output, "Supervisor returned invalid JSON.")

        revised_output = str(parsed.get("revised_output") or clean_output).strip()
        issues_found = self._coerce_str_list(parsed.get("issues_found"))
        edit_summary = self._coerce_str_list(parsed.get("edit_summary"))
        quality_score = self._coerce_quality_score(parsed.get("quality_score"))
        approved = bool(parsed.get("approved", not issues_found))

        return SupervisorReviewOutput(
            approved=approved,
            revised_output=revised_output,
            issues_found=issues_found,
            edit_summary=edit_summary,
            quality_score=quality_score,
            success=True,
        )

    def _build_prompt(self, review_input: SupervisorReviewInput) -> str:
        return SUPERVISOR_PROMPT_TEMPLATE.format(
            agent_name=review_input.agent_name,
            timestamp=datetime.now().isoformat(),
            query=review_input.query,
            supervision_notes=review_input.supervision_notes,
            agent_output=review_input.agent_output,
        )

    def _call_provider(self, prompt: str) -> Optional[str]:
        try:
            if hasattr(self.generation_provider, "generate"):
                response = self.generation_provider.generate(prompt)
            elif hasattr(self.generation_provider, "complete"):
                response = self.generation_provider.complete(prompt)
            elif hasattr(self.generation_provider, "invoke"):
                response = self.generation_provider.invoke(prompt)
            else:
                raise AttributeError("Supervisor provider has no supported generation method.")
        except Exception as exc:
            logger.warning("[Supervisor] Provider call failed: %s", exc)
            return None

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

    def _parse_response(self, raw_response: str) -> Optional[Dict[str, Any]]:
        cleaned = re.sub(r"```(?:json)?\s*", "", raw_response).strip().rstrip("`").strip()

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

        logger.warning("[Supervisor] Could not parse response: %s", raw_response[:400])
        return None

    def _fallback_review(self, original_output: str, error: str) -> SupervisorReviewOutput:
        return SupervisorReviewOutput(
            approved=False,
            revised_output=original_output,
            issues_found=[error],
            edit_summary=["Returned the original agent output without supervisor edits."],
            quality_score=None,
            success=False,
            error=error,
        )

    def _coerce_str_list(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _coerce_quality_score(self, value: Any) -> Optional[int]:
        try:
            score = int(value)
        except (TypeError, ValueError):
            return None
        return min(10, max(1, score))


class SupervisedAgentRunner:
    """Wraps an agent callable with a supervisor pass over the final output."""

    def __init__(
        self,
        agent_name: str,
        agent_callable: Callable[[str], Any],
        supervisor: OutputSupervisor,
        response_extractor: Optional[Callable[[Any], str]] = None,
    ) -> None:
        self.agent_name = agent_name
        self.agent_callable = agent_callable
        self.supervisor = supervisor
        self.response_extractor = response_extractor or self._default_response_extractor

    def run(self, query: str, supervision_notes: str = "") -> Dict[str, Any]:
        raw_result = self.agent_callable(query)
        original_output = self.response_extractor(raw_result)
        review = self.supervisor.review(
            agent_name=self.agent_name,
            query=query,
            agent_output=original_output,
            supervision_notes=supervision_notes,
        )

        return {
            "agent": self.agent_name,
            "query": query,
            "raw_result": raw_result,
            "original_output": original_output,
            "final_output": review.revised_output,
            "supervisor": review.to_dict(),
            "success": review.success,
        }

    def _default_response_extractor(self, raw_result: Any) -> str:
        if isinstance(raw_result, str):
            return raw_result

        if isinstance(raw_result, dict):
            messages = raw_result.get("messages")
            if isinstance(messages, list) and messages:
                last_message = messages[-1]
                content = getattr(last_message, "content", None)
                if content is not None:
                    return str(content)
                if isinstance(last_message, dict):
                    return str(last_message.get("content", ""))

            for key in ("final_output", "output", "answer", "content", "text"):
                if key in raw_result and raw_result[key] is not None:
                    return str(raw_result[key])

        return str(raw_result)