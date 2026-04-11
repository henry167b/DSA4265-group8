from types import SimpleNamespace

from backend.agents.supervisor_framework import OutputSupervisor, SupervisedAgentRunner


class FakeGenerationProvider:
    def __init__(self, response):
        self.response = response

    def generate(self, prompt):
        return self.response


class FailingGenerationProvider:
    def generate(self, prompt):
        raise RuntimeError("boom")


def test_output_supervisor_returns_revised_output_from_json_response():
    provider = FakeGenerationProvider(
        """
        {
          "approved": true,
          "revised_output": "Edited answer with tighter wording.",
          "issues_found": ["Original answer was too vague."],
          "edit_summary": ["Tightened the recommendation language."],
          "quality_score": 7
        }
        """
    )
    supervisor = OutputSupervisor(generation_provider=provider)

    result = supervisor.review(
        agent_name="ResearchCoordinator",
        query="Should I buy NVDA?",
        agent_output="Maybe. It depends.",
        supervision_notes="Be direct.",
    )

    assert result.success is True
    assert result.approved is True
    assert result.revised_output == "Edited answer with tighter wording."
    assert result.issues_found == ["Original answer was too vague."]
    assert result.edit_summary == ["Tightened the recommendation language."]
    assert result.quality_score == 7


def test_output_supervisor_falls_back_to_original_output_when_provider_fails():
    supervisor = OutputSupervisor(generation_provider=FailingGenerationProvider())

    result = supervisor.review(
        agent_name="ResearchCoordinator",
        query="Summarize the risks.",
        agent_output="Original answer.",
    )

    assert result.success is False
    assert result.approved is False
    assert result.revised_output == "Original answer."
    assert result.issues_found == ["Supervisor model call failed."]
    assert result.edit_summary == ["Returned the original agent output without supervisor edits."]


def test_supervised_agent_runner_extracts_last_message_content_and_returns_review():
    provider = FakeGenerationProvider(
        """
        {
          "approved": true,
          "revised_output": "Supervisor-polished answer.",
          "issues_found": [],
          "edit_summary": ["Removed filler."],
          "quality_score": 8
        }
        """
    )
    supervisor = OutputSupervisor(generation_provider=provider)

    runner = SupervisedAgentRunner(
        agent_name="ResearchCoordinator",
        agent_callable=lambda query: {
            "messages": [
                SimpleNamespace(content="Intermediate tool output"),
                SimpleNamespace(content="Base answer from agent"),
            ]
        },
        supervisor=supervisor,
    )

    result = runner.run("Give me the investment view.")

    assert result["original_output"] == "Base answer from agent"
    assert result["final_output"] == "Supervisor-polished answer."
    assert result["supervisor"]["edit_summary"] == ["Removed filler."]
    assert result["success"] is True


def test_output_supervisor_rejects_empty_agent_output_without_calling_provider():
    supervisor = OutputSupervisor(generation_provider=FakeGenerationProvider("unused"))

    result = supervisor.review(
        agent_name="ResearchCoordinator",
        query="Anything here?",
        agent_output="   ",
    )

    assert result.success is False
    assert result.approved is False
    assert result.error == "Empty agent output."
    assert result.issues_found == ["Agent returned an empty response."]