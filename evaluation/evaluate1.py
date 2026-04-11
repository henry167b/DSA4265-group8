import json
import time
import os

# Increase the time DeepEval waits before giving up on a rate limit
os.environ["DEEPEVAL_TIMEOUT"] = "600"
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "600" # <--- CRITICAL FIX
os.environ["DEEPEVAL_RETRY_MAX_RETRIES"] = "5"

# from langchain_core.messages import ToolMessage
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GPTModel


ticker = 'aapl'
task = f"Give me a full analysis and flag any risks from the latest 10-Q of {ticker}"

# read context
with open(f'evaluation/{ticker}_context.json', 'r') as f:
    context = json.load(f)

# read unsupervised report 
with open(f'evaluation/{ticker}_analysis_unsupervised_2026-04-11.md', 'r', encoding='utf-8') as f:
    unsupervised_report = f.read()

# read supervised report 
with open(f'evaluation/{ticker}_analysis_supervised_2026-04-11.md', 'r', encoding='utf-8') as f:
    supervised_report = f.read()

eval_model = GPTModel(model="gpt-5.4-mini")

# score 1 for perfection, 0 otherwise  
pairwise_judge = GEval(
    name="Supervisor Quality Boost",
    model=eval_model,
    criteria = "Determine if the actual_output (supervised) is more professional, "
                "better formatted, and more concise than the expected_output (unsupervised).",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, 
                        LLMTestCaseParams.EXPECTED_OUTPUT],
    async_mode=False
)

supervised_test_case = LLMTestCase(
    input=task,
    actual_output = supervised_report, 
    expected_output = unsupervised_report,
    retrieval_context=context
)


pairwise_judge.measure(supervised_test_case)

print(f"pairwise judge score: {pairwise_judge.score}")
print("========================================================")
print(f"Reasoning: {pairwise_judge.reason}")

output = f"""
    

    === Evaluation of supervised report FOR {ticker} ===

    Pairwise Score: {pairwise_judge.score}

    Reasoning: {pairwise_judge.reason}

"""

with open(f"evaluation/{ticker}_evaluation1.txt", "w") as f:
    f.write(output)


