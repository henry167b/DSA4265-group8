import json
import time
import os

# Increase the time DeepEval waits before giving up on a rate limit
os.environ["DEEPEVAL_TIMEOUT"] = "600"
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "600" # <--- CRITICAL FIX
os.environ["DEEPEVAL_RETRY_MAX_RETRIES"] = "5"

# from langchain_core.messages import ToolMessage
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GPTModel

# nest_asyncio.apply()

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


# Faithfulness metric = no. of truthful claims / total number of claims

# Answer Relevancy metric = no. of relevant statemetns / total number of statements


eval_model = GPTModel(model="gpt-5.4-mini")

faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=eval_model, async_mode=False)
relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=eval_model, async_mode=False)

unsupervised_test_case = LLMTestCase(
        input=task,
        actual_output=unsupervised_report,
        retrieval_context=context # The raw RAG chunks
)

faithfulness_metric.measure(unsupervised_test_case)
unsupervised_faithfulness_score = faithfulness_metric.score
relevancy_metric.measure(unsupervised_test_case)
unsupervised_relevancy_score = relevancy_metric.score

supervised_test_case = LLMTestCase(
    input=task,
    actual_output = supervised_report, 
    # expected_output = unsupervised_report,
    retrieval_context=context
)

faithfulness_metric.measure(supervised_test_case)
supervised_faithfulness_score = faithfulness_metric.score
relevancy_metric.measure(supervised_test_case)
supervised_relevancy_score = relevancy_metric.score

print(f"unsupervised faithfulness score: {unsupervised_faithfulness_score}")
print(f"unsupervised relevancy score: {unsupervised_relevancy_score}")
print(f"supervised faithfulness score: {supervised_faithfulness_score}")
print(f"supervised relevancy score: {supervised_relevancy_score}")

output = f"""
    === Evaluation of unsupervised report FOR {ticker} ===

    Faithfulness Score: {unsupervised_faithfulness_score}

    Relevancy Score: {unsupervised_relevancy_score}

    === Evaluation of supervised report FOR {ticker} ===

    Faithfulness Score: {supervised_faithfulness_score}

    Relevancy Score: {supervised_relevancy_score}

"""

with open(f"evaluation/{ticker}_evaluation.txt", "w") as f:
    f.write(output)


