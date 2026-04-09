from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from .financial_data_agent import YahooFinanceAgent
from .filing_rag_service import FilingRAGService
from .quarterly_sentiment import SentimentEmbeddingProvider
from .supervisor_framework import OutputSupervisor, SupervisedAgentRunner

from datetime import datetime
import re
import logging

logging.basicConfig(
    filename="evaluation/debug.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)



# Instantiate your existing agent
yf_agent = YahooFinanceAgent(user_email="e0968923@u.nus.edu")
filing_rag_service = FilingRAGService()
sentiment_agent = SentimentEmbeddingProvider()



# ── Tool 1: Stock data + formatted output ──────────────────────────────────
class StockDataInput(BaseModel):
    ticker: str
    days_back: int = 30


@tool(args_schema=StockDataInput)
def get_stock_data(ticker: str, days_back: int = 30) -> str:
    """Fetch current price, key metrics, analyst recommendations, and recent news for a stock ticker."""
    data = yf_agent.get_stock_data(ticker, days_back)
    return yf_agent.format_for_next_agent(data)

class CompanyOverviewInput(BaseModel):
    ticker: str

@tool(args_schema=CompanyOverviewInput)
def get_company_overview(ticker: str) -> str:
    """Fetch a brief company overview for the given ticker."""
    data = yf_agent.get_company_overview(ticker)
    return yf_agent.format_company_info(data)

class FinancialRatiosInput(BaseModel):
    ticker: str

@tool(args_schema=FinancialRatiosInput)
def get_financial_ratios(ticker: str) -> str:
    """Fetch key financial ratios for the given ticker."""
    data = yf_agent.get_financial_ratios(ticker)
    return yf_agent.format_ratios_for_next_agent(data)


class CompleteAnalysisInput(BaseModel):
    ticker: str
@tool(args_schema=CompleteAnalysisInput)
def get_complete_analysis_data(ticker: str) -> str:
    """Run a full analysis combining Yahoo Finance market data + SEC filings + XBRL financial facts."""
    data = yf_agent.get_complete_analysis_data(ticker)

    yf_formatted = yf_agent.format_for_next_agent(data)
    company_formatted = yf_agent.format_company_info(data.get("company_info", {}))
    sec_formatted = yf_agent.format_sec_data_for_next_agent(data.get("sec_filings", {}))
    facts_formatted = yf_agent.format_financial_facts_for_next_agent(data.get("financial_facts", {}))

    return f"{company_formatted}\n\n{yf_formatted}\n\n{sec_formatted}\n\n{facts_formatted}"


# ── Tool 2: SEC 10-Q filings ───────────────────────────────────────────────
class FilingsInput(BaseModel):
    ticker: str
    num_quarters: int = 4


@tool(args_schema=FilingsInput)
def get_sec_filings(ticker: str, num_quarters: int = 4) -> str:
    """Fetch recent SEC 10-Q filings metadata for a company."""
    data = yf_agent.get_recent_10q_filings(ticker, num_quarters=num_quarters)
    return yf_agent.format_sec_data_for_next_agent(data)


# ── Tool 3: XBRL financial facts ───────────────────────────────────────────
class FinancialFactsInput(BaseModel):
    ticker: str


@tool(args_schema=FinancialFactsInput)
def get_financial_facts(ticker: str) -> str:
    """Fetch structured XBRL financial data from SEC for the last 4 quarters."""
    data = yf_agent.get_financial_facts(ticker)
    return yf_agent.format_financial_facts_for_next_agent(data)


# ── Tool 4: RAG over 10-Q filings ──────────────────────────────────────────
class QuestionInput(BaseModel):
    ticker: str
    question: str

@tool(args_schema=QuestionInput)
def answer_10q_question(ticker: str, question: str) -> str:
    """Answer a question using recent 10-Q filings for the specified ticker."""
    try:
        rag = FilingRAGService()

        filings_payload = yf_agent.get_recent_10q_filings(
            ticker,
            num_quarters=1,
            include_document_html=True,
        )

        rag.index_from_prepared_filings(filings_payload)
        result = rag.answer(question)

        return result.get("answer", "Could not generate answer.")

    except Exception as e:
        print("\n=== DEBUG: exception ===")
        print(str(e))
        return f"Failed to answer 10-Q question for {ticker}: {str(e)}"

# ── Tool 5: Sentiment analysis over quarterly reports ──────────────────────────────────────────
class SentimentInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. AAPL")
    num_quarters: int = Field(
        4,
        description="Number of most recent quarterly reports to analyze"
    )


@tool(args_schema=SentimentInput)
def analyze_quarterly_sentiment(ticker: str, num_quarters: int = 4) -> str:
    """
    Analyze the sentiment of the most recent quarterly reports for a stock
    and return a concise summary for downstream agents.
    """
    try:
        result = SentimentEmbeddingProvider.analyze_quarterly_reports(
            ticker=ticker,
            num_quarters=num_quarters
        )
        return SentimentEmbeddingProvider.format_for_next_agent(result)
    except Exception as e:
        return f"Failed to analyze quarterly sentiment for {ticker}: {str(e)}"

# tool for agent 
tools = [
    get_stock_data,
    get_company_overview,
    get_sec_filings,
    get_financial_facts,
    get_complete_analysis_data,
    answer_10q_question,
    analyze_quarterly_sentiment
]

def build_agent(model: str = "gpt-5.4"):
    llm = ChatOpenAI(
        model=model,
        temperature=0
    )
    system_prompt = ("""
        You are a professional equity research analyst producing institutional-quality investment research.

        Your job is to answer the user's request using the available tools when needed, and then write a clear, rigorous, and professional final report.

        OBJECTIVE
        - Deliver a direct, well-structured, evidence-based investment analysis.
        - Use relevant tool outputs to support the analysis.
        - Prioritize factual accuracy, clarity, and financial relevance.
        
        TOOL USAGE GUIDELINES
        - Use available tools selectively to gather relevant information before forming conclusions.
        - Use stock data tools for price performance, valuation metrics, and market data.
        - Use filings retrieval tools for financials, disclosures, and risk factors from 10-Qs.
        - Use quarterly sentiment analysis when assessing management tone, forward guidance, or qualitative outlook from 10-Q filings.
        - Do not call tools unnecessarily; only use them when they add material value to the analysis.
        - Integrate tool outputs into the narrative rather than listing them mechanically.
                     
        RAG USAGE GUIDELINES
        - Use the 10-Q RAG tool for detailed, text-based questions about filings (e.g., risks, revenue drivers, liquidity).
        - The RAG tool operates on one company at a time and retrieves information from recent filings.
        - Do NOT use the RAG tool for cross-company comparisons.
        - Prefer other tools (financial ratios, stock data) for quantitative or structured data.
        - Avoid repeated RAG calls unless the question requires deeper analysis of filings.

        STYLE
        - Write in a formal, analytical, professional tone.
        - Be concise but complete.
        - Use plain, precise language.
        - Sound like an equity research analyst, not a general chatbot.
        - Use light emphasis where appropriate (e.g., bold key metrics or conclusions)
        - Prioritize readability and flow over rigid templating.

        OUTPUT RULES
        - Directly answer the user's request.
        - Organize the response into clear sections where appropriate.
        - When relevant, cover business overview, financial performance, valuation, risks, and conclusion.
        - If data is missing or unavailable, state that explicitly instead of guessing.
        - Do not make unsupported claims.
        - Do not overstate conviction.
        - Use bold formatting for all section headers using this style: **Section Name**
                     
        Preferred structure when applicable:
        1. **Investment Summary**
        2. **Business and Recent Developments**
        3. **Financial Performance**
        4. **Valuation**
        5. **Key Risks**
        6. **Conclusion**

        Do not include:
        - follow-up suggestions
        - offers for more analysis
        - questions to the user
        - conversational filler
        - chatbot-style language
        - phrases such as "if you want", "I can also", "let me know", or "would you like"

        Do not mention tools or internal workflow.

        End immediately after the conclusion.

        STRICT PROHIBITIONS
        - Do NOT include follow-up suggestions.
        - Do NOT offer additional help.
        - Do NOT ask the user questions.
        - Do NOT include conversational filler.
        - Do NOT include phrases such as:
        - "if you want"
        - "I can also"
        - "let me know"
        - "would you like"
        - "happy to help"
        - "feel free to ask"
        - Do NOT mention tools, prompts, or internal reasoning.
        - Do NOT behave like a chatbot or assistant.

        ENDING RULE
        - End the response cleanly after the final analytical paragraph or conclusion.
        - Do not append any extra sentence after the conclusion.

        QUALITY STANDARD
        - The final output should read like a polished equity research note prepared for an informed investor.
        """)
    
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt)


def _invoke_agent(query: str, model: str = "gpt-5.4"):
    agent = build_agent(model=model)
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": query}
        ]
    })

    logger.info("AGENT RESULT: %s", result)

    if isinstance(result, dict):
        messages = result.get("messages", [])
        for i, msg in enumerate(messages):
            logger.info("MESSAGE %d TYPE=%s CONTENT=%s", i, msg.__class__.__name__, msg)

    return result


def _extract_agent_output(result) -> str:
    messages = result.get("messages", []) if isinstance(result, dict) else []
    if messages:
        last_message = messages[-1]
        content = getattr(last_message, "content", None)
        if content is not None:
            return str(content)
        if isinstance(last_message, dict):
            return str(last_message.get("content", ""))
    return str(result) 

# add trace tools
def extract_tool_trace(raw_result) -> str:
    if not isinstance(raw_result, dict):
        return ""

    messages = raw_result.get("messages", [])
    if not isinstance(messages, list):
        return ""

    lines = []

    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name", "unknown_tool")
                args = tc.get("args", {})
                lines.append(f"TOOL CALL: {name} | ARGS: {args}")

        if msg.__class__.__name__ == "ToolMessage":
            content = getattr(msg, "content", "")
            lines.append(f"TOOL OUTPUT: {content}")

        if isinstance(msg, dict):
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("name", "unknown_tool")
                    args = tc.get("args", {})
                    lines.append(f"TOOL CALL: {name} | ARGS: {args}")

            if msg.get("type") == "tool":
                lines.append(f"TOOL OUTPUT: {msg.get('content', '')}")

    return "\n\n".join(lines)



# ──────────────────────────────────────────────────────────────────────────────
# build supervisor for agent output review and revision

def build_supervised_runner(
    agent_model: str = "gpt-5.4",
    supervisor_model: str = "gpt-5.4",
):
    supervisor = OutputSupervisor(generation_model=supervisor_model)
    return SupervisedAgentRunner(
        agent_name="ResearchCoordinator",
        agent_callable=lambda query: _invoke_agent(query, model=agent_model),
        supervisor=supervisor,
        response_extractor=_extract_agent_output,
    )


def run_agent(
    query: str,
    supervise: bool = False,
    supervision_notes: str = """
    Review the draft as a senior equity research analyst.

    Revise it so that it:
    - directly answers the user's task
    - removes unsupported or weakly supported claims
    - ensures the conclusion is consistent with the evidence presented
    - sharpens the relationship between financial performance, valuation, and risks
    - preserves uncertainty where evidence is incomplete
    - uses a professional institutional tone
    - contains no chatbot-style language or follow-up offers
    - ends cleanly after the conclusion
    """
):
    if not supervise:
        return _invoke_agent(query)

    runner = build_supervised_runner()
    return runner.run(query, supervision_notes=supervision_notes)


def run_agent_for_ticker(ticker: str, task: str, supervise: bool = False):
    query = f"""
        The stock ticker is {ticker}.
        Use {ticker} for all tool calls.
        Do not substitute another ticker.
        Task: {task}
        """
    return run_agent(query, supervise=supervise)



# ──────────────────────────────────────────────────────────────────────────────

def make_report_filename(ticker: str, report_type: str = "analysis", supervise: bool = False) -> str:
    safe_ticker = re.sub(r"[^A-Za-z0-9_-]", "", ticker.upper())
    date_str = datetime.now().strftime("%Y-%m-%d")

    mode = "supervised" if supervise else "unsupervised"

    return f"{safe_ticker}_{report_type}_{mode}_{date_str}.md"
    

# Example usage:

if __name__ == "__main__":
    ticker = "AAPL"
    task = "Give me a full analysis and flag any risks from the latest 10-Q"
    supervise = True

    result = run_agent_for_ticker(ticker, task, supervise=supervise)

    supervisor_info = {}

    if supervise and isinstance(result, dict):
        output = result.get("final_output", "")
        supervisor_info = result.get("supervisor", {})
        raw_for_trace = result.get("raw_result", {})
    else:
        output = _extract_agent_output(result)
        raw_for_trace = result

    tool_trace = extract_tool_trace(raw_for_trace)

    logger.info("===== TOOL TRACE =====\n%s", tool_trace)
    print("\n===== TOOL TRACE =====\n")
    print(tool_trace)

    file_path = f"evaluation/{make_report_filename(ticker, report_type='analysis', supervise=supervise)}"
    trace_path = f"evaluation/{ticker}_tool_trace.txt"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(output)

    with open(trace_path, "w", encoding="utf-8") as f:
        f.write(tool_trace)

    print("\n===== RAW RESULT TYPE =====\n")
    print(type(result))

    print("\n===== FINAL OUTPUT =====\n")
    print(repr(output))

    print("\n===== SUPERVISOR REVIEW =====\n")
    print(supervisor_info)

    print(f"\nSaved report to: {file_path}")
    print(f"Saved tool trace to: {trace_path}")

    # to run:
    # PYTHONPATH=. /usr/local/bin/python3 -m backend.agents.test_agent_super_final
    