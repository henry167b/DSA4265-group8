import json
from typing import TypedDict, List, Dict, Any, Optional

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from .retrieval_pipeline import OpenAIChatGenerationProvider

from .financial_data_agent import YahooFinanceAgent
from .filing_rag_service import FilingRAGService
from .quarterly_sentiment_tool import GenerationProvider, QuarterlySentimentTool
from .supervisor_framework_langgraph import OutputSupervisor, SupervisedAgentRunner

from datetime import datetime
import re
import logging

logging.basicConfig(
    filename="evaluation/debug.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)



# Instantiate existing agent
yf_agent = YahooFinanceAgent(user_email="e0968923@u.nus.edu")
filing_rag_service = FilingRAGService()
sentiment_agent = QuarterlySentimentTool()



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
    data = yf_agent.get_financial_facts(ticker)
    return yf_agent.format_financial_facts_for_next_agent(data)


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
    Analyze the sentiment of recent quarterly reports for a stock
    and return a concise summary for downstream agents.
    """
    try:
        filings_payload = yf_agent.get_recent_10q_filings(
            ticker,
            num_quarters=num_quarters,
            include_document_html=True,
        )

        if not filings_payload.get("success"):
            return (
                f"Quarterly sentiment analysis failed for {ticker}: "
                f"{filings_payload.get('error', 'Unknown error')}"
            )

        filings = filings_payload.get("filings", [])
        if not filings:
            return f"Quarterly sentiment analysis failed for {ticker}: No filings found."

        generation_provider = OpenAIChatGenerationProvider(
            model="gpt-4o-mini"
        )

        sentiment_tool = QuarterlySentimentTool(
            generation_provider=generation_provider
        )

        results = []

        for filing in filings:
            # Index each filing separately to avoid mixing chunks across quarters
            single_filing_payload = {
                "ticker": ticker,
                "filings": [filing],
            }

            rag = FilingRAGService()
            rag.index_from_prepared_filings(single_filing_payload)

            retrieval = rag.search(
                "risk factors management's discussion and analysis "
                "results of operations liquidity outlook",
                k=12,
            )

            candidate_chunks = (
                retrieval.get("results", [])
                if retrieval.get("success")
                else []
            )

            result = sentiment_tool.analyze_single_filing(
                ticker=ticker,
                filing=filing,
                candidate_chunks=candidate_chunks,
                chunk_count=len(candidate_chunks),
            )
            results.append(result)

        if not results:
            return (
                f"Quarterly sentiment analysis failed for {ticker}: "
                f"No sentiment results generated."
            )

        lines = [f"Quarterly sentiment analysis for {ticker.upper()}:"]

        for q in results:
            model_error = q.get("model_error")
            if model_error:
                lines.append(
                    f"- {q.get('quarter', 'N/A')} ({q.get('filing_date', 'N/A')}): "
                    f"failed={model_error}"
                )
            else:
                lines.append(
                    f"- {q.get('quarter', 'N/A')} ({q.get('filing_date', 'N/A')}): "
                    f"predicted={q.get('predicted_label', 'Unknown')}, "
                    f"sections={', '.join(q.get('selected_sections', [])) or 'None'}, "
                    f"chunks_used={q.get('chunks_used', 0)}"
                )

        return "\n".join(lines)

    except Exception as e:
        return f"Failed to analyze quarterly sentiment for {ticker}: {str(e)}"

#######################################################################################################

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





################################################################################################################
# build supervisor for agent output review and revision using langgraph framework

# define class to hold agent state during supervision
class ResearchState(TypedDict, total=False):
    ticker: str
    task: str
    query: str

    draft: str
    final_output: str

    raw_result: Dict[str, Any]
    tool_trace: str

    supervisor_pass: bool
    supervisor_info: Dict[str, Any]
    issues: List[Dict[str, Any]]
    revision_notes: str

    evidence: List[str]

    revision_count: int
    max_revisions: int

########################################################################################################

def debug_state_snapshot(node_name: str, state: ResearchState) -> None:
    print(f"\n===== NODE: {node_name} =====")
    print("revision_count:", state.get("revision_count"))
    print("supervisor_pass:", state.get("supervisor_pass"))
    print("issues:", state.get("issues"))
    print("evidence_count:", len(state.get("evidence", [])))
    print("supervisor_info:", state.get("supervisor_info"))
    print("tool_trace_preview:", (state.get("tool_trace", "")[:500] + "...") if state.get("tool_trace") else "None")

##################################################################################################################################



def prepare_query(state: ResearchState) -> dict:
    ticker = state["ticker"]
    task = state["task"]

    query = f"""
    The stock ticker is {ticker}.
    Use {ticker} for all tool calls.
    Do not substitute another ticker.
    Task: {task}
    """
    return {
        "query": query,
        "revision_count": state.get("revision_count", 0),
        "max_revisions": state.get("max_revisions", 2),
        "evidence": state.get("evidence", []),
    }


def draft_report(state: ResearchState) -> dict:
    raw_result = _invoke_agent(state["query"])
    draft = _extract_agent_output(raw_result)
    tool_trace = extract_tool_trace(raw_result)

    updates = {
        "raw_result": raw_result,
        "draft": draft,
        "tool_trace": tool_trace,
    }

    debug_state_snapshot("draft_report (after)", {**state, **updates})

    return updates


def routing_supervisor(state: ResearchState) -> dict:
    debug_state_snapshot("routing_supervisor (before)", state)

    llm = ChatOpenAI(model="gpt-5.4", temperature=0)

    draft = state.get("draft", "")
    query = state.get("query", "")
    tool_trace = state.get("tool_trace", "")
    evidence = "\n\n".join(state.get("evidence", [])) or "No additional evidence collected yet."
    revision_count = state.get("revision_count", 0)

    prompt = f"""
        You are a supervisory controller for a financial analysis agent.

        Your job is NOT to rewrite the draft.
        Your job is to determine whether the draft is sufficiently grounded in evidence and complete for the user's request.

        Return ONLY valid JSON in exactly this format:

        {{
        "pass": true/false,
        "issues": [
            {{
            "issue_type": "missing_risk_analysis | missing_financial_support | missing_sentiment_context | unsupported_claims | weak_grounding | incomplete_task_response",
            "target_agent": "filing | financial | sentiment",
            "action": "retrieve_risk_factors | retrieve_financial_facts | retrieve_sentiment",
            "reason": "short explanation"
            }}
        ],
        "revision_notes": "brief instruction for the next revision"
        }}

        USER QUERY:
        {query}

        REVISION COUNT:
        {revision_count}

        TOOL TRACE:
        {tool_trace or "No tool trace available."}

        ADDITIONAL EVIDENCE:
        {evidence}

        CURRENT DRAFT:
        {draft}
        """

    try:
        response = llm.invoke(prompt)
        raw_content = response.content if hasattr(response, "content") else str(response)
        parsed = json.loads(raw_content)

        updates = {
            "supervisor_pass": parsed.get("pass", False),
            "issues": parsed.get("issues", []),
            "revision_notes": parsed.get("revision_notes", ""),
            "supervisor_info": parsed,
        }

    except Exception as e:
        updates = {
            "supervisor_pass": False,
            "issues": [
                {
                    "issue_type": "weak_grounding",
                    "target_agent": "filing",
                    "action": "retrieve_risk_factors",
                    "reason": f"Routing supervisor failed: {str(e)}",
                }
            ],
            "revision_notes": "Strengthen filing-based grounding and re-check completeness.",
            "supervisor_info": {
                "pass": False,
                "issues": [
                    {
                        "issue_type": "weak_grounding",
                        "target_agent": "filing",
                        "action": "retrieve_risk_factors",
                        "reason": f"Routing supervisor failed: {str(e)}",
                    }
                ],
                "revision_notes": "Strengthen filing-based grounding and re-check completeness.",
                "warning": f"Fallback path triggered: {str(e)}",
            },
        }

    debug_state_snapshot("routing_supervisor (after)", {**state, **updates})
    return updates


def targeted_retrieval(state: ResearchState) -> dict:
    ticker = state["ticker"]
    issues = state.get("issues", [])
    evidence = list(state.get("evidence", []))

    for issue in issues:
        target = issue.get("target_agent")
        action = issue.get("action")

        if target == "filing" and action == "retrieve_risk_factors":
            result = answer_10q_question.invoke({
                "ticker": ticker,
                "question": "What are the key risks, liquidity issues, and recent management concerns in the latest 10-Q?"
            })
            evidence.append(str(result))

        elif target == "financial" and action == "retrieve_ratios":
            result = get_financial_ratios.invoke({"ticker": ticker})
            evidence.append(str(result))

        elif target == "sentiment" and action == "retrieve_sentiment":
            result = analyze_quarterly_sentiment.invoke({"ticker": ticker, "num_quarters": 4})
            evidence.append(str(result))

    updates = {"evidence": evidence}
    debug_state_snapshot("targeted_retrieval (after)", {**state, **updates})
    
    return updates


def _invoke_reviser(prompt: str, model: str = "gpt-5.4") -> str:
    llm = ChatOpenAI(model=model, temperature=0)

    system_prompt = """
            You are a senior equity research analyst revising an existing draft.

            Your task is to revise the draft using ONLY:
            - the original draft
            - the additional evidence provided
            - the revision instructions

            Rules:
            - Do not invent facts.
            - Do not ignore the supplied evidence.
            - Do not add claims unless they are supported by the supplied evidence.
            - Preserve the overall structure where possible.
            - Tighten unsupported language.
            - If evidence is insufficient, explicitly state the limitation.
            - Do not ask questions.
            - Do not mention tools or workflow.
            """

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ])

    return response.content if hasattr(response, "content") else str(response)



def revise_report(state: ResearchState) -> dict:
    ticker = state["ticker"]
    task = state["task"]
    draft = state["draft"]
    evidence_text = "\n\n".join(state.get("evidence", [])) or "No extra evidence was collected."
    revision_notes = state.get("revision_notes", "")

    revision_query = f"""
        Original draft:
        {draft}

        Additional evidence:
        {evidence_text}

        Revision instructions:
        {revision_notes}

        STRICT RULES:
        - Preserve ALL section headers exactly
        - Preserve section order exactly
        - Do NOT add new sections
        - Do NOT remove sections
        - Do NOT reorganize the report
        - Only modify sentences within existing sections
        - Only remove unsupported claims or tighten wording
        - Only add evidence-supported statements

        Revise the draft accordingly.
        """

    new_draft = _invoke_reviser(revision_query)

    return {
        "draft": new_draft,
        "revision_count": state.get("revision_count", 0) + 1,
    }



def finalize(state: ResearchState) -> dict:
    supervisor = OutputSupervisor(generation_model="gpt-5.4")

    review = supervisor.review(
        agent_name="ResearchCoordinator",
        query=state["query"],
        agent_output=state["draft"],
        tool_trace=state.get("tool_trace", ""), 
        supervision_notes=(
        "STRICT RULES:\n"
        "- Do NOT change section headers\n"
        "- Do NOT change section order\n"
        "- Do NOT restructure paragraphs\n"
        "- Do NOT rewrite the report from scratch\n"
        "- Only make minimal, surgical edits\n"
        "- Only remove unsupported claims or tighten wording\n"
        "- Preserve the original tone and structure\n"
        "- Do NOT introduce new analysis unless strictly necessary\n"
        )
    )

    updates = {
        "final_output": review.revised_output,
        "supervisor_info": review.to_dict(),
    }

    debug_state_snapshot("finalize (after)", {**state, **updates})
    return updates


# build langgraph app for agent supervision
def build_langgraph_app():
    graph = StateGraph(ResearchState)

    graph.add_node("prepare_query", prepare_query)
    graph.add_node("draft_report", draft_report)
    graph.add_node("supervisor_review", routing_supervisor)
    graph.add_node("targeted_retrieval", targeted_retrieval)
    graph.add_node("revise_report", revise_report)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "prepare_query")
    graph.add_edge("prepare_query", "draft_report")
    graph.add_edge("draft_report", "supervisor_review")

    graph.add_conditional_edges(
        "supervisor_review",
        route_after_review,
        {
            "finalize": "finalize",
            "targeted_retrieval": "targeted_retrieval",
        },
    )

    graph.add_edge("targeted_retrieval", "revise_report")
    graph.add_edge("revise_report", "supervisor_review")
    graph.add_edge("finalize", END)

    return graph.compile()


def route_after_review(state: ResearchState) -> str:
    if state.get("supervisor_pass", False):
        return "finalize"

    if state.get("revision_count", 0) >= state.get("max_revisions", 2):
        return "finalize"

    return "targeted_retrieval"



def run_agent_for_ticker_langgraph(ticker: str, task: str):
    app = build_langgraph_app()

    result = app.invoke({
        "ticker": ticker,
        "task": task,
        "revision_count": 0,
        "max_revisions": 2,
        "evidence": [],
    })

    return result


########################################################################################################################################
# ──────────────────────────────────────────────────────────────────────────────

def make_report_filename(ticker: str, report_type: str = "analysis", supervise: bool = False) -> str:
    safe_ticker = re.sub(r"[^A-Za-z0-9_-]", "", ticker.upper())
    date_str = datetime.now().strftime("%Y-%m-%d")

    mode = "supervised" if supervise else "unsupervised"

    return f"{safe_ticker}_{report_type}_{mode}_langgraph_{date_str}.md"
    

# Example usage:

if __name__ == "__main__":
    ticker = "AAPL"
    task = "Give me a full analysis and flag any risks from the latest 10-Q"
    supervise = True

    result = run_agent_for_ticker_langgraph(ticker, task)

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
    trace_path = f"evaluation/{ticker}_tool_trace_{'supervised' if supervise else 'unsupervised'}_langgraph.txt"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(output)

    with open(trace_path, "w", encoding="utf-8") as f:
        f.write(tool_trace)

    print("\n===== RAW RESULT TYPE =====\n")
    print(type(raw_for_trace))

    print("\n===== FINAL OUTPUT =====\n")
    print(repr(output))

    print("\n===== SUPERVISOR REVIEW =====\n")
    print(supervisor_info)

    print(f"\nSaved report to: {file_path}")
    print(f"Saved tool trace to: {trace_path}")

    # to run:
    # PYTHONPATH=. /usr/local/bin/python3 -m backend.agents.test_agent_super_final2
    