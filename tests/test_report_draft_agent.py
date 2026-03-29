from backend.agents.report_draft_agent import ReportDraftAgent
from backend.agents.yahoo_finance_agent import YahooFinanceAgent

# Example usage and test function
def test_agent():
    """Test the Report Draft Agent with Nvidia."""
    data_agent = YahooFinanceAgent()
    report_agent = ReportDraftAgent()

    # Agent 1 data
    data = data_agent.get_stock_data("NVDA")

    if data["success"]:
        # create visualization
        chart_file = report_agent.create_visualizations(data)
        
        # draft pdf
        pdf_file = report_agent.generate_report_draft(data, chart_file)
        
        print(f"Success! Report generated at: {pdf_file}")

    else:
        print(f"Error: {data['error']}")


if __name__ == "__main__":
    test_agent()
