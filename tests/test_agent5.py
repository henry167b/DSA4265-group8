from dotenv import load_dotenv
load_dotenv()

from backend.agents.agent5 import SeniorAnalystAgent

def main():
    agent = SeniorAnalystAgent()

    output = agent.edit_report(
        ticker="NVDA",
        draft_report="""
NVIDIA reported strong recent performance, driven mainly by data center demand and AI-related growth.
However, valuation remains elevated and investors may face risks if growth expectations moderate.
""",
        chart_paths=[],
        price_data={
            "current_price": 900,
            "market_cap": 2200000000000,
        },
        ratios={
            "pe_ratio": 60,
            "forward_pe": 45,
            "ev_ebitda": 35,
            "ebitda_margin": 55,
            "net_margin": 48,
        },
        filings=[
            {
                "quarter": "2025 Q1",
                "sentiment_score": 0.72,
                "tone": "Positive",
                "key_themes": ["AI demand", "Data center growth"]
            },
            {
                "quarter": "2024 Q4",
                "sentiment_score": 0.66,
                "tone": "Constructive",
                "key_themes": ["Margin expansion", "Strong guidance"]
            }
        ],
        user_requirements="Professional tone. Max 400 words. Focus on valuation and risks."
    )

    print("\n===== AGENT 5 OUTPUT =====\n")
    print("Recommendation:", output["recommendation"])
    print("Confidence:", output["confidence"])
    print("Success:", output["success"])
    print("Key Drivers:", output["key_drivers"])
    print("Key Risks:", output["key_risks"])
    print("\nFinal Report:\n")
    print(output["final_report"])
    print("\nEdit Log:\n")
    for item in output["edit_log"]:
        print("-", item)

if __name__ == "__main__":
    main()