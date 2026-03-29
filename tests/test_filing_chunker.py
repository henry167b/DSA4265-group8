import sys
from types import SimpleNamespace

sys.modules.setdefault("yfinance", SimpleNamespace())
sys.modules.setdefault("pandas", SimpleNamespace(MultiIndex=type("MultiIndex", (), {})))

from backend.agents.filing_chunker import prepare_filing_html_for_chunking


def test_prepare_filing_html_for_chunking_separates_tables_and_prose():
    html = """
    <html>
      <body>
        <h1>ITEM 1. BUSINESS</h1>
        <p>We make semiconductors for accelerated computing.</p>
        <p>Demand remained strong throughout the quarter.</p>
        <h2>Quarterly Revenue by Segment</h2>
        <table>
          <tr><th>Segment</th><th>2024</th><th>2023</th></tr>
          <tr><td>Data Center</td><td>$</td><td>47,500</td></tr>
          <tr><td>Gaming</td><td>12,000</td><td>9,000</td></tr>
        </table>
      </body>
    </html>
    """

    prepared = prepare_filing_html_for_chunking(
        html,
        prose_chunk_size=120,
        prose_chunk_overlap=20,
        table_window=5,
        table_overlap=1,
    )

    assert "ITEM 1. BUSINESS" in prepared["prose_text"]
    assert "Demand remained strong throughout the quarter." in prepared["prose_text"]
    assert "Quarterly Revenue by Segment" in prepared["table_groups"]
    assert "Gaming (2024): 12,000" in prepared["table_groups"]["Quarterly Revenue by Segment"]
    assert any(chunk.startswith("[Quarterly Revenue by Segment]") for chunk in prepared["table_chunks"])
    assert prepared["chunks"] == prepared["prose_chunks"] + prepared["table_chunks"]


def test_prepare_filing_html_for_chunking_ignores_non_text_elements():
    html = """
    <html>
      <body>
        <script>console.log('ignore me')</script>
        <style>.hidden { display: none; }</style>
        <p>Core operating discussion.</p>
        <img src="chart.png" />
      </body>
    </html>
    """

    prepared = prepare_filing_html_for_chunking(html)

    assert "ignore me" not in prepared["prose_text"]
    assert "display: none" not in prepared["prose_text"]
    assert "Core operating discussion." in prepared["prose_text"]
