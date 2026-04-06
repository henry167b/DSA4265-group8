import sys
from types import SimpleNamespace

sys.modules.setdefault("yfinance", SimpleNamespace())
sys.modules.setdefault("pandas", SimpleNamespace(MultiIndex=type("MultiIndex", (), {})))

from backend.agents.filing_chunker import prepare_filing_html_for_chunking


def test_prepare_filing_html_for_chunking_keeps_only_prose_content():
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
    assert prepared["table_groups"] == {}
    assert prepared["table_structures"] == []
    assert prepared["table_chunk_records"] == []
    assert prepared["table_chunks"] == []
    assert prepared["chunks"] == prepared["prose_chunks"]
    assert prepared["prose_chunk_records"][0]["section_name"] == "ITEM 1. BUSINESS"
    assert all("Data Center" not in chunk for chunk in prepared["chunks"])


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


def test_prepare_filing_html_for_chunking_drops_tables_entirely():
    html = """
    <html>
      <body>
        <p>Condensed Consolidated Statements of Income (Unaudited) (In millions)</p>
        <table>
          <tr><th></th><th>Three months ended October 26, 2025</th><th>Three months ended October 27, 2024</th></tr>
          <tr><td>Revenue</td><td>57,006</td><td>35,082</td></tr>
          <tr><td>Net income</td><td>31,910</td><td>19,309</td></tr>
        </table>
        <table>
          <tr><td>Title of each class</td><td>Trading Symbol(s)</td></tr>
          <tr><td>Common Stock</td><td>NVDA</td></tr>
        </table>
      </body>
    </html>
    """

    prepared = prepare_filing_html_for_chunking(html)

    assert prepared["table_structures"] == []
    assert prepared["table_chunk_records"] == []
    assert prepared["table_chunks"] == []
    assert "Revenue" not in prepared["prose_text"]
    assert "Trading Symbol" not in prepared["prose_text"]
