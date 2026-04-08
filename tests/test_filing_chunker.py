import sys
from types import SimpleNamespace

sys.modules.setdefault("yfinance", SimpleNamespace())
sys.modules.setdefault("pandas", SimpleNamespace(MultiIndex=type("MultiIndex", (), {})))

from backend.agents.filing_chunker import prepare_filing_html_for_chunking


def test_prepare_filing_html_for_chunking_keeps_prose_and_structured_tables():
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
    assert "Data Center" not in prepared["prose_text"]
    assert prepared["table_groups"]["ITEM 1. BUSINESS"]
    assert len(prepared["table_structures"]) == 1
    assert len(prepared["table_chunk_records"]) == 1
    assert len(prepared["table_chunks"]) == 1
    assert len(prepared["chunks"]) == len(prepared["prose_chunks"]) + len(prepared["table_chunks"])
    assert prepared["prose_chunk_records"][0]["section_name"] == "ITEM 1. BUSINESS"
    assert prepared["table_structures"][0]["headers"] == ["Segment", "2024", "2023"]
    assert prepared["table_structures"][0]["rows"][0] == ["Data Center", "$", "47,500"]
    assert "Columns: Segment | 2024 | 2023" in prepared["table_chunks"][0]
    assert "Data Center: 2024=$; 2023=47,500" in prepared["table_chunks"][0]


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


def test_prepare_filing_html_for_chunking_preserves_table_units_and_rows():
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

    assert len(prepared["table_structures"]) == 2
    assert len(prepared["table_chunk_records"]) == 2
    assert prepared["table_structures"][0]["table_title"] == "Condensed Consolidated Statements of Income (Unaudited) (In millions)"
    assert prepared["table_structures"][0]["units"] == "In millions"
    assert prepared["table_structures"][0]["rows"][0] == ["Revenue", "57,006", "35,082"]
    assert "Units: In millions" in prepared["table_chunks"][0]
    assert "Revenue: Three months ended October 26, 2025=57,006; Three months ended October 27, 2024=35,082" in prepared["table_chunks"][0]
    assert "Revenue" not in prepared["prose_text"]
    assert "Trading Symbol" not in prepared["prose_text"]


def test_prepare_filing_html_for_chunking_does_not_promote_summary_sentences_to_sections():
    html = """
    <html>
      <body>
        <h1>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</h1>
        <p>A summary of the Company’s restricted stock unit activity and related information for the three months ended December 27, 2025, is as follows:</p>
        <p>Share-based compensation expense increased year over year.</p>
      </body>
    </html>
    """

    prepared = prepare_filing_html_for_chunking(html, prose_chunk_size=400, prose_chunk_overlap=0)

    assert len(prepared["prose_chunk_records"]) == 1
    assert prepared["prose_chunk_records"][0]["section_name"] == "Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations"
    assert "A summary of the Company’s restricted stock unit activity" in prepared["prose_chunk_records"][0]["text"]


def test_prepare_filing_html_for_chunking_does_not_carry_overlap_across_sections():
    html = """
    <html>
      <body>
        <h1>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</h1>
        <p>Revenue increased due to stronger iPhone demand.</p>
        <p>Services also contributed to growth.</p>
        <h2>Legal Proceedings</h2>
        <p>Apple appealed the EU decision during the quarter.</p>
      </body>
    </html>
    """

    prepared = prepare_filing_html_for_chunking(html, prose_chunk_size=60, prose_chunk_overlap=40)

    assert len(prepared["prose_chunk_records"]) == 2
    assert prepared["prose_chunk_records"][0]["section_name"] == "Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations"
    assert prepared["prose_chunk_records"][1]["section_name"] == "Legal Proceedings"
    assert "Revenue increased due to stronger iPhone demand." not in prepared["prose_chunk_records"][1]["text"]


def test_prepare_filing_html_for_chunking_splits_large_tables_by_row_window():
    html = """
    <html>
      <body>
        <h1>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</h1>
        <p>Net sales by product</p>
        <table>
          <tr><th>Product</th><th>2025</th><th>2024</th></tr>
          <tr><td>iPhone</td><td>69,138</td><td>45,963</td></tr>
          <tr><td>Services</td><td>26,340</td><td>23,117</td></tr>
          <tr><td>Mac</td><td>7,949</td><td>7,780</td></tr>
        </table>
      </body>
    </html>
    """

    prepared = prepare_filing_html_for_chunking(html, table_window=2, table_overlap=1)

    assert len(prepared["table_chunk_records"]) == 2
    assert "Rows: 1-2" in prepared["table_chunks"][0]
    assert "Rows: 2-3" in prepared["table_chunks"][1]
    assert "iPhone: 2025=69,138; 2024=45,963" in prepared["table_chunks"][0]
    assert "Mac: 2025=7,949; 2024=7,780" in prepared["table_chunks"][1]


def test_prepare_filing_html_for_chunking_uses_meaningful_html_blocks_for_prose():
    html = """
    <html>
      <body>
        <div>
          <h1>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</h1>
          <div>
            <p>Revenue increased because of stronger iPhone demand.</p>
            <p>Services also contributed to growth.</p>
          </div>
        </div>
      </body>
    </html>
    """

    prepared = prepare_filing_html_for_chunking(html, prose_chunk_size=80, prose_chunk_overlap=0)

    assert len(prepared["prose_chunk_records"]) == 1
    text = prepared["prose_chunk_records"][0]["text"]
    assert "Revenue increased because of stronger iPhone demand." in text
    assert "Services also contributed to growth." in text
    assert text.count("Revenue increased because of stronger iPhone demand.") == 1


def test_prepare_filing_html_for_chunking_drops_table_of_contents_tables():
    html = """
    <html>
      <body>
        <h1>TABLE OF CONTENTS</h1>
        <table>
          <tr><td>Part I</td><td>Page</td></tr>
          <tr><td>Item 1.</td><td>1</td></tr>
          <tr><td>Item 2.</td><td>13</td></tr>
        </table>
        <h1>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</h1>
        <p>Gross margin increased year over year.</p>
      </body>
    </html>
    """

    prepared = prepare_filing_html_for_chunking(html)

    assert prepared["table_structures"] == []
    assert prepared["table_chunk_records"] == []
    assert prepared["table_chunks"] == []
    assert "Gross margin increased year over year." in prepared["prose_text"]


def test_prepare_filing_html_for_chunking_does_not_repeat_colspan_text_in_tables():
    html = """
    <html>
      <body>
        <h1>Item 1. Business</h1>
        <table>
          <tr><th colspan="3">Revenue</th><th>2025</th></tr>
          <tr><td colspan="3">iPhone</td><td>69,138</td></tr>
        </table>
      </body>
    </html>
    """

    prepared = prepare_filing_html_for_chunking(html)

    assert prepared["table_structures"][0]["headers"] == ["Revenue", "", "", "2025"]
    assert prepared["table_structures"][0]["rows"][0] == ["iPhone", "", "", "69,138"]
    assert "Columns: Revenue | - | - | 2025" in prepared["table_chunks"][0]
    assert "Revenue | Revenue | Revenue" not in prepared["table_chunks"][0]
    assert "iPhone: 2025=69,138" in prepared["table_chunks"][0]


def test_prepare_filing_html_for_chunking_skips_cover_page_and_statement_boilerplate():
    html = """
    <html>
      <body>
        <h1>SECURITIES AND EXCHANGE COMMISSION</h1>
        <p>Washington, D.C. 20549</p>
        <h1>FORM 10-Q</h1>
        <p>(Mark One)</p>
        <h1>☒ QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934</h1>
        <p>For the quarterly period ended December 27, 2025</p>
        <h1>CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS (Unaudited)</h1>
        <p>(In millions)</p>
        <p>See accompanying Notes to Condensed Consolidated Financial Statements.</p>
        <h1>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</h1>
        <p>Revenue increased year over year.</p>
      </body>
    </html>
    """

    prepared = prepare_filing_html_for_chunking(html)

    assert len(prepared["prose_chunk_records"]) == 1
    assert prepared["prose_chunk_records"][0]["section_name"] == "Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations"
    assert prepared["prose_chunk_records"][0]["text"] == "Revenue increased year over year."


def test_prepare_filing_html_for_chunking_strips_page_footer_artifacts_from_prose():
    html = """
    <html>
      <body>
        <h1>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</h1>
        <p>Apple Inc. | Q1 2026 Form 10-Q | 5 Revenue increased because of stronger iPhone demand.</p>
      </body>
    </html>
    """

    prepared = prepare_filing_html_for_chunking(html, prose_chunk_size=400, prose_chunk_overlap=0)

    text = prepared["prose_chunk_records"][0]["text"]
    assert "Apple Inc. | Q1 2026 Form 10-Q | 5" not in text
    assert text == "Revenue increased because of stronger iPhone demand."


def test_prepare_filing_html_for_chunking_starts_prose_at_notes_when_present():
    html = """
    <html>
      <body>
        <h1>UNITED STATES</h1>
        <p>Apple Inc. (Exact name of Registrant as specified in its charter)</p>
        <h1>CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS (Unaudited)</h1>
        <p>(In millions)</p>
        <h1>Notes to Condensed Consolidated Financial Statements (Unaudited)</h1>
        <p>The condensed consolidated financial statements include the accounts of Apple Inc. and its wholly owned subsidiaries.</p>
      </body>
    </html>
    """

    prepared = prepare_filing_html_for_chunking(html, prose_chunk_size=400, prose_chunk_overlap=0)

    assert len(prepared["prose_chunk_records"]) == 1
    assert prepared["prose_chunk_records"][0]["section_name"] == "Notes to Condensed Consolidated Financial Statements (Unaudited)"
    assert "The condensed consolidated financial statements include the accounts of Apple Inc." in prepared["prose_chunk_records"][0]["text"]
