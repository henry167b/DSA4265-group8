import re
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup


YEAR_PATTERN = re.compile(r"\b20\d{2}\b")
ITEM_HEADING_RE = re.compile(r"^(item\s+\d+[a-z]?\b[\.\s])", re.IGNORECASE)
DATE_PATTERN = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)
PERIOD_HEADER_PATTERN = re.compile(
    r"\b(?:three months ended|nine months ended|quarter ended|year ended|as of|in millions|in billions|unaudited)\b",
    re.IGNORECASE,
)
SEGMENT_KEYWORDS = [
    "Data Center",
    "Gaming",
    "Professional Visualization",
    "Automotive",
    "OEM and Other",
    "Google Cloud",
    "Google Services",
    "Family of Apps",
    "Reality Labs",
    "Services",
    "iPhone",
    "Mac",
    "iPad",
    "Wearables, Home and Accessories",
    "Automotive sales",
    "Energy generation and storage",
]
METRIC_KEYWORDS = [
    "Revenue",
    "Net income",
    "Gross profit",
    "Gross margin",
    "Operating income",
    "Income from operations",
    "Income tax expense",
    "Provision for income taxes",
    "Cash and cash equivalents",
    "Total assets",
    "Total liabilities",
    "Research and development",
    "Basic earnings per share",
    "Diluted earnings per share",
    "Net cash provided by operating activities",
    "Net cash used in investing activities",
    "Net cash provided by financing activities",
]
SKIP_PATTERNS = [
    re.compile(r"^\d+$"),
    re.compile(r"^page \d+$", re.IGNORECASE),
    re.compile(r"^table of contents$", re.IGNORECASE),
    re.compile(r"^\[image\]$", re.IGNORECASE),
    re.compile(r"^\[graphic\]$", re.IGNORECASE),
    re.compile(r"^\(continued\)$", re.IGNORECASE),
]
NOISE_LINE_PATTERNS = [
    re.compile(r"^(?:true|false|none|nil)$", re.IGNORECASE),
    re.compile(r"^(?:q[1-4]|fy\d{2,4}|p\d+[dym])$", re.IGNORECASE),
    re.compile(r"^(?:dei|us-gaap|iso4217)(?::|$)", re.IGNORECASE),
]


def prepare_filing_html_for_chunking(
    html_content: str,
    prose_chunk_size: int = 600,
    prose_chunk_overlap: int = 100,
    table_window: int = 10,
    table_overlap: int = 2,
) -> Dict:
    """Turn filing HTML into prose chunks and structured table facts."""
    soup = BeautifulSoup(html_content, "lxml")

    for element in soup(["style", "script", "img", "svg", "canvas", "object", "embed"]):
        element.decompose()

    for tag in soup.find_all(True):
        if ":" in (tag.name or ""):
            tag.unwrap()

    table_groups, table_structures = _extract_table_groups(soup)
    prose_paragraphs = _extract_prose_paragraphs(soup)
    prose_text = "\n\n".join(prose_paragraphs)
    prose_chunk_records = _build_prose_chunk_records(
        prose_paragraphs,
        prose_chunk_size,
        prose_chunk_overlap,
    )
    prose_chunks = [record["text"] for record in prose_chunk_records]
    table_chunk_records = _build_table_chunk_records(
        table_groups,
        table_structures,
        table_window=table_window,
        table_overlap=table_overlap,
    )
    table_chunks = [record["text"] for record in table_chunk_records]

    return {
        "prose_text": prose_text,
        "table_groups": table_groups,
        "table_structures": table_structures,
        "prose_chunk_records": prose_chunk_records,
        "prose_chunks": prose_chunks,
        "table_chunk_records": table_chunk_records,
        "table_chunks": table_chunks,
        "chunks": prose_chunks + table_chunks,
    }


def _extract_table_groups(soup: BeautifulSoup) -> Tuple[Dict[str, List[str]], List[Dict]]:
    table_groups: Dict[str, List[str]] = {}
    table_structures: List[Dict] = []

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            table.decompose()
            continue

        title = _get_table_title(table)

        header_rows: List[List[str]] = []
        data_rows = rows

        for row in rows:
            candidate_cells = [
                _clean_cell(cell.get_text(separator=" ", strip=True))
                for cell in row.find_all(["th", "td"])
            ]
            candidate_cells = _merge_dollar_fragments(candidate_cells)
            if _row_looks_like_header(candidate_cells):
                header_rows.append(candidate_cells)
                continue
            data_rows = rows[len(header_rows):]
            break

        col_headers = _combine_header_rows(header_rows)
        unit = _extract_unit_from_text(" ".join(col_headers + [title]))
        table_rows: List[Dict] = []

        for row in data_rows:
            all_cells = [
                _clean_cell(cell.get_text(separator=" ", strip=True))
                for cell in row.find_all(["td", "th"])
            ]
            all_cells = _merge_dollar_fragments(all_cells)
            non_empty = [cell for cell in all_cells if cell]
            if len(non_empty) < 2:
                continue

            label = all_cells[0] if all_cells and all_cells[0] else non_empty[0]
            if _should_skip_row(label, non_empty):
                continue

            if col_headers:
                for idx in range(1, len(all_cells)):
                    value = all_cells[idx].strip()
                    if not value:
                        continue
                    col_header = col_headers[idx].strip() if idx < len(col_headers) else ""
                    row_text = f"{label} ({col_header}): {value}" if col_header else f"{label}: {value}"
                    table_groups.setdefault(title, []).append(_clean_row_string(row_text))
            else:
                row_text = non_empty[0] + ": " + ", ".join(non_empty[1:])
                table_groups.setdefault(title, []).append(_clean_row_string(row_text))

            table_rows.append(
                {
                    "label": label,
                    "cells": all_cells,
                    "non_empty": non_empty,
                }
            )

        if table_rows and _is_financial_table(title, col_headers, table_rows):
            table_structures.append(
                {
                    "title": title,
                    "column_headers": col_headers,
                    "unit": unit,
                    "rows": table_rows,
                }
            )
        table.decompose()

    return (
        {title: rows for title, rows in table_groups.items() if rows},
        table_structures,
    )


def _extract_prose_paragraphs(soup: BeautifulSoup) -> List[str]:
    raw_text = soup.get_text(separator="\n")
    paragraphs: List[str] = []
    current_lines: List[str] = []

    for raw_line in raw_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if (
            not line
            or any(pattern.match(line) for pattern in SKIP_PATTERNS)
            or _looks_like_noise_line(line)
        ):
            continue

        if ITEM_HEADING_RE.match(line) or _looks_like_heading(line):
            if current_lines:
                paragraphs.append(" ".join(current_lines))
                current_lines = []
            paragraphs.append(line)
            continue

        current_lines.append(line)

    if current_lines:
        paragraphs.append(" ".join(current_lines))

    return paragraphs


def _looks_like_heading(line: str) -> bool:
    if len(line) > 150:
        return False
    if line.endswith(":"):
        return True
    return line.isupper() and any(char.isalpha() for char in line)


def _build_prose_chunk_records(
    paragraphs: List[str],
    chunk_size: int,
    overlap: int,
) -> List[Dict]:
    records: List[Dict] = []
    current_section = "General"
    buffer: List[str] = []

    def flush_buffer(section_name: str) -> None:
        nonlocal buffer
        if not buffer:
            return
        text = "\n\n".join(buffer).strip()
        if not text:
            buffer = []
            return
        records.append(
            {
                "text": text,
                "section_name": section_name,
                "statement_type": _infer_statement_type(text, section_name=section_name),
                "metric_name": _infer_metric_name(text),
                "period_type": _infer_period_type(text),
                "period_end": _infer_period_end(text),
                "segment_name": _infer_segment_name(text),
            }
        )
        if overlap <= 0:
            buffer = []
            return
        retained: List[str] = []
        retained_length = 0
        for paragraph in reversed(buffer):
            retained.insert(0, paragraph)
            retained_length += len(paragraph) + 2
            if retained_length >= overlap:
                break
        buffer = retained

    for paragraph in paragraphs:
        if _is_heading_paragraph(paragraph):
            flush_buffer(current_section)
            current_section = paragraph
            continue

        candidate = buffer + [paragraph]
        candidate_text = "\n\n".join(candidate)
        if buffer and len(candidate_text) > chunk_size:
            flush_buffer(current_section)
        buffer.append(paragraph)

    flush_buffer(current_section)
    return records


def _build_table_chunk_records(
    table_groups: Dict[str, List[str]],
    table_structures: List[Dict],
    table_window: int,
    table_overlap: int,
) -> List[Dict]:
    records: List[Dict] = []
    structure_by_title = {table["title"]: table for table in table_structures}

    for title, rows in table_groups.items():
        if not rows:
            continue
        structure = structure_by_title.get(title, {})
        combined_rows = _window_table_rows(rows, table_window, table_overlap)
        statement_type = _infer_statement_type(title, section_name=title)
        unit = structure.get("unit")

        for window_rows in combined_rows:
            chunk_text = f"[{title}]\n" + "\n".join(window_rows)
            records.append(
                {
                    "text": chunk_text,
                    "section_name": title,
                    "statement_type": statement_type,
                    "metric_name": _infer_metric_name(chunk_text),
                    "period_type": _infer_period_type(chunk_text),
                    "period_end": _infer_period_end(chunk_text),
                    "segment_name": _infer_segment_name(chunk_text),
                    "table_title": title,
                    "value": None,
                    "unit": unit,
                    "comparison_period": None,
                    "column_label": None,
                    "parsing_mode": "window",
                }
            )

    return records


def _build_fact_records_from_row(
    title: str,
    row: Dict,
    column_headers: List[str],
    statement_type: Optional[str],
    unit: Optional[str],
) -> List[Dict]:
    label = row.get("label", "").strip()
    metric_name = _normalize_metric_label(label) or _infer_metric_name(label)
    if not metric_name:
        return []

    values = row.get("cells", [])[1:]
    if not values:
        return []

    header_values = column_headers[1:] if column_headers and len(column_headers) > 1 else []

    records: List[Dict] = []
    for index, value in enumerate(values):
        value = value.strip()
        if not value or not _looks_value_like(value):
            continue
        column_label = header_values[index] if index < len(header_values) else ""
        descriptor = _parse_period_descriptor(column_label)
        text_lines = [f"[{title}]", f"Metric: {metric_name}"]
        if column_label:
            text_lines.append(f"Column: {column_label}")
        if descriptor["period_label"]:
            text_lines.append(f"Period: {descriptor['period_label']}")
        text_lines.append(f"Value: {value}")
        if unit:
            text_lines.append(f"Unit: {unit}")
        records.append(
            {
                "text": "\n".join(text_lines),
                "section_name": title,
                "statement_type": statement_type,
                "metric_name": metric_name,
                "period_type": descriptor["period_type"],
                "period_end": descriptor["period_end"],
                "segment_name": _infer_segment_name(label),
                "table_title": title,
                "value": value,
                "unit": unit,
                "comparison_period": descriptor["comparison_period"],
                "column_label": column_label or None,
                "parsing_mode": "fact",
            }
        )

    return records


def _clean_cell(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _window_table_rows(rows: List[str], table_window: int, table_overlap: int) -> List[List[str]]:
    if not rows:
        return []
    if table_window <= 0:
        return [rows]

    windows: List[List[str]] = []
    step = max(1, table_window - max(0, table_overlap))
    for start in range(0, len(rows), step):
        window = rows[start:start + table_window]
        if window:
            windows.append(window)
        if start + table_window >= len(rows):
            break
    return windows


def _clean_row_string(row_text: str) -> str:
    row_text = re.sub(r"\$\s*,\s*(?=[\d(])", "$", row_text)
    row_text = re.sub(r"(\d)\s*,\s*%", r"\1%", row_text)
    row_text = re.sub(r"^,\s*", "", row_text)
    row_text = re.sub(r",\s*$", "", row_text)
    row_text = re.sub(r",\s*,", ",", row_text)
    return row_text.strip()


def _merge_dollar_fragments(cells: List[str]) -> List[str]:
    merged: List[str] = []
    index = 0

    while index < len(cells):
        if cells[index] == "$" and index + 1 < len(cells):
            merged.append("$" + cells[index + 1])
            index += 2
            continue
        merged.append(cells[index])
        index += 1

    return merged


def _should_skip_row(label: str, non_empty: List[str]) -> bool:
    lowered = label.lower().strip()
    if lowered in {"", "yes", "no"}:
        return True
    if any(
        token in lowered
        for token in [
            "large accelerated filer",
            "trading symbol",
            "title of each class",
            "commission file number",
            "state of incorporation",
            "state or other jurisdiction",
            "approximate date of commencement",
            "address of principal executive offices",
            "indicate by check mark",
        ]
    ):
        return True
    if len(non_empty) <= 2 and not any(_looks_value_like(value) for value in non_empty[1:]):
        return True
    return False


def _row_looks_like_header(cells: List[str]) -> bool:
    non_empty = [cell for cell in cells if cell]
    if len(non_empty) < 2:
        return False

    header_like = 0
    data_like = 0
    for cell in non_empty[1:]:
        if _is_header_like_cell(cell):
            header_like += 1
        elif _looks_numeric(cell):
            data_like += 1

    if header_like and data_like == 0:
        return True
    return bool(header_like and header_like >= data_like and any(not cell for cell in cells[:1]))


def _combine_header_rows(header_rows: List[List[str]]) -> List[str]:
    if not header_rows:
        return []
    width = max(len(row) for row in header_rows)
    combined = [""] * width
    for idx in range(width):
        parts = []
        for row in header_rows:
            if idx >= len(row):
                continue
            value = row[idx].strip()
            if value and value not in parts:
                parts.append(value)
        combined[idx] = " ".join(parts).strip()
    return combined


def _is_header_like_cell(text: str) -> bool:
    lowered = text.lower()
    return bool(
        YEAR_PATTERN.search(text)
        or DATE_PATTERN.search(text)
        or PERIOD_HEADER_PATTERN.search(lowered)
    )


def _looks_numeric(text: str) -> bool:
    cleaned = text.replace(",", "").replace("$", "").replace("%", "").strip()
    cleaned = cleaned.strip("()")
    return bool(cleaned) and cleaned.replace(".", "", 1).isdigit()


def _looks_like_noise_line(text: str) -> bool:
    lowered = text.lower().strip()
    if any(pattern.match(lowered) for pattern in NOISE_LINE_PATTERNS):
        return True
    if len(lowered) <= 2:
        return True
    if "http://" in lowered or "https://" in lowered:
        return True
    if lowered.count(":") >= 2 and " " not in lowered:
        return True
    alpha_chars = sum(char.isalpha() for char in lowered)
    digit_chars = sum(char.isdigit() for char in lowered)
    if alpha_chars == 0 and digit_chars > 0:
        return True
    if alpha_chars and digit_chars and digit_chars > alpha_chars * 2 and len(lowered) < 30:
        return True
    return False


def _looks_value_like(text: str) -> bool:
    if _looks_numeric(text):
        return True
    return bool(re.match(r"^\(?\$?[\d,]+(?:\.\d+)?%?\)?$", text.strip()))


def _get_table_title(table) -> str:
    caption = table.find("caption")
    if caption:
        title = caption.get_text(strip=True)
        if title:
            return title

    for prev in table.find_all_previous(["h1", "h2", "h3", "h4", "p"], limit=5):
        title = prev.get_text(strip=True)
        if title and 10 < len(title) < 150:
            return title

    return "Unlabeled Table"


def _is_heading_paragraph(paragraph: str) -> bool:
    return ITEM_HEADING_RE.match(paragraph) is not None or _looks_like_heading(paragraph)


def _infer_statement_type(text: str, section_name: Optional[str] = None) -> Optional[str]:
    haystack = " ".join(part for part in [section_name or "", text] if part).lower()
    if any(term in haystack for term in ["cash flow", "operating activities", "investing activities", "financing activities"]):
        return "cash_flow"
    if any(term in haystack for term in ["cash and cash equivalents", "total assets", "accounts receivable", "inventories", "stockholders", "total liabilities"]):
        return "balance_sheet"
    if any(term in haystack for term in ["revenue", "net income", "gross profit", "gross margin", "operating income", "income from operations", "earnings per share", "income tax expense", "research and development"]):
        return "income_statement"
    if any(term.lower() in haystack for term in SEGMENT_KEYWORDS):
        return "segment"
    if "risk factor" in haystack:
        return "risk_factors"
    return None


def _normalize_metric_label(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    label = text.strip().strip(":").strip()
    label = re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()
    if not label or len(label) > 120:
        return None
    lowered = label.lower()
    if any(token in lowered for token in ["table", "continued", "notes to condensed"]):
        return None
    return label


def _infer_metric_name(text: str) -> Optional[str]:
    stripped = text.strip()
    if ":" in stripped:
        label = stripped.split(":", 1)[0].strip("[] ")
        if 1 <= len(label) <= 120:
            return label
    lowered = stripped.lower()
    for keyword in METRIC_KEYWORDS:
        if keyword.lower() in lowered:
            return keyword
    return None


def _infer_period_type(text: str) -> Optional[str]:
    lowered = text.lower()
    if "three months ended" in lowered or "quarter ended" in lowered:
        return "quarter"
    if "nine months ended" in lowered or "year to date" in lowered:
        return "year_to_date"
    if any(term in lowered for term in ["as of", "ending balances", "balance as of"]):
        return "instant"
    return None


def _infer_period_end(text: str) -> Optional[str]:
    match = DATE_PATTERN.search(text)
    if match:
        return match.group(0)
    return None


def _infer_segment_name(text: str) -> Optional[str]:
    lowered = text.lower()
    for keyword in SEGMENT_KEYWORDS:
        if keyword.lower() in lowered:
            return keyword
    return None


def _extract_unit_from_text(text: str) -> Optional[str]:
    lowered = text.lower()
    if "in millions" in lowered:
        return "USD millions"
    if "in billions" in lowered:
        return "USD billions"
    if "%" in lowered or "margin" in lowered:
        return "percent"
    return None


def _parse_period_descriptor(header_text: str) -> Dict[str, Optional[str]]:
    lowered = header_text.lower()
    period_end = _infer_period_end(header_text)
    if "three months ended" in lowered or "quarter ended" in lowered:
        period_type = "quarter"
    elif "nine months ended" in lowered or "year to date" in lowered:
        period_type = "year_to_date"
    elif "as of" in lowered:
        period_type = "instant"
    elif YEAR_PATTERN.search(header_text):
        period_type = "annual_or_comparative"
    else:
        period_type = None

    comparison_period = None
    if period_end and YEAR_PATTERN.search(header_text):
        comparison_period = period_end

    return {
        "period_label": header_text.strip() or None,
        "period_type": period_type,
        "period_end": period_end,
        "comparison_period": comparison_period,
    }


def _is_financial_table(title: str, column_headers: List[str], rows: List[Dict]) -> bool:
    text_blob = " ".join([title] + column_headers + [row.get("label", "") for row in rows[:12]]).lower()
    if any(
        token in text_blob
        for token in [
            "large accelerated filer",
            "trading symbol",
            "zip code",
            "commission file number",
            "state or other jurisdiction",
            "title of each class",
        ]
    ):
        return False

    financial_signal = any(
        token in text_blob
        for token in [
            "revenue",
            "net income",
            "gross margin",
            "research and development",
            "cash and cash equivalents",
            "total assets",
            "total liabilities",
            "operating activities",
            "segment",
            "earnings per share",
            "statements of income",
            "balance sheets",
            "cash flows",
        ]
    )
    numeric_rows = sum(
        1 for row in rows if any(_looks_value_like(value) for value in row.get("cells", [])[1:])
    )
    return financial_signal and numeric_rows > 0
