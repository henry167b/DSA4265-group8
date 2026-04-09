import re
from typing import Dict, List, Optional

from bs4 import BeautifulSoup


ITEM_HEADING_RE = re.compile(r"^(item\s+\d+[a-z]?\b[\.\s])", re.IGNORECASE)
NOTES_HEADING_RE = re.compile(r"^notes to .+financial statements(?:\s*\(unaudited\))?$", re.IGNORECASE)
BLOCK_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "div",
    "li",
}
DATE_PATTERN = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)
STRICT_SUBHEADING_PATTERNS = [
    re.compile(r"^management'?s discussion and analysis of financial condition and results of operations$", re.IGNORECASE),
    re.compile(r"^results of operations$", re.IGNORECASE),
    re.compile(r"^liquidity and capital resources$", re.IGNORECASE),
    re.compile(r"^net sales$", re.IGNORECASE),
    re.compile(r"^gross margin$", re.IGNORECASE),
    re.compile(r"^operating expenses$", re.IGNORECASE),
    re.compile(r"^research and development$", re.IGNORECASE),
    re.compile(r"^selling,\s*general and administrative$", re.IGNORECASE),
    re.compile(r"^provision for income taxes$", re.IGNORECASE),
    re.compile(r"^market risk$", re.IGNORECASE),
    re.compile(r"^controls and procedures$", re.IGNORECASE),
    re.compile(r"^legal proceedings$", re.IGNORECASE),
    re.compile(r"^risk factors$", re.IGNORECASE),
    re.compile(r"^notes to .+financial statements(?:\s*\(unaudited\))?$", re.IGNORECASE),
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
PROSE_SKIP_PATTERNS = [
    re.compile(r"^securities and exchange commission$", re.IGNORECASE),
    re.compile(r"^washington,\s*d\.c\.\s*\d{5}$", re.IGNORECASE),
    re.compile(r"^form\s+10-[qk]$", re.IGNORECASE),
    re.compile(
        r"^[☒☐]?\s*quarterly report pursuant to section 13 or 15\(d\) of the securities exchange act of 1934$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^[☒☐]?\s*transition report pursuant to section 13 or 15\(d\) of the securities exchange act of 1934$",
        re.IGNORECASE,
    ),
    re.compile(r"^\(mark one\)$", re.IGNORECASE),
    re.compile(r"^for the quarterly period ended .+$", re.IGNORECASE),
    re.compile(r"^for the transition period from .+ to .+$", re.IGNORECASE),
    re.compile(r"^commission file number:?\s*[\w.-]+$", re.IGNORECASE),
    re.compile(r"^\(in [^)]+\)$", re.IGNORECASE),
    re.compile(r"^see accompanying notes to .+$", re.IGNORECASE),
    re.compile(r"^(?:none|not applicable)\.?$", re.IGNORECASE),
    re.compile(r"^\*\s*filed herewith\.?(?:\s+\*\*\s*furnished herewith\.?)?$", re.IGNORECASE),
]
PAGE_ARTIFACT_RE = re.compile(
    r"\b[^|]{1,120}\|\s*Q[1-4]\s+\d{4}\s+Form\s+10-[QK]\s+\|\s*\d+\b",
    re.IGNORECASE,
)


def prepare_filing_html_for_chunking(
    html_content: str,
    prose_chunk_size: int = 1200,
    prose_chunk_overlap: int = 150,
    table_window: int = 10,
    table_overlap: int = 2,
) -> Dict:
    """Turn filing HTML into prose and structured table chunks for retrieval."""
    soup = BeautifulSoup(html_content, "lxml")

    for element in soup(["style", "script", "img", "svg", "canvas", "object", "embed"]):
        element.decompose()

    for tag in soup.find_all(True):
        if ":" in (tag.name or ""):
            tag.unwrap()

    prose_soup = BeautifulSoup(str(soup), "lxml")
    for table in prose_soup.find_all("table"):
        table.decompose()

    prose_paragraphs = _extract_prose_paragraphs(prose_soup)
    prose_text = "\n\n".join(prose_paragraphs)
    prose_chunk_records = _build_prose_chunk_records(
        prose_paragraphs,
        prose_chunk_size,
        prose_chunk_overlap,
    )
    prose_chunks = [record["text"] for record in prose_chunk_records]
    table_structures = _extract_table_structures(soup)
    table_chunk_records = _build_table_chunk_records(table_structures, table_window, table_overlap)
    table_chunks = [record["text"] for record in table_chunk_records]

    return {
        "prose_text": prose_text,
        "table_groups": _group_tables_by_section(table_structures),
        "table_structures": table_structures,
        "prose_chunk_records": prose_chunk_records,
        "prose_chunks": prose_chunks,
        "table_chunk_records": table_chunk_records,
        "table_chunks": table_chunks,
        "chunks": prose_chunks + table_chunks,
    }


def _extract_prose_paragraphs(soup: BeautifulSoup) -> List[str]:
    raw_blocks = _extract_html_text_blocks(soup)
    cleaned_blocks = [_clean_prose_block_text(raw_line) for raw_line in raw_blocks]
    body_start_available = any(
        line and _is_body_start_heading(line)
        for line in cleaned_blocks
    )
    paragraphs: List[str] = []
    current_lines: List[str] = []
    has_reached_body = not body_start_available

    for line in cleaned_blocks:
        if (
            not line
            or any(pattern.match(line) for pattern in SKIP_PATTERNS)
            or any(pattern.match(line) for pattern in PROSE_SKIP_PATTERNS)
            or _looks_like_noise_line(line)
        ):
            continue

        is_heading = ITEM_HEADING_RE.match(line) or _looks_like_heading(line)
        if is_heading:
            if body_start_available and not has_reached_body and not _is_body_start_heading(line):
                current_lines = []
                continue
            if _is_body_start_heading(line):
                has_reached_body = True
            if current_lines:
                paragraphs.append(" ".join(current_lines))
                current_lines = []
            paragraphs.append(line)
            continue

        if not has_reached_body:
            continue

        current_lines.append(line)

    if current_lines:
        paragraphs.append(" ".join(current_lines))

    return paragraphs


def _extract_html_text_blocks(soup: BeautifulSoup) -> List[str]:
    blocks: List[str] = []
    seen: set[str] = set()
    root = soup.body or soup

    for element in root.find_all(BLOCK_TAGS):
        if _should_skip_block_element(element):
            continue

        text = _clean_prose_block_text(element.get_text(" ", strip=True))
        if (
            not text
            or text in seen
            or any(pattern.match(text) for pattern in SKIP_PATTERNS)
            or any(pattern.match(text) for pattern in PROSE_SKIP_PATTERNS)
            or _looks_like_noise_line(text)
        ):
            continue

        seen.add(text)
        blocks.append(text)

    return blocks


def _should_skip_block_element(element) -> bool:
    if element.name not in BLOCK_TAGS:
        return True

    if element.find_parent("table") is not None:
        return True

    if element.name == "div":
        child_blocks = [
            child for child in element.find_all(BLOCK_TAGS, recursive=False)
            if child.name in BLOCK_TAGS
        ]
        if child_blocks:
            return True

    if element.name == "li" and element.find_parent("table") is not None:
        return True

    return False


def _looks_like_heading(line: str) -> bool:
    if len(line) > 120:
        return False
    stripped = line.strip()
    if any(pattern.match(stripped) for pattern in STRICT_SUBHEADING_PATTERNS):
        return True

    alpha_chars = [char for char in stripped if char.isalpha()]
    if not alpha_chars:
        return False
    uppercase_ratio = sum(char.isupper() for char in alpha_chars) / len(alpha_chars)
    return uppercase_ratio >= 0.8 and len(alpha_chars) <= 80


def _is_body_start_heading(line: str) -> bool:
    return ITEM_HEADING_RE.match(line) is not None or NOTES_HEADING_RE.match(line) is not None


def _build_prose_chunk_records(
    paragraphs: List[str],
    chunk_size: int,
    overlap: int,
) -> List[Dict]:
    records: List[Dict] = []
    current_section = "General"
    buffer: List[str] = []

    def flush_buffer(section_name: str, carry_overlap: bool = True) -> None:
        nonlocal buffer
        if not buffer:
            return
        body = "\n\n".join(buffer).strip()
        if not body:
            buffer = []
            return
        text = f"[{section_name}]\n\n{body}" if section_name and section_name != "General" else body
        records.append(
            {
                "text": text,
                "section_name": section_name,
                "period_end": _infer_period_end(text),
            }
        )
        if overlap <= 0 or not carry_overlap:
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
            flush_buffer(current_section, carry_overlap=False)
            current_section = paragraph
            continue

        candidate = buffer + [paragraph]
        candidate_text = "\n\n".join(candidate)
        if buffer and len(candidate_text) > chunk_size:
            flush_buffer(current_section)
        buffer.append(paragraph)

    flush_buffer(current_section)
    return records


def _extract_table_structures(soup: BeautifulSoup) -> List[Dict]:
    structures: List[Dict] = []
    current_section = "General"

    for element in soup.body.descendants if soup.body else soup.descendants:
        if not getattr(element, "name", None):
            continue

        if element.name != "table":
            text = _normalize_text(element.get_text(" ", strip=True))
            if text and _is_heading_paragraph(text):
                current_section = text
            continue

        table = element
        rows = _extract_html_table_rows(table)
        if not rows:
            continue

        title = _find_table_title(table)
        if not _should_keep_table(current_section, title, rows):
            continue
        units = _infer_table_units(title, rows)
        period_end = _infer_period_end(" ".join(filter(None, [title] + [" | ".join(row) for row in rows[:3]])))

        structures.append(
            {
                "section_name": current_section,
                "table_title": title,
                "units": units,
                "period_end": period_end,
                "headers": rows[0],
                "rows": rows[1:] if len(rows) > 1 else [],
                "row_count": max(0, len(rows) - 1),
                "column_count": max((len(row) for row in rows), default=0),
            }
        )

    return structures


def _extract_html_table_rows(table) -> List[List[str]]:
    extracted_rows: List[List[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue

        row: List[str] = []
        for cell in cells:
            text = _normalize_text(cell.get_text(" ", strip=True))
            colspan_raw = cell.get("colspan", 1)
            try:
                colspan = max(1, int(colspan_raw))
            except (TypeError, ValueError):
                colspan = 1
            # Preserve column count without duplicating merged-cell text across every spanned column.
            row.append(text)
            if colspan > 1:
                row.extend([""] * (colspan - 1))

        if any(row):
            extracted_rows.append(row)

    width = max((len(row) for row in extracted_rows), default=0)
    normalized_rows: List[List[str]] = []
    for row in extracted_rows:
        padded = list(row) + ([""] * max(0, width - len(row)))
        if not _looks_like_empty_row(padded):
            normalized_rows.append(padded)
    return normalized_rows


def _should_keep_table(section_name: str, title: str, rows: List[List[str]]) -> bool:
    search_space = " ".join(
        [
            section_name or "",
            title or "",
            " ".join(rows[0]) if rows else "",
            " ".join(rows[1]) if len(rows) > 1 else "",
        ]
    ).lower()

    if "table of contents" in search_space:
        return False

    if rows and _looks_like_navigation_table(rows):
        return False

    alpha_chars = sum(char.isalpha() for char in search_space)
    if alpha_chars < 8:
        return False

    return True


def _looks_like_navigation_table(rows: List[List[str]]) -> bool:
    flat_cells = [cell.strip().lower() for row in rows for cell in row if cell.strip()]
    if not flat_cells:
        return True

    page_like = sum(
        1 for cell in flat_cells
        if cell == "page" or cell.isdigit() or re.fullmatch(r"item\s+\d+[a-z]?\.?", cell)
    )
    repeated = len(flat_cells) - len(set(flat_cells))

    if page_like >= max(3, len(flat_cells) // 4):
        return True
    if repeated >= max(4, len(flat_cells) // 3) and any("part" in cell for cell in flat_cells):
        return True

    return False


def _find_table_title(table) -> str:
    title_candidates: List[str] = []

    caption = table.find("caption")
    if caption:
        caption_text = _normalize_text(caption.get_text(" ", strip=True))
        if caption_text:
            title_candidates.append(caption_text)

    sibling = table.previous_sibling
    while sibling is not None and len(title_candidates) < 3:
        name = getattr(sibling, "name", None)
        if name in {"table"}:
            break
        if name in {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6"}:
            text = _normalize_text(sibling.get_text(" ", strip=True))
            if text and not _looks_like_noise_line(text):
                title_candidates.insert(0, text)
        sibling = sibling.previous_sibling

    cleaned_candidates: List[str] = []
    for candidate in title_candidates:
        if candidate not in cleaned_candidates:
            cleaned_candidates.append(candidate)

    if not cleaned_candidates:
        return "Table"
    return " | ".join(cleaned_candidates[-2:])


def _infer_table_units(title: str, rows: List[List[str]]) -> Optional[str]:
    search_space = " ".join(
        [
            title or "",
            " ".join(rows[0]) if rows else "",
            " ".join(rows[1]) if len(rows) > 1 else "",
        ]
    )
    match = re.search(r"\((in [^)]+)\)", search_space, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _build_table_chunk_records(table_structures: List[Dict], table_window: int, table_overlap: int) -> List[Dict]:
    records: List[Dict] = []
    safe_window = max(1, table_window)
    safe_overlap = max(0, min(table_overlap, safe_window - 1))

    for table_index, table in enumerate(table_structures):
        rows = table.get("rows") or []
        headers = table.get("headers") or []
        if not rows:
            continue

        start = 0
        chunk_index = 0
        while start < len(rows):
            window_rows = rows[start:start + safe_window]
            if not window_rows:
                break

            records.append(
                {
                    "text": _render_table_chunk_text(table, headers, window_rows, start, start + len(window_rows)),
                    "section_name": table.get("section_name"),
                    "period_end": table.get("period_end"),
                    "table_title": table.get("table_title"),
                    "units": table.get("units"),
                    "table_index": table_index,
                    "row_start": start,
                    "row_end": start + len(window_rows),
                }
            )
            chunk_index += 1
            if start + len(window_rows) >= len(rows):
                break
            start += max(1, safe_window - safe_overlap)

    return records


def _render_table_chunk_text(
    table: Dict,
    headers: List[str],
    rows: List[List[str]],
    row_start: int,
    row_end: int,
) -> str:
    lines = [f"[Table] {table.get('table_title') or 'Table'}"]
    if table.get("section_name"):
        lines.append(f"Section: {table['section_name']}")
    if table.get("units"):
        lines.append(f"Units: {table['units']}")
    if headers:
        lines.append("Columns: " + " | ".join(cell or "-" for cell in headers))
    lines.append(f"Rows: {row_start + 1}-{row_end}")

    row_header_label = headers[0] if headers and headers[0] else "Row"
    value_headers = headers[1:] if len(headers) > 1 else []

    for row in rows:
        row_label = row[0] if row and row[0] else row_header_label
        values = row[1:] if len(row) > 1 else []
        value_parts: List[str] = []
        for index, value in enumerate(values):
            if not value:
                continue
            header = value_headers[index] if index < len(value_headers) and value_headers[index] else f"Column {index + 2}"
            value_parts.append(f"{header}={value}")
        if value_parts:
            lines.append(f"{row_label}: " + "; ".join(value_parts))
        else:
            lines.append(row_label)

    return "\n".join(lines)


def _group_tables_by_section(table_structures: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for table in table_structures:
        section_name = table.get("section_name") or "General"
        grouped.setdefault(section_name, []).append(table)
    return grouped


def _looks_like_empty_row(row: List[str]) -> bool:
    return not any(cell.strip() for cell in row)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _clean_prose_block_text(text: str) -> str:
    cleaned = _normalize_text(text)
    if not cleaned:
        return ""
    cleaned = PAGE_ARTIFACT_RE.sub(" ", cleaned)
    return _normalize_text(cleaned)


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
def _is_heading_paragraph(paragraph: str) -> bool:
    return ITEM_HEADING_RE.match(paragraph) is not None or _looks_like_heading(paragraph)


def _infer_period_end(text: str) -> Optional[str]:
    match = DATE_PATTERN.search(text)
    if match:
        return match.group(0)
    return None
