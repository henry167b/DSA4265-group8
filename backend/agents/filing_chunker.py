import re
from typing import Dict, List

from bs4 import BeautifulSoup


YEAR_PATTERN = re.compile(r"\b20\d{2}\b")
ITEM_HEADING_RE = re.compile(r"^(item\s+\d+[a-z]?\b[\.\s])", re.IGNORECASE)
SKIP_PATTERNS = [
    re.compile(r"^\d+$"),
    re.compile(r"^page \d+$", re.IGNORECASE),
    re.compile(r"^table of contents$", re.IGNORECASE),
    re.compile(r"^\[image\]$", re.IGNORECASE),
    re.compile(r"^\[graphic\]$", re.IGNORECASE),
    re.compile(r"^\(continued\)$", re.IGNORECASE),
]


def prepare_filing_html_for_chunking(
    html_content: str,
    prose_chunk_size: int = 600,
    prose_chunk_overlap: int = 100,
    table_window: int = 10,
    table_overlap: int = 2,
) -> Dict:
    """Turn filing HTML into prose chunks and table-aware chunks."""
    soup = BeautifulSoup(html_content, "lxml")

    for element in soup(["style", "script", "img", "svg", "canvas", "object", "embed"]):
        element.decompose()

    for tag in soup.find_all(True):
        if ":" in (tag.name or ""):
            tag.unwrap()

    table_groups = _extract_table_groups(soup)
    prose_text = _extract_prose_text(soup)
    prose_chunks = _chunk_text(prose_text, prose_chunk_size, prose_chunk_overlap)
    table_chunks = _build_table_chunks(table_groups, table_window, table_overlap)

    return {
        "prose_text": prose_text,
        "table_groups": table_groups,
        "prose_chunks": prose_chunks,
        "table_chunks": table_chunks,
        "chunks": prose_chunks + table_chunks,
    }


def _extract_table_groups(soup: BeautifulSoup) -> Dict[str, List[str]]:
    table_groups: Dict[str, List[str]] = {}

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            table.decompose()
            continue

        title = _get_table_title(table)
        table_groups.setdefault(title, [])

        first_row_raw = [
            _clean_cell(cell.get_text(separator=" ", strip=True))
            for cell in rows[0].find_all(["th", "td"])
        ]
        first_row_raw = _merge_dollar_fragments(first_row_raw)
        has_year_headers = any(YEAR_PATTERN.search(cell) for cell in first_row_raw)
        col_headers = first_row_raw if has_year_headers else []
        data_rows = rows[1:] if has_year_headers else rows

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

            if col_headers:
                for idx in range(1, len(all_cells)):
                    value = all_cells[idx].strip()
                    if not value:
                        continue
                    col_header = col_headers[idx].strip() if idx < len(col_headers) else ""
                    row_text = f"{label} ({col_header}): {value}" if col_header else f"{label}: {value}"
                    table_groups[title].append(_clean_row_string(row_text))
            else:
                row_text = non_empty[0] + ": " + ", ".join(non_empty[1:])
                table_groups[title].append(_clean_row_string(row_text))

        table.decompose()

    return {title: rows for title, rows in table_groups.items() if rows}


def _extract_prose_text(soup: BeautifulSoup) -> str:
    raw_text = soup.get_text(separator="\n")
    paragraphs: List[str] = []
    current_lines: List[str] = []

    for raw_line in raw_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or any(pattern.match(line) for pattern in SKIP_PATTERNS):
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

    return "\n\n".join(paragraphs)


def _looks_like_heading(line: str) -> bool:
    if len(line) > 150:
        return False
    if line.endswith(":"):
        return True
    return line.isupper() and any(char.isalpha() for char in line)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text.strip():
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        split_at = end

        if end < text_length:
            window = text[start:end]
            for separator in separators:
                if not separator:
                    continue
                split_index = window.rfind(separator)
                if split_index > chunk_size // 2:
                    split_at = start + split_index + len(separator)
                    break

        if split_at <= start:
            split_at = end

        chunk = text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)

        if split_at >= text_length:
            break

        start = max(split_at - overlap, start + 1)

    return chunks


def _build_table_chunks(
    table_groups: Dict[str, List[str]],
    table_window: int,
    table_overlap: int,
) -> List[str]:
    table_chunks: List[str] = []

    for title, rows in table_groups.items():
        index = 0
        while index < len(rows):
            window = rows[index:index + table_window]
            table_chunks.append(f"[{title}]\n" + "\n".join(window))
            if index + table_window >= len(rows):
                break
            index += max(table_window - table_overlap, 1)

    return table_chunks


def _clean_cell(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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
