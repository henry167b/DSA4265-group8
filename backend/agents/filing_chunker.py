import re
from typing import Dict, List, Optional

from bs4 import BeautifulSoup


ITEM_HEADING_RE = re.compile(r"^(item\s+\d+[a-z]?\b[\.\s])", re.IGNORECASE)
DATE_PATTERN = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)
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
    """Turn filing HTML into prose-only chunks for retrieval."""
    soup = BeautifulSoup(html_content, "lxml")

    for element in soup(["style", "script", "img", "svg", "canvas", "object", "embed", "table"]):
        element.decompose()

    for tag in soup.find_all(True):
        if ":" in (tag.name or ""):
            tag.unwrap()

    prose_paragraphs = _extract_prose_paragraphs(soup)
    prose_text = "\n\n".join(prose_paragraphs)
    prose_chunk_records = _build_prose_chunk_records(
        prose_paragraphs,
        prose_chunk_size,
        prose_chunk_overlap,
    )
    prose_chunks = [record["text"] for record in prose_chunk_records]

    return {
        "prose_text": prose_text,
        "table_groups": {},
        "table_structures": [],
        "prose_chunk_records": prose_chunk_records,
        "prose_chunks": prose_chunks,
        "table_chunk_records": [],
        "table_chunks": [],
        "chunks": prose_chunks,
    }
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
                "period_end": _infer_period_end(text),
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
