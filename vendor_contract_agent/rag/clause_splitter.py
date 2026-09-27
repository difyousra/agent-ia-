from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


CLAUSE_HEADERS = [
    r"^\s*(\d+(?:\.\d+)*)\s*[\.\):-]\s+(.+)$",
    r"^\s*ARTICLE\s+(\d+(?:\.\d+)*)\s*[\.\):-]?\s*(.*)$",
    r"^\s*SECTION\s+(\d+(?:\.\d+)*)\s*[\.\):-]?\s*(.*)$",
    r"^\s*CLAUSE\s+(\d+(?:\.\d+)*)\s*[\.\):-]?\s*(.*)$",
]

CLAUSE_TYPE_KEYWORDS = {
    "termination": "Termination",
    "renewal": "Renewal",
    "payment": "Payment",
    "sla": "SLA",
    "service level": "SLA",
    "confidential": "Confidentiality",
    "liability": "Liability",
    "indemn": "Indemnification",
    "warranty": "Warranty",
    "insurance": "Insurance",
    "intellectual property": "Intellectual Property",
    "ip ownership": "Intellectual Property",
    "data protection": "Data Protection",
    "privacy": "Data Protection",
    "security": "Security",
    "force majeure": "Force Majeure",
    "subcontract": "Subcontracting",
    "audit": "Audit",
    "governing law": "Governing Law",
    "dispute": "Dispute Resolution",
    "notice": "Notices",
    "penalty": "Penalties",
    "liquidated": "Penalties",
    "delivery": "Delivery",
    "scope": "Scope",
    "parties": "Parties",
}


@dataclass
class ClauseChunk:
    content: str
    clause_number: str | None = None
    clause_type: str | None = None
    section: str | None = None
    page: int | None = None


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detect_clause_type(text: str) -> str | None:
    lowered = text.lower()
    for keyword, label in CLAUSE_TYPE_KEYWORDS.items():
        if keyword in lowered:
            return label
    return None


def split_into_clauses(text: str) -> list[ClauseChunk]:
    """Prefer clause-level chunks; fall back to paragraph chunks."""
    pages = _split_pages(text)
    chunks: list[ClauseChunk] = []
    for page_number, page_text in pages:
        page_chunks = _split_page(page_text, page_number)
        chunks.extend(page_chunks)

    cleaned = [chunk for chunk in chunks if chunk.content and len(chunk.content.strip()) >= 40]
    if cleaned:
        return cleaned

    # Fallback: paragraph chunks
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [
        ClauseChunk(
            content=paragraph,
            clause_type=detect_clause_type(paragraph),
            page=None,
        )
        for paragraph in paragraphs
        if len(paragraph) >= 40
    ]


def _split_pages(text: str) -> list[tuple[int | None, str]]:
    page_pattern = re.compile(r"(?:---\s*PAGE\s+(\d+)\s*---|^--\s*(\d+)\s+of\s+\d+\s*--)", re.I | re.M)
    matches = list(page_pattern.finditer(text))
    if not matches:
        return [(None, text)]
    pages: list[tuple[int | None, str]] = []
    for index, match in enumerate(matches):
        page_num = int(match.group(1) or match.group(2))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        pages.append((page_num, text[start:end]))
    # Keep preamble before first page marker if substantial.
    preamble = text[: matches[0].start()].strip()
    if len(preamble) >= 40:
        pages.insert(0, (1, preamble))
    return pages


def _split_page(page_text: str, page_number: int | None) -> list[ClauseChunk]:
    lines = page_text.splitlines()
    compiled = [re.compile(pattern, re.I) for pattern in CLAUSE_HEADERS]
    starts: list[tuple[int, str, str]] = []
    for idx, line in enumerate(lines):
        for pattern in compiled:
            match = pattern.match(line.strip())
            if match:
                number = match.group(1).strip()
                title = (match.group(2) or "").strip()
                starts.append((idx, number, title))
                break

    if not starts:
        content = page_text.strip()
        if not content:
            return []
        return [
            ClauseChunk(
                content=content,
                clause_type=detect_clause_type(content),
                page=page_number,
            )
        ]

    chunks: list[ClauseChunk] = []
    for i, (line_idx, number, title) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(lines)
        body = "\n".join(lines[line_idx:end]).strip()
        section = title or None
        clause_type = detect_clause_type(body) or detect_clause_type(title)
        chunks.append(
            ClauseChunk(
                content=body,
                clause_number=number,
                clause_type=clause_type,
                section=section,
                page=page_number,
            )
        )
    return chunks
