"""Section-aware document chunking for financial filings.

Splits documents into chunks that:
- Respect section boundaries (never split across sections)
- Keep tables as complete chunks (tables are critical in financial docs)
- Target ~1000 tokens per chunk with ~200 token overlap
- Attach rich metadata: symbol, date, section_name, chunk_index
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single document chunk with metadata."""

    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        """Approximate token count (words * 1.3)."""
        return int(len(self.text.split()) * 1.3)


def estimate_tokens(text: str) -> int:
    """Estimate token count for text (approximate: words × 1.3)."""
    return int(len(text.split()) * 1.3)


def chunk_document(
    text: str,
    sections: dict[str, str] | None = None,
    tables: list[str] | None = None,
    symbol: str = "",
    document_date: str = "",
    document_type: str = "",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Split document into chunks with section awareness.

    Args:
        text: Full document text.
        sections: Dict mapping section_name → section_text.
        tables: List of table texts to keep intact.
        symbol: Stock ticker for metadata.
        document_date: Filing date for metadata.
        document_type: Filing type (e.g., '10-K') for metadata.
        chunk_size: Target tokens per chunk.
        chunk_overlap: Token overlap between chunks.

    Returns:
        List of Chunk objects with metadata.
    """
    chunks: list[Chunk] = []
    base_meta = {
        "symbol": symbol,
        "document_date": document_date,
        "document_type": document_type,
    }

    # If we have sections, chunk each section separately
    if sections:
        for section_name, section_text in sections.items():
            section_chunks = _split_text(
                section_text, chunk_size, chunk_overlap
            )
            for i, chunk_text in enumerate(section_chunks):
                meta = {
                    **base_meta,
                    "section_name": section_name,
                    "chunk_index": len(chunks),
                    "section_chunk_index": i,
                }
                chunks.append(Chunk(text=chunk_text, metadata=meta))
    else:
        # No sections detected — chunk the whole text
        text_chunks = _split_text(text, chunk_size, chunk_overlap)
        for i, chunk_text in enumerate(text_chunks):
            meta = {
                **base_meta,
                "section_name": "full_document",
                "chunk_index": i,
            }
            chunks.append(Chunk(text=chunk_text, metadata=meta))

    # Add tables as separate complete chunks
    if tables:
        for i, table_text in enumerate(tables):
            if estimate_tokens(table_text) > 20:
                meta = {
                    **base_meta,
                    "section_name": "table",
                    "chunk_index": len(chunks),
                    "table_index": i,
                }
                chunks.append(Chunk(text=table_text, metadata=meta))

    return chunks


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split text into chunks respecting sentence boundaries.

    Uses a sliding window approach with paragraph/sentence boundaries.
    """
    if estimate_tokens(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Split by paragraphs first, then by sentences if paragraphs are too large
    paragraphs = _split_paragraphs(text)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        if para_tokens > chunk_size:
            # Paragraph too large — flush current and split paragraph by sentences
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                # Keep overlap from end
                current_parts, current_tokens = _keep_overlap(
                    current_parts, chunk_overlap
                )

            sentence_chunks = _split_by_sentences(para, chunk_size, chunk_overlap)
            chunks.extend(sentence_chunks)
            current_parts = []
            current_tokens = 0
            continue

        if current_tokens + para_tokens > chunk_size and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts, current_tokens = _keep_overlap(
                current_parts, chunk_overlap
            )

        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return [c.strip() for c in chunks if c.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs."""
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _split_by_sentences(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split text by sentences for very long paragraphs."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = estimate_tokens(sent)

        if current_tokens + sent_tokens > chunk_size and current:
            chunks.append(" ".join(current))
            # Keep overlap sentences
            overlap_tokens = 0
            overlap_parts: list[str] = []
            for s in reversed(current):
                t = estimate_tokens(s)
                if overlap_tokens + t > chunk_overlap:
                    break
                overlap_parts.insert(0, s)
                overlap_tokens += t
            current = overlap_parts
            current_tokens = overlap_tokens

        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def _keep_overlap(
    parts: list[str],
    overlap_tokens: int,
) -> tuple[list[str], int]:
    """Keep trailing parts that fit within overlap budget."""
    kept: list[str] = []
    tokens = 0

    for part in reversed(parts):
        t = estimate_tokens(part)
        if tokens + t > overlap_tokens:
            break
        kept.insert(0, part)
        tokens += t

    return kept, tokens
