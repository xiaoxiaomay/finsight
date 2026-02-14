"""SEC filing document processor.

Orchestrates the full RAG ingestion pipeline:
1. Download filing from SEC EDGAR
2. Parse HTML → clean text with section detection
3. Section-aware chunking
4. Generate embeddings
5. Index for retrieval

This module connects sec_edgar tool → chunking → embeddings → retriever.
"""

from __future__ import annotations

from src.agents.rag.chunking import Chunk, chunk_document
from src.agents.rag.embeddings import EmbeddingModel
from src.agents.tools.sec_edgar import (
    FilingDocument,
    download_filing,
    get_filings,
)
from src.config.logging_config import get_logger

logger = get_logger("rag.document_processor")


class DocumentProcessor:
    """Process SEC filings into indexed chunks for RAG retrieval."""

    def __init__(self, embedding_model: EmbeddingModel | None = None) -> None:
        self.embedding_model = embedding_model or EmbeddingModel()

    def process_filing(
        self,
        symbol: str,
        form_type: str = "10-K",
        filing_index: int = 0,
    ) -> list[Chunk]:
        """Download and process a SEC filing for a symbol.

        Args:
            symbol: Stock ticker (e.g., 'AAPL').
            form_type: Filing type ('10-K', '10-Q').
            filing_index: Which filing to get (0=most recent).

        Returns:
            List of processed chunks with embeddings.
        """
        # Step 1: Get filing metadata
        filings = get_filings(symbol, form_type=form_type, count=filing_index + 1)
        if not filings or filing_index >= len(filings):
            logger.warning("no_filing_found", symbol=symbol, form_type=form_type)
            return []

        filing = filings[filing_index]

        # Step 2: Download and parse
        doc = download_filing(filing)
        if not doc.full_text:
            logger.warning("empty_filing", accession=filing.accession_number)
            return []

        # Step 3: Chunk
        chunks = self.chunk_filing(doc)

        # Step 4: Generate embeddings
        self.embed_chunks(chunks)

        logger.info(
            "filing_processed",
            symbol=symbol,
            form_type=form_type,
            filing_date=filing.filing_date,
            chunks=len(chunks),
        )

        return chunks

    def chunk_filing(self, doc: FilingDocument) -> list[Chunk]:
        """Chunk a downloaded filing document.

        Args:
            doc: Parsed filing document.

        Returns:
            List of Chunk objects with metadata.
        """
        return chunk_document(
            text=doc.full_text,
            sections=doc.sections,
            tables=doc.tables,
            symbol=doc.filing.cik,
            document_date=doc.filing.filing_date,
            document_type=doc.filing.form_type,
        )

    def process_text(
        self,
        text: str,
        symbol: str = "",
        document_date: str = "",
        document_type: str = "",
        sections: dict[str, str] | None = None,
    ) -> list[Chunk]:
        """Process raw text directly (for testing or non-EDGAR documents).

        Args:
            text: Document text to process.
            symbol: Stock ticker for metadata.
            document_date: Document date for metadata.
            document_type: Document type for metadata.
            sections: Optional pre-detected sections.

        Returns:
            List of processed chunks with embeddings.
        """
        chunks = chunk_document(
            text=text,
            sections=sections,
            symbol=symbol,
            document_date=document_date,
            document_type=document_type,
        )

        self.embed_chunks(chunks)
        return chunks

    def embed_chunks(self, chunks: list[Chunk]) -> None:
        """Generate embeddings for a list of chunks (in-place).

        Adds 'embedding' key to each chunk's metadata.
        """
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedding_model.embed(texts)

        for chunk, emb in zip(chunks, embeddings, strict=True):
            chunk.metadata["embedding"] = emb
