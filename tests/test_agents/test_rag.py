"""Tests for RAG pipeline: chunking, embeddings, retriever."""

import numpy as np

from src.agents.rag.chunking import Chunk, chunk_document, estimate_tokens


class TestChunking:
    """Test document chunking logic."""

    def test_estimate_tokens(self) -> None:
        """Token estimation should be approximately 1.3x word count."""
        text = "This is a simple test sentence with eight words."
        tokens = estimate_tokens(text)
        assert 10 <= tokens <= 15

    def test_short_text_single_chunk(self) -> None:
        """Short text should produce a single chunk."""
        text = "This is a short document about Apple Inc."
        chunks = chunk_document(text, symbol="AAPL", document_date="2024-01-01")

        assert len(chunks) == 1
        assert chunks[0].metadata["symbol"] == "AAPL"
        assert "Apple" in chunks[0].text

    def test_section_aware_chunking(self) -> None:
        """Chunks should respect section boundaries."""
        sections = {
            "Risk Factors": "Risk factor 1. " * 50,
            "MD&A": "Management discussion point. " * 50,
        }

        chunks = chunk_document(
            text="",
            sections=sections,
            symbol="MSFT",
            document_type="10-K",
        )

        assert len(chunks) >= 2
        section_names = {c.metadata["section_name"] for c in chunks}
        assert "Risk Factors" in section_names
        assert "MD&A" in section_names

    def test_tables_as_complete_chunks(self) -> None:
        """Tables should be kept as complete chunks."""
        text = "Some document text."
        tables = [
            "Revenue | 2023 | 2024\nProduct A | 100M | 120M\nProduct B | 50M | 60M"
        ]

        chunks = chunk_document(text, tables=tables, symbol="GOOG")

        table_chunks = [c for c in chunks if c.metadata.get("section_name") == "table"]
        assert len(table_chunks) == 1
        assert "Revenue" in table_chunks[0].text

    def test_chunk_metadata(self) -> None:
        """Each chunk should have proper metadata."""
        chunks = chunk_document(
            text="Some test content.",
            symbol="AAPL",
            document_date="2024-03-15",
            document_type="10-K",
        )

        for chunk in chunks:
            assert "symbol" in chunk.metadata
            assert "document_date" in chunk.metadata
            assert "document_type" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_long_text_multiple_chunks(self) -> None:
        """Long text should be split into multiple chunks."""
        text = "This is a paragraph about financial performance. " * 500
        chunks = chunk_document(text, chunk_size=1000, chunk_overlap=200)

        assert len(chunks) >= 3

    def test_empty_text(self) -> None:
        """Empty text should produce no chunks."""
        chunks = chunk_document("")
        assert len(chunks) == 0


class TestEmbeddings:
    """Test embedding generation."""

    def test_embed_produces_correct_shape(self) -> None:
        """Embeddings should be (n, 384) shaped arrays."""
        from src.agents.rag.embeddings import EmbeddingModel

        model = EmbeddingModel()
        texts = ["Hello world", "Financial analysis of Apple"]
        embeddings = model.embed(texts)

        assert embeddings.shape == (2, 384)

    def test_embed_single(self) -> None:
        """Single embedding should be (384,) shaped."""
        from src.agents.rag.embeddings import EmbeddingModel

        model = EmbeddingModel()
        embedding = model.embed_single("test text")

        assert embedding.shape == (384,)

    def test_embeddings_normalized(self) -> None:
        """Embeddings should be approximately unit-normalized."""
        from src.agents.rag.embeddings import EmbeddingModel

        model = EmbeddingModel()
        embedding = model.embed_single("financial report analysis")

        norm = np.linalg.norm(embedding)
        assert 0.9 < norm < 1.1

    def test_cosine_similarity(self) -> None:
        """Cosine similarity should work correctly."""
        from src.agents.rag.embeddings import cosine_similarity

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        sims = cosine_similarity(a, b)

        assert abs(sims[0] - 1.0) < 0.01
        assert abs(sims[1] - 0.0) < 0.01

    def test_different_texts_different_embeddings(self) -> None:
        """Different texts should produce different embeddings."""
        from src.agents.rag.embeddings import EmbeddingModel

        model = EmbeddingModel()
        emb1 = model.embed_single("Apple revenue growth")
        emb2 = model.embed_single("Federal Reserve interest rates")

        diff = np.abs(emb1 - emb2).sum()
        assert diff > 0.1


class TestRetriever:
    """Test hybrid retriever."""

    def _make_retriever(self):
        """Create a retriever with test data."""
        from src.agents.rag.embeddings import EmbeddingModel
        from src.agents.rag.retriever import HybridRetriever

        retriever = HybridRetriever(embedding_model=EmbeddingModel())

        texts = [
            "Apple Inc reported revenue of $94.8 billion in Q1 2024.",
            "The Federal Reserve held interest rates steady at 5.25-5.50%.",
            "Apple's gross margin expanded to 46.6% driven by services growth.",
            "Microsoft cloud revenue grew 28% year over year.",
            "Risk factors include supply chain disruptions in Asia.",
        ]

        chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                text=text,
                metadata={
                    "symbol": "AAPL" if "Apple" in text else "MSFT" if "Microsoft" in text else "",
                    "section_name": "Risk Factors" if "Risk" in text else "MD&A",
                    "chunk_index": i,
                },
            )
            chunks.append(chunk)

        embeddings = retriever.embedding_model.embed([c.text for c in chunks])
        for chunk, emb in zip(chunks, embeddings, strict=True):
            chunk.metadata["embedding"] = emb

        retriever.index_chunks(chunks)
        return retriever

    def test_retriever_import(self) -> None:
        """HybridRetriever should be importable."""
        from src.agents.rag.retriever import HybridRetriever

        assert HybridRetriever is not None

    def test_index_chunks(self) -> None:
        """Should index chunks correctly."""
        retriever = self._make_retriever()
        assert retriever.total_chunks == 5

    def test_semantic_search(self) -> None:
        """Should retrieve relevant chunks by meaning."""
        retriever = self._make_retriever()
        results = retriever.retrieve("Apple revenue earnings", top_k=3)

        assert len(results) > 0
        assert len(results) <= 3
        assert all(r.score > 0 for r in results)

    def test_metadata_filter(self) -> None:
        """Should filter by symbol."""
        retriever = self._make_retriever()
        results = retriever.retrieve("revenue growth", top_k=5, symbol="AAPL")

        for r in results:
            assert r.chunk.metadata.get("symbol") == "AAPL"

    def test_section_filter(self) -> None:
        """Should filter by section name."""
        retriever = self._make_retriever()
        results = retriever.retrieve("risk", top_k=5, section="Risk Factors")

        for r in results:
            assert r.chunk.metadata.get("section_name") == "Risk Factors"

    def test_empty_retriever(self) -> None:
        """Empty retriever should return no results."""
        from src.agents.rag.embeddings import EmbeddingModel
        from src.agents.rag.retriever import HybridRetriever

        retriever = HybridRetriever(embedding_model=EmbeddingModel())
        results = retriever.retrieve("test query")

        assert results == []


class TestSECEdgar:
    """Test SEC EDGAR parsing (offline tests)."""

    def test_detect_sections(self) -> None:
        """Should detect standard 10-K sections."""
        from src.agents.tools.sec_edgar import detect_sections

        text = (
            "Some preamble text.\n"
            "Item 1. Business\n"
            + "Description of the company's business operations. " * 20
            + "\nItem 1A. Risk Factors\n"
            + "The following risk factors could affect results. " * 20
            + "\nItem 7. Management's Discussion and Analysis\n"
            + "Revenue increased due to strong demand. " * 20
        )

        sections = detect_sections(text)

        assert "Business" in sections
        assert "Risk Factors" in sections
        assert "MD&A" in sections

    def test_parse_filing_html(self) -> None:
        """Should parse HTML into clean text."""
        from src.agents.tools.sec_edgar import parse_filing_html

        html = """
        <html><body>
        <p>Apple Inc. 10-K Annual Report</p>
        <table><tr><td>Revenue</td><td>$394B</td></tr></table>
        <script>alert('xss')</script>
        </body></html>
        """

        text, tables = parse_filing_html(html)

        assert "Apple Inc" in text
        assert "alert" not in text
        assert len(tables) >= 1
        assert "Revenue" in tables[0]

    def test_parse_table(self) -> None:
        """Should convert HTML table to readable text."""
        from bs4 import BeautifulSoup

        from src.agents.tools.sec_edgar import _parse_table

        html = "<table><tr><td>Item</td><td>Value</td></tr><tr><td>Revenue</td><td>100M</td></tr></table>"
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        result = _parse_table(table)

        assert "Revenue" in result
        assert "100M" in result


class TestDocumentProcessor:
    """Test document processor."""

    def test_process_text(self) -> None:
        """Should process raw text into chunks with embeddings."""
        from src.agents.rag.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        chunks = processor.process_text(
            text="Apple revenue was strong in 2024. " * 10,
            symbol="AAPL",
            document_date="2024-03-15",
            document_type="10-K",
        )

        assert len(chunks) >= 1
        assert chunks[0].metadata["symbol"] == "AAPL"
        assert "embedding" in chunks[0].metadata
        assert chunks[0].metadata["embedding"].shape == (384,)
