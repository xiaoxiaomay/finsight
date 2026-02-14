"""Hybrid retrieval: semantic search + BM25 + metadata filtering.

Combines multiple retrieval strategies for robust document retrieval:
1. Semantic search — cosine similarity on embeddings (good for meaning)
2. BM25 — keyword matching (good for tickers, numbers, exact terms)
3. Metadata filtering — symbol, date range, section name
4. Score fusion — Reciprocal Rank Fusion (RRF) to combine rankings
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.agents.rag.chunking import Chunk
from src.agents.rag.embeddings import EmbeddingModel, cosine_similarity
from src.config.logging_config import get_logger

logger = get_logger("rag.retriever")


@dataclass
class RetrievalResult:
    """A single retrieval result with score and metadata."""

    chunk: Chunk
    score: float
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    source: str = ""  # 'semantic', 'bm25', 'hybrid'


@dataclass
class InMemoryVectorStore:
    """In-memory vector store for development and testing.

    For production, swap with PgVectorStore.
    """

    chunks: list[Chunk] = field(default_factory=list)
    embeddings: np.ndarray | None = None

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the store."""
        new_embeddings = []
        for chunk in chunks:
            emb = chunk.metadata.get("embedding")
            if emb is not None:
                new_embeddings.append(emb)
                self.chunks.append(chunk)

        if new_embeddings:
            new_arr = np.array(new_embeddings, dtype=np.float32)
            if self.embeddings is None:
                self.embeddings = new_arr
            else:
                self.embeddings = np.vstack([self.embeddings, new_arr])

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_fn: callable | None = None,
    ) -> list[tuple[int, float]]:
        """Semantic search over stored embeddings.

        Returns:
            List of (index, similarity_score) tuples.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        scores = cosine_similarity(query_embedding, self.embeddings)

        # Apply metadata filter
        if filter_fn is not None:
            for i, chunk in enumerate(self.chunks):
                if not filter_fn(chunk.metadata):
                    scores[i] = -1.0

        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > -1.0]

    @property
    def size(self) -> int:
        return len(self.chunks)


class BM25Index:
    """BM25 keyword index for document retrieval."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25 = None

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the BM25 index."""
        self._chunks.extend(chunks)
        self._rebuild()

    def _rebuild(self) -> None:
        """Rebuild BM25 index from all chunks."""
        if not self._chunks:
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25_not_installed", fallback="disabled")
            return

        tokenized = [self._tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_fn: callable | None = None,
    ) -> list[tuple[int, float]]:
        """BM25 keyword search.

        Returns:
            List of (index, bm25_score) tuples.
        """
        if self._bm25 is None or not self._chunks:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        if filter_fn is not None:
            for i, chunk in enumerate(self._chunks):
                if not filter_fn(chunk.metadata):
                    scores[i] = 0.0

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercasing tokenizer."""
        return text.lower().split()

    @property
    def size(self) -> int:
        return len(self._chunks)


class HybridRetriever:
    """Hybrid retriever combining semantic search, BM25, and metadata filtering.

    Uses Reciprocal Rank Fusion (RRF) to combine results from different
    retrieval methods into a single ranked list.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> None:
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = InMemoryVectorStore()
        self.bm25_index = BM25Index()
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to both semantic and BM25 indexes.

        Args:
            chunks: List of Chunk objects (should have embeddings in metadata).
        """
        self.vector_store.add(chunks)
        self.bm25_index.add(chunks)
        logger.info(
            "chunks_indexed",
            semantic_size=self.vector_store.size,
            bm25_size=self.bm25_index.size,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        symbol: str | None = None,
        section: str | None = None,
        document_type: str | None = None,
    ) -> list[RetrievalResult]:
        """Hybrid retrieval combining semantic and BM25 search.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            symbol: Optional symbol filter.
            section: Optional section name filter.
            document_type: Optional document type filter.

        Returns:
            List of RetrievalResult objects, ranked by fused score.
        """
        # Build metadata filter
        filter_fn = self._build_filter(symbol, section, document_type)

        # Semantic search
        query_emb = self.embedding_model.embed_single(query)
        semantic_results = self.vector_store.search(
            query_emb, top_k=top_k * 2, filter_fn=filter_fn
        )

        # BM25 search
        bm25_results = self.bm25_index.search(
            query, top_k=top_k * 2, filter_fn=filter_fn
        )

        # Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(
            semantic_results, bm25_results, k=60
        )

        # Build results
        results = []
        for idx, score, sem_score, bm25_score in fused[:top_k]:
            chunk = self.vector_store.chunks[idx]
            results.append(RetrievalResult(
                chunk=chunk,
                score=score,
                semantic_score=sem_score,
                bm25_score=bm25_score,
                source="hybrid",
            ))

        return results

    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[tuple[int, float]],
        bm25_results: list[tuple[int, float]],
        k: int = 60,
    ) -> list[tuple[int, float, float, float]]:
        """Combine rankings using Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (k + rank_i)) across all result lists.

        Returns:
            List of (index, fused_score, semantic_score, bm25_score).
        """
        scores: dict[int, dict] = {}

        # Process semantic results
        for rank, (idx, sim) in enumerate(semantic_results):
            if idx not in scores:
                scores[idx] = {"rrf": 0.0, "semantic": 0.0, "bm25": 0.0}
            scores[idx]["rrf"] += self.semantic_weight / (k + rank + 1)
            scores[idx]["semantic"] = sim

        # Process BM25 results
        for rank, (idx, bm25_score) in enumerate(bm25_results):
            if idx not in scores:
                scores[idx] = {"rrf": 0.0, "semantic": 0.0, "bm25": 0.0}
            scores[idx]["rrf"] += self.bm25_weight / (k + rank + 1)
            scores[idx]["bm25"] = bm25_score

        # Sort by RRF score
        ranked = sorted(scores.items(), key=lambda x: x[1]["rrf"], reverse=True)
        return [
            (idx, s["rrf"], s["semantic"], s["bm25"])
            for idx, s in ranked
        ]

    @staticmethod
    def _build_filter(
        symbol: str | None,
        section: str | None,
        document_type: str | None,
    ) -> callable | None:
        """Build a metadata filter function."""
        if symbol is None and section is None and document_type is None:
            return None

        def filter_fn(meta: dict) -> bool:
            if symbol and meta.get("symbol", "").upper() != symbol.upper():
                return False
            if section and meta.get("section_name", "").lower() != section.lower():
                return False
            return not (document_type and meta.get("document_type", "") != document_type)

        return filter_fn

    @property
    def total_chunks(self) -> int:
        return self.vector_store.size
