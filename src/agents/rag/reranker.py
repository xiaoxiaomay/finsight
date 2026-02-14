"""Cross-encoder reranking for improved retrieval precision.

Uses a cross-encoder model (ms-marco-MiniLM-L-6-v2) to re-score
(query, document) pairs after initial retrieval.

Cross-encoders are more accurate than bi-encoders for reranking
because they see both query and document together, but are slower
(hence used only on the top-k candidates from initial retrieval).
"""

from __future__ import annotations

from src.agents.rag.retriever import RetrievalResult
from src.config.logging_config import get_logger

logger = get_logger("rag.reranker")

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Cross-encoder reranker for retrieval results.

    Lazily loads model on first use. Falls back to identity (no reranking)
    if model is unavailable.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER) -> None:
        self.model_name = model_name
        self._model = None
        self._available = True

    def _load_model(self) -> None:
        if self._model is not None or not self._available:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info("reranker_loaded", model=self.model_name)
        except (ImportError, Exception):
            logger.warning("reranker_not_available", fallback="no_reranking")
            self._available = False

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results using cross-encoder.

        Args:
            query: Original search query.
            results: Initial retrieval results to rerank.
            top_k: Number of results to return (None = all).

        Returns:
            Reranked list of RetrievalResult objects.
        """
        if not results:
            return results

        self._load_model()

        if not self._available or self._model is None:
            return results[:top_k] if top_k else results

        # Score all (query, document) pairs
        pairs = [(query, r.chunk.text) for r in results]
        scores = self._model.predict(pairs)

        # Update scores and sort
        for result, score in zip(results, scores, strict=True):
            result.score = float(score)

        results.sort(key=lambda r: r.score, reverse=True)

        if top_k:
            results = results[:top_k]

        return results
