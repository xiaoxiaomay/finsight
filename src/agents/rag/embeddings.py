"""Embedding generation and management.

Uses sentence-transformers (all-MiniLM-L6-v2) for local embedding generation.
Falls back to a simple TF-IDF-based embedding if sentence-transformers is unavailable.

The model produces 384-dimensional vectors, suitable for pgvector storage.
"""

from __future__ import annotations

import hashlib

import numpy as np

from src.config.logging_config import get_logger

logger = get_logger("rag.embeddings")

# Default model — small but effective for financial text retrieval
DEFAULT_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class EmbeddingModel:
    """Wrapper around sentence-transformers for embedding generation.

    Lazily loads the model on first use.
    Falls back to deterministic hash-based embeddings if the model is unavailable.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._use_fallback = False

    def _load_model(self) -> None:
        """Lazily load the sentence-transformer model."""
        if self._model is not None or self._use_fallback:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("embedding_model_loaded", model=self.model_name)
        except ImportError:
            logger.warning(
                "sentence_transformers_not_available",
                fallback="hash_based_embeddings",
            )
            self._use_fallback = True
        except Exception:
            logger.warning("embedding_model_load_failed", model=self.model_name)
            self._use_fallback = True

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (len(texts), EMBEDDING_DIM).
        """
        self._load_model()

        if self._use_fallback:
            return self._fallback_embed(texts)

        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Returns:
            numpy array of shape (EMBEDDING_DIM,).
        """
        result = self.embed([text])
        return result[0]

    def _fallback_embed(self, texts: list[str]) -> np.ndarray:
        """Deterministic hash-based embedding fallback.

        Not suitable for production but allows code to run without torch.
        """
        embeddings = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            # Use SHA-256 hash to create deterministic pseudo-embedding
            h = hashlib.sha256(text.encode()).digest()
            # Expand to EMBEDDING_DIM by repeating hash
            raw = np.frombuffer(h * (EMBEDDING_DIM // 32 + 1), dtype=np.uint8)[:EMBEDDING_DIM]
            vec = raw.astype(np.float32) / 255.0 - 0.5
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings[i] = vec
        return embeddings

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return EMBEDDING_DIM


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vectors.

    Args:
        a: Query vector(s) of shape (d,) or (n, d).
        b: Document vectors of shape (m, d).

    Returns:
        Similarity scores of shape (n, m) or (m,).
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)

    # Normalize
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)

    sims = a_norm @ b_norm.T
    return sims.squeeze()
