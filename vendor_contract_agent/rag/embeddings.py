from __future__ import annotations

import hashlib
import logging
import threading
from typing import Iterable, Sequence

from .config import RAGConfig

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Reusable embedding service with lazy model loading and caching."""

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig.from_env()
        self._model = None
        self._lock = threading.Lock()
        self._cache: dict[str, list[float]] = {}
        self._backend = self.config.embedding_backend

    def embed_text(self, text: str, *, is_query: bool = False) -> list[float]:
        texts = [text]
        return self.embed_batch(texts, is_query=is_query)[0]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text, is_query=True)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_batch(texts, is_query=False)

    def embed_batch(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        backend = self._resolve_backend()
        results: list[list[float] | None] = [None] * len(texts)
        missing: list[tuple[int, str]] = []
        for index, text in enumerate(texts):
            key = self._cache_key(text, is_query=is_query, backend=backend)
            cached = self._cache.get(key)
            if cached is not None:
                results[index] = cached
            else:
                missing.append((index, text))

        if missing:
            vectors = self._encode([text for _, text in missing], is_query=is_query, backend=backend)
            for (index, text), vector in zip(missing, vectors):
                key = self._cache_key(text, is_query=is_query, backend=backend)
                self._cache[key] = vector
                results[index] = vector

        return [vector or [0.0] * self.config.embedding_dim for vector in results]

    def _resolve_backend(self) -> str:
        if self._backend != "auto":
            return self._backend
        try:
            import FlagEmbedding  # noqa: F401

            return "bge"
        except Exception:
            try:
                import sentence_transformers  # noqa: F401

                return "bge"
            except Exception:
                logger.warning("BGE-M3 unavailable; using deterministic hash embeddings")
                return "hash"

    def _encode(self, texts: Sequence[str], *, is_query: bool, backend: str) -> list[list[float]]:
        if backend == "hash":
            return [_hash_embedding(text, self.config.embedding_dim) for text in texts]
        model = self._load_model()
        # FlagEmbedding BGEM3FlagModel
        if hasattr(model, "encode"):
            try:
                output = model.encode(
                    list(texts),
                    batch_size=16,
                    max_length=8192,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
                dense = output["dense_vecs"] if isinstance(output, dict) else output
                return [list(map(float, row)) for row in dense]
            except TypeError:
                # sentence-transformers path
                vectors = model.encode(
                    list(texts),
                    normalize_embeddings=True,
                    prompt_name="query" if is_query else None,
                )
                return [list(map(float, row)) for row in vectors]
        raise RuntimeError("Unsupported embedding model interface")

    def _load_model(self):
        with self._lock:
            if self._model is not None:
                return self._model
            logger.info("Loading embedding model %s", self.config.embedding_model)
            try:
                from FlagEmbedding import BGEM3FlagModel

                self._model = BGEM3FlagModel(self.config.embedding_model, use_fp16=True)
            except Exception:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.config.embedding_model)
            return self._model

    @staticmethod
    def _cache_key(text: str, *, is_query: bool, backend: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{backend}:{'q' if is_query else 'd'}:{digest}"


def _hash_embedding(text: str, dim: int) -> list[float]:
    """Deterministic pseudo-embedding for tests/offline mode."""
    values = [0.0] * dim
    tokens = text.lower().split() or [text]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i in range(0, min(len(digest), 32)):
            idx = (digest[i] + i * 17) % dim
            values[idx] += ((digest[i] / 255.0) * 2.0) - 1.0
    # L2 normalize
    norm = sum(v * v for v in values) ** 0.5 or 1.0
    return [v / norm for v in values]
