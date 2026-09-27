from __future__ import annotations

import logging
import math
import re
import threading
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from .config import RAGConfig
from .schemas import ChunkMetadata, RetrievedChunk

logger = logging.getLogger(__name__)


class InMemoryChunkStore:
    """In-memory store used for unit tests."""

    def __init__(self, embedding_dim: int = 1024) -> None:
        self.embedding_dim = embedding_dim
        self._chunks: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def ensure_schema(self) -> None:
        return None

    def close(self) -> None:
        return None

    def delete_contract(self, contract_id: str) -> int:
        with self._lock:
            to_delete = [cid for cid, row in self._chunks.items() if row["contract_id"] == contract_id]
            for cid in to_delete:
                del self._chunks[cid]
            return len(to_delete)

    def upsert_chunks(
        self,
        chunks: Sequence[ChunkMetadata],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        with self._lock:
            written = 0
            for chunk, embedding in zip(chunks, embeddings):
                content_hash = chunk.content_hash or ""
                existing_id = None
                for cid, row in self._chunks.items():
                    if row["contract_id"] == chunk.contract_id and row["content_hash"] == content_hash:
                        existing_id = cid
                        break
                chunk_id = existing_id or chunk.id or str(uuid4())
                self._chunks[chunk_id] = {
                    "id": chunk_id,
                    "contract_id": chunk.contract_id,
                    "document_name": chunk.document_name,
                    "clause_number": chunk.clause_number,
                    "clause_type": chunk.clause_type,
                    "section": chunk.section,
                    "page": chunk.page,
                    "content": chunk.content,
                    "content_hash": content_hash,
                    "embedding": list(embedding),
                }
                written += 1
            return written

    def semantic_search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 10,
        contract_id: str | None = None,
    ) -> list[RetrievedChunk]:
        rows = list(self._chunks.values())
        if contract_id:
            rows = [r for r in rows if r["contract_id"] == contract_id]
        scored = [
            _row_to_chunk(row, _cosine_similarity(query_embedding, row["embedding"]), "semantic")
            for row in rows
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        contract_id: str | None = None,
    ) -> list[RetrievedChunk]:
        return _keyword_over_rows(list(self._chunks.values()), query, top_k, contract_id)

    def contract_exists(self, contract_id: str) -> bool:
        return any(row["contract_id"] == contract_id for row in self._chunks.values())

    def count(self, contract_id: str | None = None) -> int:
        if not contract_id:
            return len(self._chunks)
        return sum(1 for row in self._chunks.values() if row["contract_id"] == contract_id)

    def list_contract_ids(self) -> list[str]:
        return sorted({row["contract_id"] for row in self._chunks.values()})


class ChromaChunkStore:
    """Persistent ChromaDB vector store for contract clause chunks."""

    def __init__(self, config: RAGConfig | None = None, persist_directory: str | None = None) -> None:
        self.config = config or RAGConfig.from_env()
        self.persist_directory = str(
            Path(persist_directory or self.config.chroma_path).expanduser().resolve()
        )
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is required. Install with: pip install chromadb"
            ) from exc

        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.config.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Chroma store ready path=%s collection=%s count=%s",
            self.persist_directory,
            self.config.chroma_collection,
            self._collection.count(),
        )

    def ensure_schema(self) -> None:
        return None

    def close(self) -> None:
        return None

    def delete_contract(self, contract_id: str) -> int:
        before = self.count(contract_id)
        if before == 0:
            return 0
        self._collection.delete(where={"contract_id": contract_id})
        return before

    def count(self, contract_id: str | None = None) -> int:
        if not contract_id:
            return int(self._collection.count())
        # Avoid loading all ids when possible.
        try:
            result = self._collection.get(where={"contract_id": contract_id}, include=[])
            return len(result.get("ids") or [])
        except Exception:
            return 0

    def upsert_chunks(
        self,
        chunks: Sequence[ChunkMetadata],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        vectors: list[list[float]] = []
        seen_ids: set[str] = set()
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            base_id = chunk.content_hash or chunk.id or str(uuid4())
            chunk_id = base_id
            suffix = 1
            while chunk_id in seen_ids:
                chunk_id = f"{base_id}-{suffix}"
                suffix += 1
            seen_ids.add(chunk_id)
            ids.append(chunk_id)
            documents.append(chunk.content)
            metadatas.append(_chunk_metadata(chunk))
            vectors.append(list(map(float, embedding)))

        # Chroma recommends moderate batch sizes.
        batch_size = 100
        written = 0
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            self._collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=vectors[start:end],
            )
            written += end - start
        return written

    def semantic_search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 10,
        contract_id: str | None = None,
    ) -> list[RetrievedChunk]:
        if self._collection.count() == 0:
            return []
        kwargs: dict[str, Any] = {
            "query_embeddings": [list(map(float, query_embedding))],
            "n_results": min(top_k, max(1, self._collection.count())),
            "include": ["documents", "metadatas", "distances"],
        }
        if contract_id:
            kwargs["where"] = {"contract_id": contract_id}
        result = self._collection.query(**kwargs)
        return _chroma_query_to_chunks(result, source="semantic")

    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        contract_id: str | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip() or self._collection.count() == 0:
            return []

        # Prefer Chroma document contains for exact phrases / legal tokens.
        where = {"contract_id": contract_id} if contract_id else None
        try:
            contains_result = self._collection.get(
                where=where,
                where_document={"$contains": query.strip()},
                include=["documents", "metadatas"],
                limit=max(top_k * 5, 50),
            )
            rows = _chroma_get_to_rows(contains_result)
            if rows:
                scored = _keyword_over_rows(rows, query, top_k, contract_id)
                if scored:
                    return scored
        except Exception:
            logger.debug("Chroma where_document contains failed; falling back", exc_info=True)

        # Fallback: scan a broader subset (or all if filtered by contract).
        if contract_id:
            raw = self._collection.get(
                where={"contract_id": contract_id},
                include=["documents", "metadatas"],
            )
            rows = _chroma_get_to_rows(raw)
            return _keyword_over_rows(rows, query, top_k, contract_id)

        # Global keyword: get in batches via peek/get without where.
        raw = self._collection.get(include=["documents", "metadatas"], limit=min(5000, self._collection.count()))
        rows = _chroma_get_to_rows(raw)
        return _keyword_over_rows(rows, query, top_k, None)

    def contract_exists(self, contract_id: str) -> bool:
        result = self._collection.get(where={"contract_id": contract_id}, include=[], limit=1)
        return bool(result.get("ids"))

    def list_contract_ids(self) -> list[str]:
        result = self._collection.get(include=["metadatas"], limit=min(20000, max(1, self._collection.count())))
        ids = {
            (meta or {}).get("contract_id")
            for meta in (result.get("metadatas") or [])
            if meta and meta.get("contract_id")
        }
        return sorted(ids)


def create_store(config: RAGConfig | None = None, force_memory: bool = False):
    cfg = config or RAGConfig.from_env()
    backend = (cfg.vector_backend or "chroma").lower()
    if force_memory or backend == "memory" or cfg.database_url.startswith("memory://"):
        logger.info("Using in-memory chunk store")
        return InMemoryChunkStore(embedding_dim=cfg.embedding_dim)
    logger.info("Using ChromaDB chunk store at %s", cfg.chroma_path)
    return ChromaChunkStore(cfg)


def _chunk_metadata(chunk: ChunkMetadata) -> dict[str, Any]:
    # Chroma metadata values must be str/int/float/bool.
    meta = {
        "contract_id": chunk.contract_id,
        "document_name": chunk.document_name,
        "content_hash": chunk.content_hash or "",
        "clause_number": chunk.clause_number or "",
        "clause_type": chunk.clause_type or "",
        "section": chunk.section or "",
    }
    if chunk.page is not None:
        meta["page"] = int(chunk.page)
    return meta


def _row_to_chunk(row: dict[str, Any], score: float, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=str(row["id"]),
        content=row["content"],
        contract_id=row["contract_id"],
        document_name=row["document_name"],
        clause_number=row.get("clause_number") or None,
        clause_type=row.get("clause_type") or None,
        section=row.get("section") or None,
        page=row.get("page"),
        score=float(score),
        source=source,
    )


def _chroma_query_to_chunks(result: dict[str, Any], source: str) -> list[RetrievedChunk]:
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    chunks: list[RetrievedChunk] = []
    for chunk_id, doc, meta, distance in zip(ids, docs, metas, distances):
        meta = meta or {}
        # cosine distance -> similarity
        score = 1.0 - float(distance) if distance is not None else 0.0
        chunks.append(
            RetrievedChunk(
                id=str(chunk_id),
                content=doc or "",
                contract_id=str(meta.get("contract_id") or ""),
                document_name=str(meta.get("document_name") or ""),
                clause_number=meta.get("clause_number") or None,
                clause_type=meta.get("clause_type") or None,
                section=meta.get("section") or None,
                page=meta.get("page"),
                score=score,
                source=source,
            )
        )
    return chunks


def _chroma_get_to_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk_id, doc, meta in zip(
        result.get("ids") or [],
        result.get("documents") or [],
        result.get("metadatas") or [],
    ):
        meta = meta or {}
        rows.append(
            {
                "id": chunk_id,
                "content": doc or "",
                "contract_id": meta.get("contract_id") or "",
                "document_name": meta.get("document_name") or "",
                "clause_number": meta.get("clause_number") or "",
                "clause_type": meta.get("clause_type") or "",
                "section": meta.get("section") or "",
                "page": meta.get("page"),
                "content_hash": meta.get("content_hash") or "",
            }
        )
    return rows


def _keyword_over_rows(
    rows: list[dict[str, Any]],
    query: str,
    top_k: int,
    contract_id: str | None,
) -> list[RetrievedChunk]:
    tokens = _tokenize(query)
    filtered = [r for r in rows if not contract_id or r.get("contract_id") == contract_id]
    scored: list[RetrievedChunk] = []
    q = query.strip().lower()
    for row in filtered:
        text = " ".join(
            [
                str(row.get("clause_number") or ""),
                str(row.get("clause_type") or ""),
                str(row.get("section") or ""),
                str(row.get("document_name") or ""),
                str(row.get("content") or ""),
            ]
        ).lower()
        if not text:
            continue
        hits = sum(1 for token in tokens if token in text) if tokens else 0
        phrase_boost = 2.0 if q and q in text else 0.0
        if hits == 0 and phrase_boost == 0.0:
            continue
        score = (hits / max(len(tokens), 1)) + phrase_boost
        scored.append(_row_to_chunk(row, score, "keyword"))
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:[./,][a-z0-9]+)?", text.lower()) if t]


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = na = nb = 0.0
    for i in range(n):
        av = float(a[i])
        bv = float(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
