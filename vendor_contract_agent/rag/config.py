from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGConfig:
    database_url: str = "chroma://"
    vector_backend: str = "chroma"  # chroma | memory
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "contract_chunks"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_dim: int = 1024
    embedding_backend: str = "auto"  # auto | bge | hash
    reranker_backend: str = "auto"  # auto | bge | lexical
    semantic_top_k: int = 10
    keyword_top_k: int = 10
    hybrid_top_k: int = 20
    rerank_top_n: int = 5
    rrf_k: int = 60

    @classmethod
    def from_env(cls) -> "RAGConfig":
        default_chroma = str((Path.cwd() / "chroma_db").resolve())
        return cls(
            database_url=os.getenv("DATABASE_URL", "chroma://"),
            vector_backend=os.getenv("VECTOR_BACKEND", "chroma"),
            chroma_path=os.getenv("CHROMA_PATH", default_chroma),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "contract_chunks"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            embedding_backend=os.getenv("EMBEDDING_BACKEND", "auto"),
            reranker_backend=os.getenv("RERANKER_BACKEND", "auto"),
            semantic_top_k=int(os.getenv("RAG_SEMANTIC_TOP_K", "10")),
            keyword_top_k=int(os.getenv("RAG_KEYWORD_TOP_K", "10")),
            hybrid_top_k=int(os.getenv("RAG_HYBRID_TOP_K", "20")),
            rerank_top_n=int(os.getenv("RAG_RERANK_TOP_N", "5")),
            rrf_k=int(os.getenv("RAG_RRF_K", "60")),
        )
