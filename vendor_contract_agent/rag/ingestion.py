from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from ..ingest import extract_contract_text
from .clause_splitter import content_hash, split_into_clauses
from .config import RAGConfig
from .db import create_store
from .embeddings import EmbeddingService
from .schemas import ChunkMetadata

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """Idempotent document → clause → embedding → store pipeline."""

    def __init__(
        self,
        store=None,
        embedder: EmbeddingService | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        self.config = config or RAGConfig.from_env()
        self.store = store or create_store(self.config)
        self.embedder = embedder or EmbeddingService(self.config)

    def ingest_file(
        self,
        path: str | Path,
        *,
        contract_id: str | None = None,
        replace_existing: bool = True,
    ) -> dict:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document introuvable: {file_path}")
        text = extract_contract_text(file_path)
        if not text.strip():
            raise ValueError("Empty document")
        doc_contract_id = contract_id or file_path.stem
        return self.ingest_text(
            text,
            contract_id=doc_contract_id,
            document_name=file_path.name,
            replace_existing=replace_existing,
        )

    def ingest_text(
        self,
        text: str,
        *,
        contract_id: str,
        document_name: str,
        replace_existing: bool = True,
    ) -> dict:
        cleaned = _clean_text(text)
        if not cleaned.strip():
            raise ValueError("Empty document")

        clauses = split_into_clauses(cleaned)
        if not clauses:
            raise ValueError("No clause chunks extracted")

        if replace_existing:
            deleted = self.store.delete_contract(contract_id)
            logger.info("Removed %s existing chunks for contract_id=%s", deleted, contract_id)

        chunk_models: list[ChunkMetadata] = []
        for clause in clauses:
            body = clause.content.strip()
            chunk_models.append(
                ChunkMetadata(
                    id=str(uuid4()),
                    contract_id=contract_id,
                    document_name=document_name,
                    clause_number=clause.clause_number,
                    clause_type=clause.clause_type,
                    section=clause.section,
                    page=clause.page,
                    content=body,
                    content_hash=content_hash(f"{contract_id}|{clause.clause_number}|{body}"),
                )
            )

        embeddings = self.embedder.embed_documents([c.content for c in chunk_models])
        written = self.store.upsert_chunks(chunk_models, embeddings)
        logger.info(
            "Ingested contract_id=%s document=%s chunks=%s",
            contract_id,
            document_name,
            written,
        )
        return {
            "contract_id": contract_id,
            "document_name": document_name,
            "chunks_written": written,
            "chunk_ids": [c.id for c in chunk_models],
        }


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()
