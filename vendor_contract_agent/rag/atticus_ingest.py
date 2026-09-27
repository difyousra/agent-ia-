from __future__ import annotations

import csv
import logging
from pathlib import Path
from uuid import uuid4

from ..dataset import download_atticus_dataset
from .clause_splitter import content_hash, detect_clause_type, split_into_clauses
from .config import RAGConfig
from .db import create_store
from .embeddings import EmbeddingService
from .ingestion import DocumentIngestionPipeline, _clean_text
from .schemas import ChunkMetadata

logger = logging.getLogger(__name__)


class AtticusDatasetIngestion:
    """Ingest the Atticus AOK dataset into the Chroma RAG store.

    Source used in production RAG:
    - full_contracts/*.pdf  (full text split into clause-level chunks)

    CSV annotations are intentionally excluded from the RAG knowledge base.
    """

    def __init__(
        self,
        store=None,
        embedder: EmbeddingService | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        self.config = config or RAGConfig.from_env()
        self.store = store or create_store(self.config)
        self.embedder = embedder or EmbeddingService(self.config)
        self.doc_pipeline = DocumentIngestionPipeline(
            store=self.store,
            embedder=self.embedder,
            config=self.config,
        )

    def ingest_all(
        self,
        *,
        dataset_root: Path | None = None,
        include_pdfs: bool = True,
        include_csvs: bool = False,
        limit: int | None = None,
        replace_existing: bool = True,
    ) -> dict:
        root = dataset_root or download_atticus_dataset()
        publication = root / "Final Publication"
        csv_dir = publication / "individual_contract_clauses"
        pdf_dir = publication / "full_contracts"

        # CSV kept only for optional/debug use; default is PDF-only RAG.
        csv_files = sorted(csv_dir.glob("*.csv")) if include_csvs and csv_dir.exists() else []
        pdf_files = []
        if include_pdfs and pdf_dir.exists():
            seen = set()
            for path in sorted(pdf_dir.iterdir()):
                if path.suffix.lower() != ".pdf":
                    continue
                key = path.name.lower()
                if key in seen:
                    continue
                seen.add(key)
                pdf_files.append(path)

        if limit is not None:
            csv_files = csv_files[:limit]
            pdf_files = pdf_files[:limit]

        summary = {
            "dataset_root": str(root),
            "csv_files": len(csv_files),
            "pdf_files": len(pdf_files),
            "contracts": [],
            "total_chunks": 0,
            "errors": [],
        }

        if include_csvs:
            for index, csv_path in enumerate(csv_files, start=1):
                contract_id = csv_path.stem
                document_name = f"{csv_path.stem}.pdf"
                try:
                    result = self.ingest_atticus_csv(
                        csv_path,
                        contract_id=contract_id,
                        document_name=document_name,
                        replace_existing=replace_existing,
                    )
                    summary["contracts"].append(result)
                    summary["total_chunks"] += result["chunks_written"]
                    logger.info(
                        "[CSV %s/%s] %s -> %s chunks",
                        index,
                        len(csv_files),
                        csv_path.name,
                        result["chunks_written"],
                    )
                except Exception as exc:
                    logger.exception("Failed CSV ingest %s", csv_path.name)
                    summary["errors"].append({"file": csv_path.name, "error": str(exc)})

        for index, pdf_path in enumerate(pdf_files, start=1):
            contract_id = pdf_path.stem
            try:
                result = self.doc_pipeline.ingest_file(
                    pdf_path,
                    contract_id=contract_id,
                    replace_existing=replace_existing,
                )
                summary["contracts"].append({"source": "pdf", **result})
                summary["total_chunks"] += result["chunks_written"]
                logger.info(
                    "[PDF %s/%s] %s -> %s chunks",
                    index,
                    len(pdf_files),
                    pdf_path.name,
                    result["chunks_written"],
                )
            except Exception as exc:
                logger.exception("Failed PDF ingest %s", pdf_path.name)
                summary["errors"].append({"file": pdf_path.name, "error": str(exc)})

        summary["store_count"] = self.store.count()
        summary["contract_ids"] = getattr(self.store, "list_contract_ids", lambda: [])()
        return summary

    def ingest_atticus_csv(
        self,
        csv_path: Path,
        *,
        contract_id: str,
        document_name: str,
        replace_existing: bool = True,
    ) -> dict:
        rows = _load_atticus_csv_rows(csv_path)
        if not rows:
            raise ValueError(f"Empty Atticus CSV: {csv_path.name}")

        if replace_existing:
            deleted = self.store.delete_contract(contract_id)
            logger.info("Removed %s chunks for contract_id=%s", deleted, contract_id)

        chunk_models: list[ChunkMetadata] = []
        for row in rows:
            clause_text = _clean_text(row["clause"])
            if len(clause_text) < 20:
                continue
            labels = row["labels"]
            primary_label = labels[0] if labels else detect_clause_type(clause_text)
            answers = "; ".join(row["answers"]) if row["answers"] else ""
            # Keep both clause text and derived answer for better retrieval.
            content = clause_text
            if answers:
                content = f"{clause_text}\n\n[ANNOTATION]\nLabels: {', '.join(labels)}\nAnswers: {answers}"

            chunk_models.append(
                ChunkMetadata(
                    id=str(uuid4()),
                    contract_id=contract_id,
                    document_name=document_name,
                    clause_number=None,
                    clause_type=primary_label,
                    section=primary_label,
                    page=None,
                    content=content,
                    content_hash=content_hash(f"{contract_id}|csv|{primary_label}|{clause_text}"),
                )
            )

        if not chunk_models:
            raise ValueError(f"No usable clauses in {csv_path.name}")

        embeddings = self.embedder.embed_documents([c.content for c in chunk_models])
        written = self.store.upsert_chunks(chunk_models, embeddings)
        return {
            "source": "csv",
            "contract_id": contract_id,
            "document_name": document_name,
            "chunks_written": written,
            "chunk_ids": [c.id for c in chunk_models],
        }


def _load_atticus_csv_rows(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            clause = (raw.get("Clause") or "").strip()
            if not clause:
                continue
            labels: list[str] = []
            answers: list[str] = []
            for index in range(1, 8):
                label = (raw.get(f"Label {index}") or "").strip()
                answer = (raw.get(f"Label {index}-Answer") or "").strip()
                if label:
                    labels.append(label)
                if answer:
                    answers.append(answer)
            rows.append({"clause": clause, "labels": labels, "answers": answers})
    return rows
