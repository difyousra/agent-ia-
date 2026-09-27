from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from vendor_contract_agent.rag.config import RAGConfig
from vendor_contract_agent.rag.db import create_store
from vendor_contract_agent.rag.embeddings import EmbeddingService
from vendor_contract_agent.rag.ingestion import DocumentIngestionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="Ingest contract documents into the RAG knowledge base")
    parser.add_argument("paths", nargs="+", help="PDF/TXT/MD/DOCX files or directories")
    parser.add_argument("--contract-id", default=None, help="Force a contract_id for a single file")
    parser.add_argument("--memory", action="store_true", help="Force in-memory store")
    parser.add_argument("--chroma-path", default=None, help="Chroma persistence directory")
    args = parser.parse_args(argv)

    config = RAGConfig.from_env()
    if args.chroma_path or args.memory:
        config = RAGConfig(
            database_url="memory://" if args.memory else "chroma://",
            vector_backend="memory" if args.memory else "chroma",
            chroma_path=args.chroma_path or config.chroma_path,
            chroma_collection=config.chroma_collection,
            embedding_backend=config.embedding_backend,
            reranker_backend=config.reranker_backend,
        )
    store = create_store(config, force_memory=args.memory)
    embedder = EmbeddingService(config)
    pipeline = DocumentIngestionPipeline(store=store, embedder=embedder, config=config)

    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("*")))
        else:
            files.append(path)
    files = [f for f in files if f.suffix.lower() in {".pdf", ".txt", ".md", ".docx"}]
    if not files:
        print("No supported documents found")
        return 1

    results = []
    for index, file_path in enumerate(files, start=1):
        contract_id = args.contract_id if len(files) == 1 else None
        print(f"[{index}/{len(files)}] Ingest {file_path}")
        result = pipeline.ingest_file(file_path, contract_id=contract_id)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    out = Path("eval_atticus") / "ingestion_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
