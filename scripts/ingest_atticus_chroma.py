from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from vendor_contract_agent.rag.atticus_ingest import AtticusDatasetIngestion
from vendor_contract_agent.rag.config import RAGConfig
from vendor_contract_agent.rag.db import create_store
from vendor_contract_agent.rag.embeddings import EmbeddingService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(
        description="Ingest Atticus PDFs into ChromaDB RAG store (CSV excluded by default)"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of PDF/CSV files (debug)")
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Ingest only annotated CSV clauses (debug; not used in production RAG)",
    )
    parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="Ingest only full-contract PDFs (default behavior)",
    )
    parser.add_argument("--memory", action="store_true", help="Use in-memory store instead of Chroma")
    parser.add_argument(
        "--chroma-path",
        default=None,
        help="Override Chroma persistence directory",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing Chroma collection/path before ingest",
    )
    args = parser.parse_args(argv)

    config = RAGConfig.from_env()
    if args.chroma_path:
        config = RAGConfig(
            database_url=config.database_url,
            vector_backend="memory" if args.memory else "chroma",
            chroma_path=args.chroma_path,
            chroma_collection=config.chroma_collection,
            embedding_model=config.embedding_model,
            reranker_model=config.reranker_model,
            embedding_dim=config.embedding_dim,
            embedding_backend=config.embedding_backend,
            reranker_backend=config.reranker_backend,
            semantic_top_k=config.semantic_top_k,
            keyword_top_k=config.keyword_top_k,
            hybrid_top_k=config.hybrid_top_k,
            rerank_top_n=config.rerank_top_n,
            rrf_k=config.rrf_k,
        )
    elif args.memory:
        config = RAGConfig(
            database_url="memory://",
            vector_backend="memory",
            chroma_path=config.chroma_path,
            chroma_collection=config.chroma_collection,
            embedding_backend=config.embedding_backend,
            reranker_backend=config.reranker_backend,
        )

    if args.reset and not args.memory:
        chroma_dir = Path(config.chroma_path)
        if chroma_dir.exists():
            import shutil

            shutil.rmtree(chroma_dir)
            print(f"Reset Chroma path: {chroma_dir}")

    store = create_store(config, force_memory=args.memory)
    embedder = EmbeddingService(config)
    pipeline = AtticusDatasetIngestion(store=store, embedder=embedder, config=config)

    # Production RAG is PDF-only. CSV is opt-in via --csv-only.
    include_csvs = bool(args.csv_only)
    include_pdfs = not args.csv_only
    summary = pipeline.ingest_all(
        include_csvs=include_csvs,
        include_pdfs=include_pdfs,
        limit=args.limit,
        replace_existing=True,
    )

    out_dir = Path("eval_atticus")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "atticus_chroma_ingest.json"
    # Keep report compact: drop long chunk_ids lists.
    compact = {
        "dataset_root": summary["dataset_root"],
        "csv_files": summary["csv_files"],
        "pdf_files": summary["pdf_files"],
        "total_chunks": summary["total_chunks"],
        "store_count": summary["store_count"],
        "contract_count": len(summary.get("contract_ids") or []),
        "errors": summary["errors"],
        "contracts_sample": summary["contracts"][:5],
    }
    out.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"Wrote {out}")
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
