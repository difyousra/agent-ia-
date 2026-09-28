from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .agent import VendorContractAgent
from .config import AgentConfig
from .rag.config import RAGConfig
from .rag.db import create_store
from .rag.embeddings import EmbeddingService
from .rag.ingestion import DocumentIngestionPipeline
from .rag.pipeline import RAGPipeline
from .rag.rerank import RerankerService
from .schemas.response import AgentQueryRequest, AgentQueryResponse

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


class IngestResponse(BaseModel):
    contract_id: str
    document_name: str
    chunks_written: int
    chunk_ids: list[str] = Field(default_factory=list)


def create_app() -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = RAGConfig.from_env()
        agent_config = AgentConfig.from_env()
        force_memory = config.database_url.startswith("memory://")
        store = create_store(config, force_memory=force_memory)
        embedder = EmbeddingService(config)
        reranker = RerankerService(config)
        rag = RAGPipeline(store=store, embedder=embedder, reranker=reranker, config=config)
        agent = VendorContractAgent(rag=rag, config=agent_config)
        ingestion = DocumentIngestionPipeline(store=store, embedder=embedder, config=config)
        state.update(
            {
                "config": config,
                "agent_config": agent_config,
                "store": store,
                "agent": agent,
                "ingestion": ingestion,
                "rag": rag,
            }
        )
        logger.info("API ready (store=%s)", type(store).__name__)
        yield
        close = getattr(store, "close", None)
        if callable(close):
            close()

    app = FastAPI(
        title="Vendor Contract Intelligence Agent API",
        version="1.2.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        store = state["store"]
        agent_config: AgentConfig = state["agent_config"]
        return {
            "status": "ok",
            "store": type(store).__name__,
            "chunks": store.count(),
            "legal_search_enabled": agent_config.legal_search_enabled,
            "default_jurisdiction": agent_config.default_jurisdiction,
        }

    @app.post("/api/contracts/query", response_model=AgentQueryResponse)
    def query_contracts(payload: AgentQueryRequest) -> AgentQueryResponse:
        query = (payload.query or "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Empty query")
        agent: VendorContractAgent = state["agent"]
        try:
            return agent.answer_query(
                query,
                contract_id=payload.contract_id,
                jurisdiction=payload.jurisdiction,
                debug=payload.debug,
                top_k=payload.top_k,
                top_n=payload.top_n,
            )
        except Exception as exc:
            logger.exception("query failed")
            return AgentQueryResponse(
                answer=None,
                insufficient_information=True,
                confidence=0.0,
                missing_information=[str(exc)],
                analysis={"error": str(exc)},
            )

    @app.post("/api/contracts/ingest", response_model=IngestResponse)
    async def ingest_contract(
        file: UploadFile = File(...),
        contract_id: str | None = Form(default=None),
    ) -> IngestResponse:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Empty document")
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".pdf", ".txt", ".md", ".docx"}:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty document")

        tmp_dir = Path(".rag_uploads")
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / file.filename
        tmp_path.write_bytes(data)
        try:
            ingestion: DocumentIngestionPipeline = state["ingestion"]
            result = ingestion.ingest_file(tmp_path, contract_id=contract_id)
            return IngestResponse(**result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("ingestion failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request, exc):  # type: ignore[no-untyped-def]
        logger.exception("Unhandled API error")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)},
        )

    return app


app = create_app()
