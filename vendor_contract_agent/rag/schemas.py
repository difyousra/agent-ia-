from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    id: Optional[str] = None
    contract_id: str
    document_name: str
    clause_number: Optional[str] = None
    clause_type: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    content: str
    content_hash: Optional[str] = None


class RetrievedChunk(BaseModel):
    id: str
    content: str
    contract_id: str
    document_name: str
    clause_number: Optional[str] = None
    clause_type: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    score: float = 0.0
    source: str = "unknown"  # semantic | keyword | hybrid | rerank


class RerankedChunk(BaseModel):
    content: str
    rerank_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceRef(BaseModel):
    document_name: Optional[str] = None
    contract_id: Optional[str] = None
    clause_number: Optional[str] = None
    clause_type: Optional[str] = None
    page: Optional[int] = None
    text: Optional[str] = None


class FactWithSource(BaseModel):
    fact: str
    source: SourceRef


class ContractQueryRequest(BaseModel):
    query: str
    contract_id: Optional[str] = None
    debug: bool = False
    top_k: Optional[int] = None
    top_n: Optional[int] = None


class RetrievalDebug(BaseModel):
    semantic_results: list[RetrievedChunk] = Field(default_factory=list)
    keyword_results: list[RetrievedChunk] = Field(default_factory=list)
    rrf_results: list[RetrievedChunk] = Field(default_factory=list)
    reranked_results: list[RerankedChunk] = Field(default_factory=list)


class ContractQueryResponse(BaseModel):
    answer: Optional[str] = None
    facts: list[FactWithSource] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    confidence: Optional[float] = None
    analysis: Optional[dict[str, Any]] = None
    insufficient_information: bool = False
    debug: Optional[RetrievalDebug] = None
