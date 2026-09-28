from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class TraceableSource(BaseModel):
    source_type: Optional[str] = None  # contract | legal
    # Contract fields
    document: Optional[str] = None
    contract_id: Optional[str] = None
    clause: Optional[str] = None
    page: Optional[Any] = None
    # Legal fields
    title: Optional[str] = None
    reference: Optional[str] = None
    jurisdiction: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None
    text: Optional[str] = None


class RoutingDebug(BaseModel):
    route: Optional[str] = None
    tools_used: list[str] = Field(default_factory=list)
    reason: Optional[str] = None


class AgentQueryRequest(BaseModel):
    query: str
    contract_id: Optional[str] = None
    jurisdiction: Optional[str] = None
    debug: bool = False
    top_k: Optional[int] = None
    top_n: Optional[int] = None


class AgentQueryResponse(BaseModel):
    answer: Optional[str] = None
    route: Optional[str] = None
    contract_facts: list[Any] = Field(default_factory=list)
    legal_facts: list[Any] = Field(default_factory=list)
    comparison: list[Any] = Field(default_factory=list)
    obligations: list[Any] = Field(default_factory=list)
    actions: list[Any] = Field(default_factory=list)
    risks: list[Any] = Field(default_factory=list)
    missing_information: list[Any] = Field(default_factory=list)
    sources: list[TraceableSource] = Field(default_factory=list)
    confidence: Optional[float] = None
    insufficient_information: bool = False
    analysis: Optional[dict[str, Any]] = None
    debug: Optional[RoutingDebug] = None

    # Backward-compatible aliases used by older UI/API consumers
    facts: list[Any] = Field(default_factory=list)
