from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class LegalSearchResult(BaseModel):
    title: Optional[str] = None
    reference: Optional[str] = None
    document_type: Optional[str] = None
    date: Optional[str] = None
    content: Optional[str] = None
    url: Optional[str] = None


class LegalSearchResponse(BaseModel):
    source_type: str = "legal"
    jurisdiction: str = "DZ"
    results: list[LegalSearchResult] = Field(default_factory=list)
    error: Optional[str] = None
    available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
