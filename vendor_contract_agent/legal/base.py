from __future__ import annotations

from abc import ABC, abstractmethod

from .schemas import LegalSearchResponse


class LegalSearchProvider(ABC):
    """Replaceable adapter for external legal information APIs."""

    @abstractmethod
    def search(
        self,
        query: str,
        jurisdiction: str = "DZ",
        top_k: int = 5,
    ) -> LegalSearchResponse:
        raise NotImplementedError
