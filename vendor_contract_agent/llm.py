from __future__ import annotations

import os

from openai import OpenAI, BadRequestError


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "").strip() or None
        self.model = model or os.getenv("OPENAI_MODEL") or "deepseek-chat"
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY est manquant. Définissez-le dans l'environnement ou un fichier .env."
            )
        kwargs: dict = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
            )
        except BadRequestError:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=messages,
            )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Le modèle a renvoyé une réponse vide.")
        return content
