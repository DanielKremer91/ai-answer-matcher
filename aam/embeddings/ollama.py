from __future__ import annotations

from typing import Optional

import httpx

from aam.embeddings.base import (
    REQUEST_ERRORS,
    EmbeddingError,
    EmbeddingProvider,
    check_vector_count,
    malformed_response,
    raise_for_status,
)

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaEmbeddings(EmbeddingProvider):
    name = "ollama"
    batch_size = 16

    def __init__(self, model: str, base_url: Optional[str] = None, client: Optional[httpx.Client] = None):
        super().__init__(model, client)
        self.base_url = (base_url or DEFAULT_OLLAMA_URL).rstrip("/")

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = self.client.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": texts})
        except REQUEST_ERRORS as exc:
            raise EmbeddingError(f"ollama: {exc}") from exc
        raise_for_status(resp, "ollama")
        try:
            vectors = resp.json()["embeddings"]
        except (ValueError, KeyError, TypeError) as exc:
            raise malformed_response("ollama") from exc
        return check_vector_count(vectors, texts, "ollama")
