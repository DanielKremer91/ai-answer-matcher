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

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiEmbeddings(EmbeddingProvider):
    name = "gemini"
    batch_size = 32

    def __init__(self, model: str, api_key: str, client: Optional[httpx.Client] = None):
        super().__init__(model, client)
        if not api_key:
            raise EmbeddingError("gemini: API-Schlüssel fehlt")
        self.api_key = api_key

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = f"{GEMINI_BASE}/{self.model}:batchEmbedContents"
        # SEMANTIC_SIMILARITY: both sides of every comparison are passages, not query vs. document.
        body = {"requests": [{"model": f"models/{self.model}", "taskType": "SEMANTIC_SIMILARITY",
                              "content": {"parts": [{"text": t}]}} for t in texts]}
        try:
            resp = self.client.post(url, headers={"x-goog-api-key": self.api_key}, json=body)
        except REQUEST_ERRORS as exc:
            raise EmbeddingError(f"gemini: {exc}") from exc
        raise_for_status(resp, "gemini")
        try:
            vectors = [e["values"] for e in resp.json()["embeddings"]]
        except (ValueError, KeyError, TypeError) as exc:
            raise malformed_response("gemini") from exc
        return check_vector_count(vectors, texts, "gemini")
