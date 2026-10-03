"""Embedding providers. Every text of a run goes through the same provider and model."""

from __future__ import annotations

from typing import Callable, Optional

import httpx
import numpy as np

from aam.cache import JsonCache
from aam.embeddings.base import EmbeddingError, EmbeddingProvider
from aam.embeddings.gemini import GeminiEmbeddings
from aam.embeddings.ollama import OllamaEmbeddings
from aam.embeddings.openai import OpenAIEmbeddings

PROVIDERS = ("openai", "gemini", "ollama")
DEFAULT_EMBED_MODELS = {
    "openai": "text-embedding-3-small",
    "gemini": "gemini-embedding-001",
    "ollama": "bge-m3",
}


def make_provider(
    name: str,
    model: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Optional[httpx.Client] = None,
) -> EmbeddingProvider:
    if name == "openai":
        return OpenAIEmbeddings(model, api_key or "", client)
    if name == "gemini":
        return GeminiEmbeddings(model, api_key or "", client)
    if name == "ollama":
        return OllamaEmbeddings(model, base_url, client)
    raise ValueError(f"Unbekannter Embedding-Anbieter: {name}")


def embed_cached(
    provider: EmbeddingProvider,
    texts: list[str],
    cache: Optional[JsonCache],
    *,
    progress: Optional[Callable[[int, int], None]] = None,
) -> list[Optional[np.ndarray]]:
    """Embed texts, reusing cached vectors keyed by provider, model and text.

    Each distinct text is embedded once and its vector is fanned out to every
    position. Misses are embedded batch by batch and written to the cache after
    each batch, so a fatal error late in the run keeps the earlier batches.
    ``progress(done, total)`` is called after each batch with counts of distinct
    texts that had to be embedded.
    """

    def key(text: str) -> str:
        return f"{provider.name}|{provider.model}|{text}"

    found: dict[str, Optional[np.ndarray]] = {}
    todo: list[str] = []
    for text in dict.fromkeys(texts):  # distinct, first-seen order
        hit = cache.get("embeddings", key(text)) if cache else None
        if hit is not None:
            found[text] = np.asarray(hit["v"], dtype=np.float32)
        else:
            todo.append(text)
    step = max(1, provider.batch_size)
    for start in range(0, len(todo), step):
        batch = todo[start : start + step]
        for text, vec in zip(batch, provider.embed(batch)):
            found[text] = vec
            if vec is not None and cache is not None:
                cache.set("embeddings", key(text), {"v": vec.tolist()})
        if progress:
            progress(min(start + step, len(todo)), len(todo))
    return [found[text] for text in texts]


__all__ = ["PROVIDERS", "DEFAULT_EMBED_MODELS", "EmbeddingError", "EmbeddingProvider", "embed_cached", "make_provider"]
