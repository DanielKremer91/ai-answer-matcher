"""Provider-agnostic embedding interface with batching, retries and split-on-error."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Callable, Optional

import httpx
import numpy as np


# Transport failures plus malformed URLs (e.g. a mistyped Ollama URL) and header
# encoding problems; all of them are fatal for the run.
REQUEST_ERRORS = (httpx.HTTPError, httpx.InvalidURL, UnicodeError)

# Rate limits and server hiccups are retried with these pauses (seconds).
RETRY_PAUSES = (2, 4, 8)
MAX_RETRY_AFTER = 30.0

# HTTP statuses that indicate a problem with the input itself (splittable).
_PER_INPUT_STATUSES = frozenset({400, 413, 422})


class EmbeddingError(Exception):
    """Embedding failure. ``fatal`` errors abort the run instead of being split."""

    def __init__(self, message: str, status: Optional[int] = None, fatal: bool = True,
                 retry_after: Optional[float] = None):
        super().__init__(message)
        self.status = status
        self.fatal = fatal
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        """429, 5xx and transport failures (timeouts, refused connections) are worth retrying."""
        if self.status is not None:
            return self.status == 429 or self.status >= 500
        return isinstance(self.__cause__, httpx.TransportError)


class EmbeddingProvider(ABC):
    name: str = "base"
    batch_size: int = 32

    def __init__(self, model: str, client: Optional[httpx.Client] = None,
                 sleeper: Optional[Callable[[float], None]] = None):
        self.model = model
        self.client = client or httpx.Client(timeout=60.0)
        self.sleeper = sleeper or time.sleep  # injectable so tests never really sleep

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed one batch. Raise EmbeddingError on any failure."""

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Call _embed_batch, retrying 429/5xx/transport errors; re-raise once retries are exhausted."""
        for attempt in range(len(RETRY_PAUSES) + 1):
            try:
                return self._embed_batch(texts)
            except EmbeddingError as exc:
                if not exc.retryable or attempt == len(RETRY_PAUSES):
                    raise
                pause = RETRY_PAUSES[attempt]
                if exc.retry_after is not None:
                    pause = min(exc.retry_after, MAX_RETRY_AFTER)
                self.sleeper(pause)
        raise AssertionError("unreachable")  # pragma: no cover

    def _embed_split(self, texts: list[str]) -> list[Optional[np.ndarray]]:
        try:
            return [np.asarray(v, dtype=np.float32) for v in self._embed_with_retry(texts)]
        except EmbeddingError as exc:
            if exc.fatal:
                raise
            if len(texts) == 1:
                return [None]
            mid = len(texts) // 2
            return self._embed_split(texts[:mid]) + self._embed_split(texts[mid:])

    def embed(self, texts: list[str]) -> list[Optional[np.ndarray]]:
        out: list[Optional[np.ndarray]] = []
        for start in range(0, len(texts), self.batch_size):
            out.extend(self._embed_split(texts[start : start + self.batch_size]))
        return out

    def probe_dimension(self) -> int:
        vectors = self._embed_with_retry(["Dimensionstest"])
        if not vectors:
            raise EmbeddingError(f"{self.name}: empty response", fatal=True)
        return len(vectors[0])


def raise_for_status(resp: httpx.Response, provider: str) -> None:
    """Raise EmbeddingError for HTTP errors; only per-input statuses are splittable."""
    status = resp.status_code
    if status < 400:
        return
    body = resp.text[:300]
    fatal = status not in _PER_INPUT_STATUSES
    if status == 400 and "api key" in resp.text.lower():
        # Gemini reports an invalid key as HTTP 400.
        fatal = True
    raise EmbeddingError(f"{provider}: HTTP {status}: {body}", status=status, fatal=fatal,
                         retry_after=_retry_after(resp))


def _retry_after(resp: httpx.Response) -> Optional[float]:
    """Numeric Retry-After header in seconds; HTTP dates and garbage are ignored."""
    try:
        value = float(resp.headers.get("Retry-After", ""))
    except ValueError:
        return None
    return value if value >= 0 else None


def malformed_response(provider: str) -> EmbeddingError:
    return EmbeddingError(f"{provider}: malformed response", fatal=True)


def check_vector_count(vectors: list, texts: list[str], provider: str) -> list:
    """Ensure the provider returned exactly one vector per input text."""
    if len(vectors) != len(texts):
        raise EmbeddingError(f"{provider}: response length mismatch", fatal=True)
    return vectors
