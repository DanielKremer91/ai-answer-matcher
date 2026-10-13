"""Deterministic stand-ins for tests: a bag-of-words embedding provider and sample data."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from aam.embeddings.base import EmbeddingProvider
from aam.models import OWN, SOURCE, Document


def bow_vector(text: str, dim: int = 64) -> np.ndarray:
    """Hash every word into one of `dim` buckets; shared words give a high cosine."""
    vector = np.zeros(dim, dtype=np.float32)
    for word in re.findall(r"\w+", text.lower()):
        vector[int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % dim] += 1.0
    return vector


class FakeProvider(EmbeddingProvider):
    name = "fake"
    batch_size = 64

    def __init__(self, dim: int = 64):
        super().__init__("fake-model")
        self.dim = dim
        self.calls = 0  # number of texts embedded
        self.seen_texts: list[str] = []

    def _embed_batch(self, texts):
        self.calls += len(texts)
        self.seen_texts.extend(texts)
        return [bow_vector(t, self.dim).tolist() for t in texts]


ANSWER = """Eine Dachsanierung kostet je nach Umfang zwischen 100 und 300 Euro pro Quadratmeter.

Kosten im Überblick
Eine Neueindeckung ohne Dämmung liegt bei 100 bis 150 Euro pro Quadratmeter.
Mit Dämmung steigen die Kosten auf 200 bis 300 Euro pro Quadratmeter.

Förderung
Die KfW fördert die Dachdämmung mit zinsgünstigen Krediten.
Das BAFA zahlt Zuschüsse von bis zu 20 Prozent.
"""

OWN_DOCS = [
    Document(
        "https://me.de/kosten",
        "Kosten der Dachsanierung\n"
        "Eine Neueindeckung ohne Dämmung liegt bei 100 bis 150 Euro pro Quadratmeter.\n"
        "Mit Dämmung steigen die Kosten auf 200 bis 300 Euro pro Quadratmeter.",
        OWN,
    ),
    Document(
        "https://me.de/blog",
        "Unser Team stellt sich vor.\n"
        "Wir sind seit 20 Jahren als Dachdecker in Köln tätig und bilden regelmäßig aus.",
        OWN,
    ),
]

SOURCE_DOCS = [
    Document(
        "https://quelle.de/foerderung",
        "Die KfW fördert die Dachdämmung mit zinsgünstigen Krediten.\n"
        "Das BAFA zahlt Zuschüsse von bis zu 20 Prozent.",
        SOURCE,
    ),
]


def sample_result(with_sources: bool = True):
    """A complete run over the sample data with the fake provider."""
    from datetime import date

    from aam.pipeline import analyze, prepare

    prepared = prepare("dachsanierung kosten", ANSWER, own_docs=OWN_DOCS,
                       source_docs=SOURCE_DOCS if with_sources else [])
    return analyze(prepared, FakeProvider(), None, today=date(2026, 10, 3))
