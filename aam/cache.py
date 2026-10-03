"""Embedding caches: a JSON-per-entry file cache and an in-memory cache for one session."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Optional


class JsonCache:
    def __init__(self, root: Path = Path(".cache")):
        self.root = Path(root)

    def _path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / namespace / f"{digest}.json"

    def get(self, namespace: str, key: str) -> Optional[dict]:
        path = self._path(namespace, key)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                return None
            return value
        except (ValueError, OSError):
            return None

    def set(self, namespace: str, key: str, value: dict) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value, ensure_ascii=False)
        # unique temp file per write: concurrent sessions never share one
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def clear(self) -> int:
        if not self.root.exists():
            return 0
        removed = 0
        for file in self.root.rglob("*.json"):
            file.unlink(missing_ok=True)  # another session may clear concurrently
            removed += 1
        return removed

class MemoryCache:
    """Cache for one Streamlit session with the JsonCache interface; nothing touches the disk."""

    def __init__(self):
        self._data: dict[tuple[str, str], dict] = {}

    def get(self, namespace: str, key: str) -> Optional[dict]:
        return self._data.get((namespace, key))

    def set(self, namespace: str, key: str, value: dict) -> None:
        self._data[(namespace, key)] = value

    def clear(self) -> int:
        removed = len(self._data)
        self._data.clear()
        return removed
