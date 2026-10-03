from aam.cache import JsonCache


def test_roundtrip(tmp_path):
    c = JsonCache(tmp_path / "c")
    assert c.get("wayback", "https://x.de/a") is None
    c.set("wayback", "https://x.de/a", {"text": "hallo", "n": 1})
    assert c.get("wayback", "https://x.de/a") == {"text": "hallo", "n": 1}


def test_namespaces_and_keys_are_independent(tmp_path):
    c = JsonCache(tmp_path)
    c.set("emb", "openai|text-embedding-3-small|hallo", {"v": [1.0]})
    assert c.get("emb", "openai|text-embedding-3-large|hallo") is None
    assert c.get("wayback", "openai|text-embedding-3-small|hallo") is None


def test_clear_removes_everything(tmp_path):
    c = JsonCache(tmp_path)
    c.set("a", "1", {"x": 1})
    c.set("b", "2", {"x": 2})
    c.clear()  # must not raise FileNotFoundError
    assert c.get("a", "1") is None


def test_corrupt_file_is_treated_as_miss(tmp_path):
    c = JsonCache(tmp_path)
    c.set("a", "1", {"x": 1})
    path = next((tmp_path / "a").iterdir())
    path.write_text("{not json")
    assert c.get("a", "1") is None


def test_atomic_write_no_tmp_files_remain(tmp_path):
    c = JsonCache(tmp_path)
    c.set("wayback", "https://x.de/a", {"text": "atomic"})
    # Verify no .tmp files remain
    tmp_files = list(tmp_path.glob("**/*.tmp"))
    assert len(tmp_files) == 0, f"Temporary files should be cleaned up, but found: {tmp_files}"
    # Verify value is retrievable
    assert c.get("wayback", "https://x.de/a") == {"text": "atomic"}


def test_invalid_utf8_is_treated_as_miss(tmp_path):
    c = JsonCache(tmp_path)
    c.set("a", "1", {"x": 1})
    path = next((tmp_path / "a").iterdir())
    # Write invalid UTF-8 bytes
    path.write_bytes(b"\xff\xfe\x00bad")
    assert c.get("a", "1") is None


def test_list_json_is_treated_as_miss(tmp_path):
    c = JsonCache(tmp_path)
    c.set("a", "1", {"x": 1})
    path = next((tmp_path / "a").iterdir())
    # Write a JSON array instead of object
    path.write_text("[]")
    assert c.get("a", "1") is None


def test_failed_write_leaves_no_tmp_file_and_keeps_old_value(tmp_path, monkeypatch):
    import os

    c = JsonCache(tmp_path)
    c.set("a", "1", {"x": 1})

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    try:
        c.set("a", "1", {"x": 2})
    except OSError:
        pass
    monkeypatch.undo()
    assert list(tmp_path.glob("**/*.tmp")) == []
    assert c.get("a", "1") == {"x": 1}


def test_clear_tolerates_files_vanishing_concurrently(tmp_path, monkeypatch):
    from pathlib import Path

    c = JsonCache(tmp_path)
    c.set("a", "1", {"x": 1})
    ghost = tmp_path / "a" / "gone.json"
    real_rglob = Path.rglob
    monkeypatch.setattr(Path, "rglob", lambda self, pattern: [*real_rglob(self, pattern), ghost])
    c.clear()  # must not raise FileNotFoundError
    assert c.get("a", "1") is None
