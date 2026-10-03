from aam.envfile import load_env_file


def test_missing_file_is_ignored(tmp_path, monkeypatch):
    assert load_env_file(tmp_path / ".env") == []


def test_sets_unset_variables_and_keeps_existing(tmp_path, monkeypatch):
    monkeypatch.delenv("BLM_TEST_A", raising=False)
    monkeypatch.setenv("BLM_TEST_B", "aus-shell")
    env = tmp_path / ".env"
    env.write_text('# Kommentar\nBLM_TEST_A="wert a"\nexport BLM_TEST_B=aus-datei\n\nkaputte zeile\n', encoding="utf-8")
    names = load_env_file(env)
    import os
    assert os.environ["BLM_TEST_A"] == "wert a"
    assert os.environ["BLM_TEST_B"] == "aus-shell"
    assert names == ["BLM_TEST_A"]
    monkeypatch.delenv("BLM_TEST_A")


def test_single_quotes_and_empty_values(tmp_path, monkeypatch):
    monkeypatch.delenv("BLM_TEST_C", raising=False)
    monkeypatch.delenv("BLM_TEST_D", raising=False)
    env = tmp_path / ".env"
    env.write_text("BLM_TEST_C='x y'\nBLM_TEST_D=\n", encoding="utf-8")
    load_env_file(env)
    import os
    assert os.environ["BLM_TEST_C"] == "x y"
    assert "BLM_TEST_D" not in os.environ


def test_inline_comments_are_stripped(tmp_path, monkeypatch):
    import os
    for v in ("BLM_TEST_E", "BLM_TEST_F", "BLM_TEST_G"):
        monkeypatch.delenv(v, raising=False)
    env = tmp_path / ".env"
    env.write_text('BLM_TEST_E=sk-abc # privat\nBLM_TEST_F="a # b" # kommentar\nBLM_TEST_G=x#y\n', encoding="utf-8")
    load_env_file(env)
    assert os.environ["BLM_TEST_E"] == "sk-abc"
    assert os.environ["BLM_TEST_F"] == "a # b"
    assert os.environ["BLM_TEST_G"] == "x#y"


def test_empty_value_with_comment_stays_unset(tmp_path, monkeypatch):
    import os
    monkeypatch.delenv("BLM_TEST_H", raising=False)
    env = tmp_path / ".env"
    env.write_text("BLM_TEST_H=   # später eintragen\n", encoding="utf-8")
    assert load_env_file(env) == []
    assert "BLM_TEST_H" not in os.environ
