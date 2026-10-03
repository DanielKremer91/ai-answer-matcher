import os

import pytest


@pytest.fixture(autouse=True)
def restore_environ():
    """load_env_file writes os.environ directly; undo it after every test."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)
