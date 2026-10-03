"""Shared pytest fixtures. The extraction tests use these to guarantee they can
never make a real API call: network access is blocked and no API key is visible."""
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def no_network(monkeypatch):
    """Any attempt to open a network connection fails the test immediately."""
    def blocked(*args, **kwargs):
        raise RuntimeError("Network access is blocked in extraction tests - no real API calls allowed")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


@pytest.fixture
def no_api_key(monkeypatch):
    """No OPENAI_API_KEY in the environment, and no .env file is read."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import extraction.openai_extractor as module
    monkeypatch.setattr(module, "load_dotenv", lambda *a, **k: False)
