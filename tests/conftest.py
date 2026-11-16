"""A stand-in for the SocialScrape API: answers from recorded sample responses, so no test goes on the network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
TEST_KEY = "ss_test_key"


def envelope(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


class FakeAPI:
    """An httpx transport handler. Queue answers per path with `add`; every request is kept in `requests`."""

    def __init__(self) -> None:
        self.requests: list = []
        self._answers: dict[str, list[tuple[int, dict]]] = {}

    def add(self, path: str, body: dict | str, status: int = 200) -> None:
        answer = envelope(body) if isinstance(body, str) else body
        self._answers.setdefault(path, []).append((status, answer))

    def fail(self, path: str, code: str, message: str, status: int) -> None:
        self.add(path, {"ok": False, "error": {"code": code, "message": message, "retryable": False},
                        "request_id": "req_test"}, status)

    def __call__(self, request):
        import httpx

        self.requests.append(request)
        queue = self._answers.get(request.url.path)
        if not queue:
            return httpx.Response(404, json={"ok": False, "request_id": "req_test", "error": {
                "code": "not_found", "message": f"no answer queued for {request.url.path}", "retryable": False}})
        status, body = queue.pop(0) if len(queue) > 1 else queue[0]
        return httpx.Response(status, json=body)

    def params(self, index: int = -1) -> dict[str, str]:
        return dict(self.requests[index].url.params)


@pytest.fixture
def api() -> FakeAPI:
    return FakeAPI()


@pytest.fixture
def sample():
    """`sample("tiktok.profile")`: a fresh copy of a recorded answer, to alter in a test."""
    return envelope


@pytest.fixture
def client(api):
    import httpx
    from socialscrape import SocialScrape

    return SocialScrape(api_key=TEST_KEY, max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(api)))


@pytest.fixture
def async_client(api):
    import httpx
    from socialscrape import AsyncSocialScrape

    return AsyncSocialScrape(api_key=TEST_KEY, max_retries=0,
                             http_client=httpx.AsyncClient(transport=httpx.MockTransport(api)))


@pytest.fixture(autouse=True)
def _no_key_in_env(monkeypatch):
    monkeypatch.delenv("SOCIALSCRAPE_API_KEY", raising=False)
    monkeypatch.delenv("SOCIALSCRAPE_BASE_URL", raising=False)


@pytest.fixture
def session(client):
    """A Session on the fake API, to put into a component in place of the one it would make."""
    from socialscrape_haystack._client import Session

    return Session(client=client)


@pytest.fixture
def async_session(client, async_client):
    from socialscrape_haystack._client import Session

    return Session(client=client, async_client=async_client)
