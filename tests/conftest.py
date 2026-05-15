from collections.abc import AsyncIterator

import fakeredis
import pytest
from httpx import ASGITransport, AsyncClient

import app.core.limiter as limiter_module
from app.core.circuit_breaker import NullCircuitBreaker
from app.main import app
from app.services.conversation import ConversationService, get_conversation_service
from app.services.llm.base import LLMProvider, Message
from app.services.llm.dependencies import get_llm_provider


class MockLLMProvider(LLMProvider):
    def __init__(self, response: str = "Mock response") -> None:
        self._response = response

    async def generate(self, messages: list[Message]) -> str:
        return self._response

    async def stream_generate(self, messages: list[Message]) -> AsyncIterator[str]:
        for word in self._response.split():
            yield word + " "


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def fake_redis() -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis(decode_responses=True)


@pytest.fixture(autouse=True)
def patch_infrastructure(monkeypatch: pytest.MonkeyPatch) -> None:
    # disable Redis-backed rate limiter
    monkeypatch.setattr(limiter_module.limiter, "enabled", False, raising=False)

    # replace Redis-backed circuit breakers with no-ops
    null_cb = NullCircuitBreaker()
    monkeypatch.setattr("app.services.llm.openai_provider._openai_circuit_breaker", lambda: null_cb)
    monkeypatch.setattr("app.services.llm.ollama_provider._ollama_circuit_breaker", lambda: null_cb)


@pytest.fixture
async def client(mock_llm: MockLLMProvider, fake_redis: fakeredis.FakeAsyncRedis) -> AsyncIterator[AsyncClient]:
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm
    app.dependency_overrides[get_conversation_service] = lambda: ConversationService(fake_redis)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
