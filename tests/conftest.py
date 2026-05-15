from collections.abc import AsyncIterator

import pytest
import fakeredis.aioredis
from httpx import ASGITransport, AsyncClient

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
def fake_redis() -> fakeredis.aioredis.FakeRedis:
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
async def client(mock_llm: MockLLMProvider, fake_redis: fakeredis.aioredis.FakeRedis) -> AsyncIterator[AsyncClient]:
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm
    app.dependency_overrides[get_conversation_service] = lambda: ConversationService(fake_redis)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
