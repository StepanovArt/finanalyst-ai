import pytest
import fakeredis.aioredis

from app.services.conversation import ConversationService
from app.services.llm.base import Message


@pytest.fixture
def svc(fake_redis: fakeredis.aioredis.FakeRedis) -> ConversationService:
    return ConversationService(fake_redis)


async def test_empty_history(svc: ConversationService) -> None:
    history = await svc.get_history("nonexistent")
    assert history == []


async def test_append_and_retrieve(svc: ConversationService) -> None:
    messages: list[Message] = [
        {"role": "user", "content": "What is FCF?"},
        {"role": "assistant", "content": "Free Cash Flow is..."},
    ]
    await svc.append_messages("conv-1", messages)
    history = await svc.get_history("conv-1")
    assert len(history) == 2
    assert history[0]["content"] == "What is FCF?"


async def test_append_accumulates(svc: ConversationService) -> None:
    msg1: list[Message] = [{"role": "user", "content": "First"}]
    msg2: list[Message] = [{"role": "user", "content": "Second"}]
    await svc.append_messages("conv-2", msg1)
    await svc.append_messages("conv-2", msg2)
    history = await svc.get_history("conv-2")
    assert len(history) == 2


async def test_clear_removes_history(svc: ConversationService) -> None:
    messages: list[Message] = [{"role": "user", "content": "Delete me"}]
    await svc.append_messages("conv-3", messages)
    await svc.clear("conv-3")
    assert await svc.get_history("conv-3") == []
