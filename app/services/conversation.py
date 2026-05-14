import json
from functools import lru_cache

from redis.asyncio import Redis, from_url

from app.core.config import settings
from app.services.llm.base import Message


class ConversationService:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._ttl = settings.conversation_ttl_seconds

    async def get_history(self, conversation_id: str) -> list[Message]:
        raw = await self._redis.get(f"conv:{conversation_id}")
        if raw is None:
            return []
        return json.loads(raw)

    async def append_messages(self, conversation_id: str, messages: list[Message]) -> None:
        history = await self.get_history(conversation_id)
        history.extend(messages)
        await self._redis.set(
            f"conv:{conversation_id}",
            json.dumps(history),
            ex=self._ttl,
        )

    async def clear(self, conversation_id: str) -> None:
        await self._redis.delete(f"conv:{conversation_id}")


@lru_cache(maxsize=1)
def _build_redis() -> Redis:
    return from_url(settings.redis_url, decode_responses=True)


def get_conversation_service() -> ConversationService:
    return ConversationService(_build_redis())
