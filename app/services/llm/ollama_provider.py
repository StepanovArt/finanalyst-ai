from collections.abc import AsyncIterator
from functools import lru_cache

import openai
import stamina
from openai import AsyncOpenAI
from redis.asyncio import from_url

from app.core.circuit_breaker import CircuitBreaker
from app.core.config import settings
from app.core.exceptions import LLMRateLimitError, LLMTimeoutError, LLMUnavailableError
from app.services.llm.base import LLMProvider, Message


def _map_error(exc: openai.APIError) -> LLMRateLimitError | LLMTimeoutError | LLMUnavailableError:
    if isinstance(exc, openai.RateLimitError):
        return LLMRateLimitError(str(exc))
    if isinstance(exc, openai.APITimeoutError):
        return LLMTimeoutError(str(exc))
    return LLMUnavailableError(str(exc))


@lru_cache(maxsize=1)
def _ollama_circuit_breaker() -> CircuitBreaker:
    return CircuitBreaker(from_url(settings.redis_url, decode_responses=True), name="ollama")


class OllamaProvider(LLMProvider):
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama",  # required by the client, not validated by Ollama
            timeout=settings.llm_timeout_seconds,
        )
        self._model = settings.ollama_model

    async def generate(self, messages: list[Message]) -> str:
        @stamina.retry(
            on=(LLMRateLimitError, LLMUnavailableError),
            attempts=settings.llm_max_attempts,
            wait_initial=1.0,
            wait_max=30.0,
        )
        async def _call() -> str:
            async with _ollama_circuit_breaker():
                try:
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,  # type: ignore[arg-type]
                    )
                    return response.choices[0].message.content or ""
                except openai.APIError as exc:
                    raise _map_error(exc) from exc

        return await _call()

    async def stream_generate(self, messages: list[Message]) -> AsyncIterator[str]:
        async with _ollama_circuit_breaker():
            try:
                stream = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore[arg-type]
                    stream=True,
                )
            except openai.APIError as exc:
                raise _map_error(exc) from exc

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
