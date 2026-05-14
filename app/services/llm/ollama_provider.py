from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.core.config import settings
from app.services.llm.base import LLMProvider, Message


class OllamaProvider(LLMProvider):
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama",  # required by the client, not validated by Ollama
        )
        self._model = settings.ollama_model

    async def generate(self, messages: list[Message]) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
        )
        return response.choices[0].message.content or ""

    async def stream_generate(self, messages: list[Message]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
