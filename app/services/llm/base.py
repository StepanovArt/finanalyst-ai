from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TypedDict


class Message(TypedDict):
    role: str
    content: str


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, messages: list[Message]) -> str: ...

    @abstractmethod
    def stream_generate(self, messages: list[Message]) -> AsyncIterator[str]: ...
