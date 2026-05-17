from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):  # defines the structure of a message in the chat
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):  # defines the structure of the request body for the chat endpoint
    messages: list[Message] = Field(..., min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    conversation_id: str | None = None


class Usage(
    BaseModel
):  # defines the structure of the usage information returned in the chat response
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):  # defines the structure of the response body for the chat endpoint
    content: str
    model: str
    usage: Usage | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HealthResponse(
    BaseModel
):  # defines the structure of the response body for the health check endpoint
    status: str
    version: str
