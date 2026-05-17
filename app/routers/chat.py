import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.limiter import limiter
from app.schemas import ChatRequest, ChatResponse
from app.services.conversation import ConversationService, get_conversation_service
from app.services.guard import check_messages
from app.services.llm import LLMProvider, Message, get_llm_provider
from app.services.prompts import build_messages_with_system

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
@limiter.limit(settings.rate_limit_chat)
async def chat(
    request: Request,
    body: ChatRequest,
    llm: LLMProvider = Depends(get_llm_provider),
    conv: ConversationService = Depends(get_conversation_service),
) -> ChatResponse:
    check_messages(body.messages)

    user_messages = [Message(role=m.role, content=m.content) for m in body.messages]

    history = await conv.get_history(body.conversation_id) if body.conversation_id else []
    messages = build_messages_with_system([*history, *user_messages])

    content = await llm.generate(messages)

    if body.conversation_id:
        await conv.append_messages(
            body.conversation_id,
            [*user_messages, Message(role="assistant", content=content)],
        )

    return ChatResponse(content=content, model=type(llm).__name__)


async def _sse_generator(llm: LLMProvider, messages: list[Message]) -> AsyncIterator[str]:
    async for token in llm.stream_generate(messages):
        yield f"data: {json.dumps({'content': token})}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/stream")
@limiter.limit(settings.rate_limit_stream)
async def chat_stream(
    request: Request,
    body: ChatRequest,
    llm: LLMProvider = Depends(get_llm_provider),
) -> StreamingResponse:
    check_messages(body.messages)
    raw = [Message(role=m.role, content=m.content) for m in body.messages]
    messages = build_messages_with_system(raw)
    return StreamingResponse(_sse_generator(llm, messages), media_type="text/event-stream")
