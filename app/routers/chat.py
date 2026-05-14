from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.schemas import ChatRequest, ChatResponse
from app.services.llm import LLMProvider, Message, get_llm_provider
from app.services.prompts import build_messages_with_system

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    llm: LLMProvider = Depends(get_llm_provider),
) -> ChatResponse | StreamingResponse:
    raw = [Message(role=m.role, content=m.content) for m in body.messages]
    messages = build_messages_with_system(raw)

    if body.stream:
        return StreamingResponse(
            llm.stream_generate(messages),
            media_type="text/event-stream",
        )

    content = await llm.generate(messages)
    return ChatResponse(content=content, model=type(llm).__name__)
