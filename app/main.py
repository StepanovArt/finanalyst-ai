from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core._exceptions import LLMError, LLMRateLimitError, LLMTimeoutError, LLMUnavailableError
from app.routers import chat, health

app = FastAPI(title=settings.app_name, version="0.1.0")

app.include_router(health.router)
app.include_router(chat.router)


@app._exception_handler(LLMRateLimitError)
async def llm_rate_limit_handler(_: Request, _exc: LLMRateLimitError) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": "LLM rate limit _exceeded, try again later."})


@app._exception_handler(LLMTimeoutError)
async def llm_timeout_handler(_: Request, _exc: LLMTimeoutError) -> JSONResponse:
    return JSONResponse(status_code=504, content={"detail": "LLM provider timed out."})


@app._exception_handler(LLMUnavailableError)
async def llm_unavailable_handler(_: Request, _exc: LLMUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "LLM provider is temporarily unavailable."})


@app._exception_handler(LLMError)
async def llm_error_handler(_: Request, _exc: LLMError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": "LLM provider returned an unexpected error."})
