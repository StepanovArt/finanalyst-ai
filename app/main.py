from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import settings
from app.core.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError, LLMUnavailableError, PromptInjectionError
from app.core.logging import setup_logging
from app.core.middleware import RequestIDMiddleware
from app.routers import chat, health

setup_logging()

limiter = Limiter(key_func=get_remote_address, storage_uri=settings.redis_url)

app = FastAPI(title=settings.app_name, version="0.1.0")
app.state.limiter = limiter
app.add_middleware(RequestIDMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(health.router)
app.include_router(chat.router)


@app.exception_handler(LLMRateLimitError)
async def llm_rate_limit_handler(_: Request, exc: LLMRateLimitError) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": "LLM rate limit exceeded, try again later."})


@app.exception_handler(LLMTimeoutError)
async def llm_timeout_handler(_: Request, exc: LLMTimeoutError) -> JSONResponse:
    return JSONResponse(status_code=504, content={"detail": "LLM provider timed out."})


@app.exception_handler(LLMUnavailableError)
async def llm_unavailable_handler(_: Request, exc: LLMUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "LLM provider is temporarily unavailable."})


@app.exception_handler(LLMError)
async def llm_error_handler(_: Request, exc: LLMError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": "LLM provider returned an unexpected error."})


@app.exception_handler(PromptInjectionError)
async def prompt_injection_handler(_: Request, exc: PromptInjectionError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})
