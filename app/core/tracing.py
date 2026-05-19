"""
Langfuse tracing integration.

Returns a LangChain-compatible CallbackHandler that forwards LangGraph
traces to the self-hosted Langfuse instance. When Langfuse keys are not
configured the helper returns None and tracing is silently skipped.

Why optional tracing?
Tests and local runs without Langfuse should work without any extra config.
The caller guards on `handler is not None` before passing to ainvoke.
"""

from __future__ import annotations

from loguru import logger

from app.core.config import settings


def get_langfuse_handler(
    session_id: str | None = None,
    user_id: str | None = None,
) -> object | None:
    """Build a Langfuse CallbackHandler if credentials are present.

    Args:
        session_id: optional session identifier forwarded to Langfuse
        user_id: optional user identifier forwarded to Langfuse

    Returns:
        CallbackHandler instance, or None when Langfuse is not configured
    """
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None

    try:
        from langfuse.callback import CallbackHandler  # type: ignore[import-untyped]

        handler = CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            session_id=session_id,
            user_id=user_id,
        )
        return handler
    except Exception as exc:
        logger.warning(f"Langfuse handler initialisation failed: {exc}")
        return None
