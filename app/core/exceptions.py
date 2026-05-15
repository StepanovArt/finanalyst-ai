class LLMError(Exception):
    """Base exception for all LLM provider errors."""


class LLMRateLimitError(LLMError):
    """Provider returned 429 — too many requests."""


class LLMTimeoutError(LLMError):
    """Provider did not respond within the allowed time."""


class LLMUnavailableError(LLMError):
    """Provider is temporarily unavailable (circuit open or 5xx)."""


class ConversationNotFoundError(Exception):
    """Requested conversation_id does not exist in storage."""


class PromptInjectionError(Exception):
    """User message contains suspected prompt injection."""
