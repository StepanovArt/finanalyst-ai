from functools import lru_cache

from app.core.config import settings
from app.services.llm.base import LLMProvider
from app.services.llm.ollama_provider import OllamaProvider
from app.services.llm.openai_provider import OpenAIProvider


@lru_cache(maxsize=1)
def _build_provider() -> LLMProvider:
    match settings.llm_provider.lower():
        case "openai":
            return OpenAIProvider()
        case "ollama":
            return OllamaProvider()
        case other:
            raise ValueError(f"Unknown LLM_PROVIDER: {other!r}. Use 'openai' or 'ollama'.")


def get_llm_provider() -> LLMProvider:
    return _build_provider()
