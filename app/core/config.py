from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "finanalyst-ai"
    debug: bool = False
    log_level: str = "INFO"

    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.2"

    redis_url: str = "redis://localhost:6379"
    conversation_ttl_seconds: int = 86400  # 24 hours

    llm_timeout_seconds: float = 30.0
    llm_max_attempts: int = 3


settings = Settings()
