from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="NIYAM_", extra="ignore")

    database_url: str = "postgresql+psycopg://niyam:niyam@localhost:5432/niyam"
    embedding_dim: int = 384
    data_dir: str = "data"
    log_level: str = "INFO"

    # Scraping: identify ourselves and stay slow.
    user_agent: str = "NiyamBot/0.1 (+https://github.com/Gulshan-heap/niyam)"
    request_interval_seconds: float = 2.0
    http_timeout_seconds: float = 60.0

    # Answer generation through LiteLLM; the provider key (GEMINI_API_KEY, GROQ_API_KEY, ...)
    # is read from the environment by LiteLLM itself.
    llm_model: str = "groq/openai/gpt-oss-120b"
    llm_fallbacks: list[str] = ["groq/qwen/qwen3.8-27b", "gemini/gemini-2.5-flash"]
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 60.0

    # Daily ingestion worker (India time). RBI posts during the working day.
    ingest_hour: int = 20
    ingest_minute: int = 0
    ingest_on_start: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
