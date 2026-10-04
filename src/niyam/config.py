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


@lru_cache
def get_settings() -> Settings:
    return Settings()
