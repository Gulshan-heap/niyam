from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="NIYAM_", extra="ignore")

    database_url: str = "postgresql+psycopg://niyam:niyam@localhost:5432/niyam"
    embedding_dim: int = 384
    data_dir: str = "data"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
