"""Typed application settings loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Network Reliability Platform"
    environment: str = "development"
    database_url: str = "sqlite:///./network_reliability.db"
    redis_url: str | None = None
    use_inline_jobs: bool = True
    log_level: str = "INFO"
    max_exact_edges: int = 20

    model_config = SettingsConfigDict(env_file=".env", env_prefix="NR_", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
