from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_log_level: str = Field(default="INFO")
    api_rate_limit_per_minute: int = Field(default=120)

    postgres_user: str = Field(default="mlops")
    postgres_password: str = Field(default="mlops")
    postgres_db: str = Field(default="mlops")
    postgres_host: str = Field(default="postgres")
    postgres_port: int = Field(default=5432)

    redis_host: str = Field(default="redis")
    redis_port: int = Field(default=6379)
    redis_cache_ttl_seconds: int = Field(default=300)

    kafka_bootstrap_servers: str = Field(default="redpanda:9092")
    kafka_prediction_topic: str = Field(default="predictions")

    triton_url: str = Field(default="triton:8001")
    triton_model_name: str = Field(default="resnet50")
    triton_model_version: str = Field(default="1")

    clickhouse_host: str = Field(default="clickhouse")
    clickhouse_port: int = Field(default=8123)
    clickhouse_user: str = Field(default="default")
    clickhouse_password: str = Field(default="")
    clickhouse_database: str = Field(default="mlops")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
