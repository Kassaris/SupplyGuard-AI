"""
Central configuration for SupplyGuard AI.

WHY THIS FILE EXISTS
--------------------
Every other module needs *some* configuration: which LLM to call, where the
data lives, the Postgres connection string, the spending cap, ...  If each
module read `os.environ` on its own we would end up with:
  * typos in env-var names that fail silently (returns None),
  * no type conversion ("75000" is a string, not an int),
  * secrets accidentally printed in logs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the folder that contains pyproject.toml.
# __file__ = .../SupplyGuard-AI/src/supplyguard_ai/config.py  -> parents[2] = SupplyGuard-AI
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All runtime configuration, read from environment variables / `.env`.

    Field names are matched to env vars case-insensitively, so the field
    `azure_openai_endpoint` is filled from `AZURE_OPENAI_ENDPOINT`.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",  # absolute path -> works from any cwd
        env_file_encoding="utf-8",
        extra="ignore",  # .env contains keys (LANGSMITH_*, NEO4J_*) used by other libs; don't crash on them
    )

    # ------------------------------------------------------------------ LLM
    # "Language model access ... The model is configurable" (assignment §05).
    # Switching provider is a config change, never a code change.
    model_provider: Literal["AZURE", "OPENAI", "OLLAMA"] = "AZURE"

    azure_openai_api_key: SecretStr | None = None
    azure_openai_endpoint: str | None = None
    openai_api_version: str = "2024-10-21"
    azure_openai_deployment_name: str = "gpt-4.1-mini"

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4.1-mini"

    ollama_model: str = "llama3.1"

    # temperature=0: we want reproducible, auditable answers, not creativity.
    llm_temperature: float = 0.0

    # ------------------------------------------------------------- Embeddings
    azure_embedding_endpoint: str | None = None
    azure_embedding_api_key: SecretStr | None = None
    azure_openai_embedding_deployment: str = "text-embedding-3-small"

    # ------------------------------------------------------------- Postgres
    postgres_user: str = "postgres"
    postgres_password: SecretStr = SecretStr("postgres")
    postgres_db: str = "supplyguard"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # ------------------------------------------------------------- Storage
    # "postgres" = production / docker compose (durable cases, approvals, audit).
    # "memory"   = unit tests: no network, no Docker (CONTRIBUTING.md §7).
    #              tests/conftest.py sets STORAGE_BACKEND=memory.
    storage_backend: Literal["postgres", "memory"] = "postgres"

    # ----------------------------------------------- Other services (compose)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("neo4j")
    chroma_host: str = "localhost"
    chroma_port: int = 8001  # host port in docker-compose.yml (container port 8000)
    # MCP endpoints. None (default, local dev / unit tests) = call the
    # procurement system in-process; docker-compose sets both URLs, and then
    # the app reaches procurement ONLY through the MCP server.
    procurement_mcp_url: str | None = None
    external_mcp_url: str | None = None
    external_info_url: str = "http://localhost:8099"  # the provided external_info_service.py
    mcp_host: str | None = None  # bind address of our MCP servers; None = auto (0.0.0.0 in Docker, else 127.0.0.1)
    sandbox_url: str | None = None  # e.g. http://sandbox:8200 inside compose

    # ---------------------------------------------------------------- Data
    # Where the provided (read-only) support data lives.
    # Env var SUPPORT_DATA_DIR (set to /code/support-data in docker-compose.yml).
    support_data_dir: Path = Field(default=PROJECT_ROOT / "support-data")

    @property
    def postgres_dsn(self) -> str:
        """Connection string for psycopg / LangGraph's PostgresSaver (used later)."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide singleton.

    `lru_cache` means the .env file is parsed once.  In tests you can call
    `get_settings.cache_clear()` and set env vars to get a fresh instance.
    """
    return Settings()


def postgres_dsn() -> str:
    """Module-level shortcut for the Postgres connection string.

    auth/users.py and ingestion/seed_users.py import this function
    (`from supplyguard_ai.config import postgres_dsn`) and call `postgres_dsn()`.
    It returns the same value as `get_settings().postgres_dsn`.
    """
    return get_settings().postgres_dsn
