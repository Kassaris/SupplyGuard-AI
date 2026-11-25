"""
Language-model access layer.

ASSIGNMENT REQUIREMENT (§05)
    "A hosted or locally served LLM, accessed only through LangChain model
     interfaces. The model is configurable, and every call is traced."

DESIGN DECISIONS
----------------
1. ONE place creates chat models.  Agents, graph nodes and chains call
   `get_chat_model()` and receive a `BaseChatModel`.  Nobody imports
   `AzureChatOpenAI` directly.  -> Swapping Azure for OpenAI/Ollama is a
   one-line `.env` change (MODEL_PROVIDER=OLLAMA).

2. We return the LangChain *interface* (`BaseChatModel`), so every caller can
   use the standard LangChain Core API: `.invoke`, `.stream`, `.bind_tools`,
   `.with_structured_output` (Week 1, days 3-4).

3. Why `AzureChatOpenAI` explicitly instead of `init_chat_model("azure_openai:...")`?
   Both work.  Building the Azure class explicitly lets us pass the exact
   deployment / api_version / key from our typed `Settings`, which is easier
   to read for reviewers and fails with a clear error if a value is missing.
   For OpenAI and Ollama we use `init_chat_model`, LangChain's generic
   factory, because they need only a model name.

4. Tracing: LangSmith tracing is ZERO-CODE (Week 2, day 8).  If the env vars
   LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set, every LangChain /
   LangGraph call is traced automatically.  `ensure_tracing_env()` just
   copies them from `.env` into `os.environ` (pydantic-settings reads the
   file but does not export it), so the LangSmith client can see them.

5. `max_retries` + `timeout`: hosted LLMs fail transiently (429, 5xx).
   LangChain retries with exponential back-off for us.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from supplyguard_ai.config import PROJECT_ROOT, Settings, get_settings


def ensure_tracing_env() -> None:
    """Export `.env` values (incl. LANGSMITH_*) into the process environment.

    `override=False` -> a real environment variable (e.g. set in CI) always
    wins over the file.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _build_model(settings: Settings) -> BaseChatModel:
    """Create the provider-specific chat model behind the common interface."""
    common = {
        "temperature": settings.llm_temperature,
        "max_retries": 3,
        "timeout": 60,
    }

    if settings.model_provider == "AZURE":
        # Imported lazily so that a user running Ollama doesn't need Azure configured.
        from langchain_openai import AzureChatOpenAI

        if not settings.azure_openai_endpoint or not settings.azure_openai_api_key:
            raise RuntimeError(
                "MODEL_PROVIDER=AZURE but AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY are missing in .env"
            )
        return AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            azure_deployment=settings.azure_openai_deployment_name,
            api_version=settings.openai_api_version,
            api_key=settings.azure_openai_api_key,  # SecretStr is accepted directly
            **common,
        )

    if settings.model_provider == "OPENAI":
        return init_chat_model(
            settings.openai_model,
            model_provider="openai",
            api_key=settings.openai_api_key.get_secret_value() if settings.openai_api_key else None,
            **common,
        )

    # OLLAMA: locally served model (requires `uv add langchain-ollama`).
    return init_chat_model(settings.ollama_model, model_provider="ollama", temperature=settings.llm_temperature)


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """The single entry point every agent / node uses to get an LLM.

    Cached: model clients hold HTTP connection pools, so we reuse one.
    """
    ensure_tracing_env()
    return _build_model(get_settings())


# --------------------------------------------------------------------------- embeddings
EMBEDDING_DIM = 1536  # text-embedding-3-small; the vector columns are created with this size


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """The single entry point for embedding models (document + memory indexing, search).

    Same rule as chat models: callers get the LangChain `Embeddings`
    interface (`embed_documents`, `embed_query`), never a provider class,
    so the vector stores work with any provider - and tests pass
    `DeterministicFakeEmbedding(size=EMBEDDING_DIM)` instead.

    `chunk_size=64`: texts are sent in batches of 64 per request (fewer
    round-trips during ingestion than one call per chunk).
    """
    ensure_tracing_env()
    s = get_settings()
    if s.model_provider == "AZURE":
        from langchain_openai import AzureOpenAIEmbeddings

        if not s.azure_embedding_endpoint or not s.azure_embedding_api_key:
            raise RuntimeError("AZURE_EMBEDDING_ENDPOINT / AZURE_EMBEDDING_API_KEY missing in .env")
        return AzureOpenAIEmbeddings(
            azure_endpoint=s.azure_embedding_endpoint,
            azure_deployment=s.azure_openai_embedding_deployment,
            api_key=s.azure_embedding_api_key,
            api_version=s.openai_api_version,
            chunk_size=64,
            max_retries=3,
        )
    if s.model_provider == "OPENAI":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=s.openai_api_key, chunk_size=64)
    from langchain_ollama import OllamaEmbeddings  # requires `uv add langchain-ollama`

    return OllamaEmbeddings(model="nomic-embed-text")
