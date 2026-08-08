"""Pick which AI provider to use, based on the keys present in .env.

In "auto" mode: an OpenAI key wins, then Azure credentials, otherwise "local"
(no chat model; embeddings fall back to a small on-disk model).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings

from src.config import get_settings

Provider = Literal["openai", "azure", "local"]


def active_provider() -> Provider:
    settings = get_settings()
    if settings.llm_provider != "auto":
        return settings.llm_provider  # type: ignore[return-value]

    if settings.openai_api_key:
        return "openai"
    if settings.use_azure:
        return "azure"
    return "local"


@lru_cache
def get_chat_model():
    settings = get_settings()
    provider = active_provider()

    if provider == "openai":
        return ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
        )
    if provider == "azure":
        return AzureChatOpenAI(
            azure_deployment=settings.azure_openai_chat_deployment,
            openai_api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            temperature=0.0,
        )
    return None


@lru_cache
def get_embedding_model():
    settings = get_settings()
    provider = active_provider()

    if provider == "openai":
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
    if provider == "azure":
        return AzureOpenAIEmbeddings(
            azure_deployment=settings.azure_openai_embedding_deployment,
            openai_api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
        )
    return None
