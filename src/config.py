"""Project paths and settings, loaded once from .env via pydantic-settings."""

from functools import lru_cache
import os
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
INDEX_DIR = DATA_DIR / "index"
SOURCES_DIR = DATA_DIR / "sources"
URL_CATALOG_PATH = SOURCES_DIR / "urls.json"
URL_REPORT_PATH = SOURCES_DIR / "url_report.json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Provider: auto | openai | azure | local
    llm_provider: str = "auto"

    # OpenAI direct API (recommended for initial development)
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Azure OpenAI (enterprise / production)
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-02-01"
    azure_openai_chat_deployment: str = "gpt-4o-mini"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"

    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "carepolicy-rag"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "carepolicy_chunks"
    qdrant_path: str = "data/index/qdrant"
    qdrant_mode: str = "local"  # "local" or "server"

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieve: int = 20
    top_k_rerank: int = 5
    rerank_score_threshold: float = 0.25

    api_key: str = "dev-key-change-me"

    # Hugging Face Hub (cross-encoder reranker downloads)
    hf_token: str = Field(
        default="",
        validation_alias=AliasChoices("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_KEY"),
    )

    @property
    def use_azure(self) -> bool:
        return bool(self.azure_openai_api_key and self.azure_openai_endpoint)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # huggingface_hub reads os.environ directly — .env alone is not enough
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token
    return settings
