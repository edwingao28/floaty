"""Type-safe configuration loading from environment."""

from __future__ import annotations

import functools

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # --- Anthropic ---
    ANTHROPIC_API_KEY: str = Field(min_length=1)
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"
    ANTHROPIC_FALLBACK_MODEL: str = "claude-haiku-4-5-20251001"

    # --- Agent ---
    QUALITY_THRESHOLD: float = Field(default=0.8, ge=0.0, le=1.0)
    MAX_REFINEMENTS: int = Field(default=3, ge=1, le=10)

    # --- Scoring ---
    RULES_WEIGHT: float = Field(default=0.6, ge=0.0, le=1.0)
    LLM_WEIGHT: float = Field(default=0.4, ge=0.0, le=1.0)
    CONVERGENCE_DELTA: float = Field(default=0.03, ge=0.0)
    OSCILLATION_WINDOW: int = Field(default=2, ge=1)
    LLM_JUDGE_SAMPLE_RATE: float = Field(default=1.0, ge=0.0, le=1.0)

    # --- Storage ---
    CHROMA_DB_PATH: str = "./data/chroma_db"
    DATABASE_URL: str = "sqlite:///data/checkpoints.db"

    # --- Observability ---
    LANGSMITH_TRACING: bool = False
    LOG_LEVEL: str = "INFO"

    # --- Shopify ---
    SHOPIFY_SHOP_URL: str = ""
    SHOPIFY_ACCESS_TOKEN: str = ""
    SHOPIFY_API_VERSION: str = "2026-01"

    # --- Amazon SP-API ---
    AMAZON_REFRESH_TOKEN: str = ""
    AMAZON_LWA_CLIENT_ID: str = ""
    AMAZON_LWA_CLIENT_SECRET: str = ""
    AMAZON_SELLER_ID: str = ""
    AMAZON_MARKETPLACE_ID: str = "ATVPDKIKX0DER"
    AMAZON_REGION: str = "NA"

    # --- Etsy ---
    ETSY_API_KEY: str = ""
    ETSY_SHOP_ID: str = ""
    ETSY_ACCESS_TOKEN: str = ""
    ETSY_REFRESH_TOKEN: str = ""


@functools.lru_cache(maxsize=1)
def get_config() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
