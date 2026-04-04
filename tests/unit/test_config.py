"""Tests for listing_agent.config."""

import pytest
from pydantic import ValidationError

from listing_agent.config import Settings, get_config


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear get_config LRU cache before each test."""
    get_config.cache_clear()
    yield
    get_config.cache_clear()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove ANTHROPIC_API_KEY so it doesn't leak from host env."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_config_defaults(monkeypatch):
    """Set only ANTHROPIC_API_KEY, verify all defaults."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    cfg = Settings()

    assert cfg.ANTHROPIC_API_KEY == "sk-test-key"
    assert cfg.ANTHROPIC_MODEL == "claude-sonnet-4-6"
    assert cfg.ANTHROPIC_FALLBACK_MODEL == "claude-haiku-4-5-20251001"
    assert cfg.QUALITY_THRESHOLD == 0.8
    assert cfg.MAX_REFINEMENTS == 3
    assert cfg.CHROMA_DB_PATH == "./data/chroma_db"
    assert cfg.DATABASE_URL == "sqlite:///data/checkpoints.db"
    assert cfg.LANGSMITH_TRACING is False
    assert cfg.LOG_LEVEL == "INFO"
    assert cfg.SHOPIFY_SHOP_URL == ""
    assert cfg.SHOPIFY_ACCESS_TOKEN == ""
    assert cfg.SHOPIFY_API_VERSION == "2026-01"
    assert cfg.AMAZON_REFRESH_TOKEN == ""
    assert cfg.AMAZON_LWA_CLIENT_ID == ""
    assert cfg.AMAZON_LWA_CLIENT_SECRET == ""
    assert cfg.AMAZON_SELLER_ID == ""
    assert cfg.AMAZON_MARKETPLACE_ID == "ATVPDKIKX0DER"
    assert cfg.AMAZON_REGION == "NA"
    assert cfg.ETSY_API_KEY == ""
    assert cfg.ETSY_SHOP_ID == ""
    assert cfg.ETSY_ACCESS_TOKEN == ""
    assert cfg.ETSY_REFRESH_TOKEN == ""


def test_config_env_override(monkeypatch):
    """Env vars override defaults."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    monkeypatch.setenv("QUALITY_THRESHOLD", "0.95")
    monkeypatch.setenv("MAX_REFINEMENTS", "7")
    cfg = Settings()

    assert cfg.QUALITY_THRESHOLD == 0.95
    assert cfg.MAX_REFINEMENTS == 7


def test_config_requires_api_key():
    """Missing ANTHROPIC_API_KEY raises ValidationError."""
    with pytest.raises(ValidationError):
        Settings()


def test_config_quality_threshold_range(monkeypatch):
    """QUALITY_THRESHOLD outside 0-1 raises ValidationError."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    monkeypatch.setenv("QUALITY_THRESHOLD", "1.5")
    with pytest.raises(ValidationError):
        Settings()

    monkeypatch.setenv("QUALITY_THRESHOLD", "-0.1")
    with pytest.raises(ValidationError):
        Settings()


def test_config_max_refinements_range(monkeypatch):
    """MAX_REFINEMENTS outside 1-10 raises ValidationError."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    monkeypatch.setenv("MAX_REFINEMENTS", "0")
    with pytest.raises(ValidationError):
        Settings()

    monkeypatch.setenv("MAX_REFINEMENTS", "11")
    with pytest.raises(ValidationError):
        Settings()


def test_get_config_singleton(monkeypatch):
    """get_config() returns same instance on repeated calls."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    a = get_config()
    b = get_config()
    assert a is b
