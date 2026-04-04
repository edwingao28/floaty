from pathlib import Path

import pytest
from listing_agent.rag.retriever import PlatformRetriever


@pytest.fixture
def retriever(tmp_path):
    knowledge_dir = Path(__file__).parent.parent.parent / "src/listing_agent/rag/knowledge_base"
    return PlatformRetriever(knowledge_dir=str(knowledge_dir), persist_dir=str(tmp_path / "chroma"))


def test_retriever_returns_shopify_rules(retriever):
    rules = retriever.get_rules("shopify", "ceramic mug")
    assert "255" in rules or "title" in rules.lower()
    assert len(rules) > 50


def test_retriever_returns_amazon_rules(retriever):
    rules = retriever.get_rules("amazon", "ceramic mug")
    assert "200" in rules or "bullet" in rules.lower()
    assert len(rules) > 50


def test_retriever_returns_etsy_rules(retriever):
    rules = retriever.get_rules("etsy", "handmade mug")
    assert "140" in rules or "tag" in rules.lower()
    assert len(rules) > 50


def test_retriever_unknown_platform_raises(retriever):
    with pytest.raises(ValueError, match="Unknown platform"):
        retriever.get_rules("tiktok_shop", "mug")
