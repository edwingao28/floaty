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


from unittest.mock import MagicMock, patch

from listing_agent.nodes.researcher import research_platforms
from listing_agent.state import AgentState, ProductAttributes


@pytest.fixture
def state_with_attrs() -> AgentState:
    return {
        "raw_product_data": {"description": "handmade ceramic mug"},
        "target_platforms": ["shopify", "amazon", "etsy"],
        "product_attributes": ProductAttributes(
            title="Handmade Ceramic Mug",
            category="home_and_kitchen",
            features=["12oz", "dishwasher safe"],
            materials=["ceramic"],
            target_audience="coffee lovers",
            price_range="mid-range ($20-$45)",
            keywords=["ceramic", "mug", "handmade"],
            raw_input="handmade ceramic mug",
        ),
    }


def test_research_platforms_populates_rules(state_with_attrs):
    with patch("listing_agent.nodes.researcher.PlatformRetriever") as MockRetriever:
        mock_retriever = MagicMock()
        mock_retriever.get_rules.return_value = "Title max 255 chars. Use keywords."
        MockRetriever.return_value = mock_retriever

        result = research_platforms(state_with_attrs)

    assert "platform_rules" in result
    rules = result["platform_rules"]
    # Should have rules for all 3 platforms
    assert len(rules) == 3
    assert mock_retriever.get_rules.call_count == 3
