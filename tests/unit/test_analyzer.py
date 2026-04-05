import pytest
from unittest.mock import MagicMock, patch
from listing_agent.nodes.analyzer import analyze_product
from listing_agent.state import AgentState, ProductAttributes


@pytest.fixture
def base_state() -> AgentState:
    return {
        "raw_product_data": {"description": "handmade ceramic mug, 12oz, minimalist design, dishwasher safe, made in studio"},
        "target_platforms": ["shopify", "amazon", "etsy"],
    }


def test_analyze_product_returns_state_with_attributes(base_state):
    content = """{
        "title": "Handmade Ceramic Mug",
        "category": "home_and_kitchen",
        "features": ["12oz capacity", "dishwasher safe", "minimalist design"],
        "materials": ["ceramic"],
        "target_audience": "coffee lovers, minimalist home decor enthusiasts",
        "price_range": "mid-range ($20-$45)",
        "brand": "",
        "keywords": ["ceramic mug", "handmade", "minimalist"]
    }"""

    with patch("listing_agent.nodes.analyzer.invoke_with_fallback", return_value=content):
        result = analyze_product(base_state)

    assert "product_attributes" in result
    attrs = result["product_attributes"]
    assert isinstance(attrs, ProductAttributes)
    assert attrs.title == "Handmade Ceramic Mug"
    assert "ceramic" in attrs.materials
    assert len(attrs.features) == 3
    assert attrs.raw_input == "handmade ceramic mug, 12oz, minimalist design, dishwasher safe, made in studio"


def test_analyze_product_handles_malformed_json(base_state):
    with patch("listing_agent.nodes.analyzer.invoke_with_fallback", return_value="not valid json"):
        result = analyze_product(base_state)

    assert "errors" in result
    assert len(result["errors"]) == 1
    assert "Failed to parse" in result["errors"][0]


def test_analyze_product_handles_llm_exception(base_state):
    with patch("listing_agent.nodes.analyzer.invoke_with_fallback", side_effect=ConnectionError("network error")):
        result = analyze_product(base_state)

    assert "errors" in result
    assert len(result["errors"]) == 1
    assert "Failed to analyze product" in result["errors"][0]


def test_analyze_product_uses_fallback_on_primary_failure(base_state):
    """Falls back to secondary model when primary raises."""
    fallback_content = '{"title":"Test","category":"other","features":[],"keywords":[],"raw_input":"test"}'
    mock_cfg = MagicMock()
    mock_cfg.ANTHROPIC_MODEL = "claude-sonnet-4-6"
    mock_cfg.ANTHROPIC_FALLBACK_MODEL = "claude-haiku-4-5-20251001"
    with patch("listing_agent.nodes._llm.get_config", return_value=mock_cfg), \
         patch("listing_agent.nodes._llm.ChatAnthropic") as MockLLM:
        primary_instance = MagicMock()
        primary_instance.invoke.side_effect = Exception("Rate limit")
        fallback_instance = MagicMock()
        fallback_instance.invoke.return_value = MagicMock(content=fallback_content)
        MockLLM.side_effect = [primary_instance, fallback_instance]

        result = analyze_product(base_state)

    assert "product_attributes" in result
