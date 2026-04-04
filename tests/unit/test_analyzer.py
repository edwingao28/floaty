import pytest
from unittest.mock import MagicMock, patch
from listing_agent.nodes.analyzer import analyze_product
from listing_agent.state import AgentState


@pytest.fixture
def base_state() -> AgentState:
    return {
        "raw_product_data": {"description": "handmade ceramic mug, 12oz, minimalist design, dishwasher safe, made in studio"},
        "target_platforms": ["shopify", "amazon", "etsy"],
    }


def test_analyze_product_returns_state_with_attributes(base_state):
    mock_llm_response = MagicMock()
    mock_llm_response.content = """{
        "name": "Handmade Ceramic Mug",
        "category": "home_and_kitchen",
        "features": ["12oz capacity", "dishwasher safe", "minimalist design"],
        "materials": ["ceramic"],
        "target_audience": "coffee lovers, minimalist home decor enthusiasts",
        "price_range": "mid-range ($20-$45)"
    }"""

    with patch("listing_agent.nodes.analyzer.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response
        mock_get_llm.return_value = mock_llm

        result = analyze_product(base_state)

    assert "product_attributes" in result
    attrs = result["product_attributes"]
    assert attrs["name"] == "Handmade Ceramic Mug"
    assert "ceramic" in attrs["materials"]
    assert len(attrs["features"]) == 3


def test_analyze_product_handles_malformed_json(base_state):
    mock_llm_response = MagicMock()
    mock_llm_response.content = "not valid json"

    with patch("listing_agent.nodes.analyzer.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response
        mock_get_llm.return_value = mock_llm

        result = analyze_product(base_state)

    assert "errors" in result
    assert len(result["errors"]) == 1
    assert "Failed to parse" in result["errors"][0]
